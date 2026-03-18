# Dynamic Orchestrator — Stage 1 Design

**Date**: 2026-03-18
**Status**: Approved
**Scope**: Replace static Plan→Execute→Synthesize pipeline with a dynamic main-agent tool-calling loop.

---

## 1. Context and Motivation

The current system uses a static 3-phase pipeline:

```
LLMPlanner.plan()          ← generates full DAG upfront (static)
WorkflowEngine._execute_plan()  ← executes frozen DAG
LLMSynthesizer.synthesize()    ← post-processes results
```

This prevents:
- Dynamic DAG expansion based on intermediate results (e.g. dispatch one task per discovered item)
- Mid-task decision changes if upstream results are unexpected
- Natural orchestration of complex multi-branch workflows

The design goal is to replace this with a **dynamic orchestration loop** where the main agent uses tool calls to dispatch sub-agents, wait for results, and reply — all driven by the model's own decisions.

---

## 2. Architecture Overview

### Components Replaced

| Removed | Replaced By |
|---------|-------------|
| `LLMPlanner` | Main agent model decides via tool calls |
| `WorkflowEngine` | `DynamicOrchestrator` loop |
| `LLMSynthesizer` | Main agent calls `reply_to_user` tool |
| `PlanOutput`, `PlannedTask` Pydantic models | No longer needed |

### Components Preserved

| Kept | Reason |
|------|--------|
| `ResourceBridgeExecutor` | Sub-agent execution logic unchanged |
| `SingleAgentExecutor` | Sub-agent inner loop unchanged |
| `ResourceManager` | Tool registry, skill loading unchanged |
| `Tape` / `TapeStore` | Conversation history unchanged |
| `ExecutionContext`, `TaskContract`, `TaskResult` | Core types unchanged |
| `RunPolicy`, `ToolLease` | Policy types unchanged |

### New File

```
babybot/agent_kernel/dynamic_orchestrator.py
```

### Call Chain (New)

```
OrchestratorAgent.process_task(user_input)
  └─ DynamicOrchestrator.run(goal, context)
       ├─ _build_initial_messages(goal, context)
       ├─ main loop (up to MAX_STEPS=30):
       │    call_model(messages, orchestration_tools)
       │      ↓ tool_calls
       │    dispatch_task(resource_id, description, deps)
       │      └─ asyncio.create_task(ResourceBridgeExecutor.execute(...))
       │    wait_for_tasks(task_ids)
       │      └─ asyncio.gather(*[in_flight[tid] for tid in task_ids])
       │    get_task_result(task_id)
       │      └─ results[task_id].output
       │    reply_to_user(text)
       │      └─ terminates loop, returns FinalResult
       └─ FinalResult
```

---

## 3. `DynamicOrchestrator` Class

```python
class DynamicOrchestrator:
    MAX_STEPS = 30    # max model calls before forced fallback
    MAX_TASKS = 20    # max sub-agent dispatches per session

    def __init__(self, resource_manager: ResourceManager, gateway: OpenAICompatibleGateway): ...

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        in_flight: dict[str, asyncio.Task] = {}   # task_id → asyncio.Task
        results:   dict[str, TaskResult]   = {}   # task_id → TaskResult (completed)
        messages = self._build_initial_messages(goal, context)

        for step in range(self.MAX_STEPS):
            heartbeat.beat()  # check cancellation each step
            response = await self._call_model(messages, context)

            if response.is_final_text():
                return FinalResult(conclusion=response.text)

            for tool_call in response.tool_calls:
                result = await self._dispatch_tool(tool_call, in_flight, results, context)
                if tool_call.name == "reply_to_user":
                    return FinalResult(conclusion=result, task_results=results)
                messages += [assistant_msg, tool_result_msg]

        # Forced fallback: collect completed results
        return self._build_fallback_result(goal, results)
```

**State per `run()` call:**
- `in_flight`: maps `task_id` → running `asyncio.Task`
- `results`: maps `task_id` → completed `TaskResult`
- `messages`: growing list of model conversation messages

---

## 4. Orchestration Tools

All four tools are registered under the `basic_orchestration` tool group. The main agent's `ToolLease` includes only `basic_orchestration` + `channel_{channel_name}`.

### `dispatch_task`

```python
def dispatch_task(
    resource_id: str,     # e.g. "skill.weather-query"
    description: str,     # full task description for the sub-agent
    deps: list[str] = [], # task_ids that must complete before this starts
) -> str:                 # returns generated task_id
```

- Validates `resource_id` against the resource catalog
- If `deps` non-empty: `await asyncio.gather(*[in_flight[d] for d in deps])` before creating task
- Creates `asyncio.Task` running `ResourceBridgeExecutor.execute(task_contract, child_context)`
- Returns `task_id` (e.g. `"task_a1b2c3"`)
- Enforces `MAX_TASKS` limit; returns error string if exceeded

### `wait_for_tasks`

```python
def wait_for_tasks(
    task_ids: list[str],
) -> dict[str, str]:  # {task_id: "succeeded: <output>" | "failed: <error>"}
```

- `await asyncio.gather(*[in_flight[tid] for tid in task_ids])`
- Moves completed tasks from `in_flight` → `results`
- Returns summary strings for all requested tasks

### `get_task_result`

```python
def get_task_result(
    task_id: str,
) -> str:  # result output, "pending", or "not_found"
```

- Non-blocking check of `results[task_id]`
- Returns `"pending"` if task still running

### `reply_to_user`

```python
def reply_to_user(
    text: str,
) -> str:  # signals loop termination
```

- Sets a sentinel flag; orchestration loop terminates after this call
- `text` becomes `FinalResult.conclusion`

---

## 5. Main Agent System Prompt

Three sections, dynamically assembled:

### Part 1: Role and Rules (static)

```
你是任务编排Agent。理解用户请求，动态调度子Agent完成任务，最终向用户回复结果。

编排规则：
1. 简单问题（聊天、知识问答）→ 直接调用 reply_to_user，无需创建子任务
2. 需要工具的任务 → dispatch_task 创建子Agent，wait_for_tasks 等待结果，reply_to_user 回复
3. 可并行的任务 → 同时 dispatch 多个（不设deps），再 wait_for_tasks
4. 有依赖的任务 → 在 deps 中声明依赖，工具层自动等待
5. 禁止虚构执行结果；需要外部信息必须通过 dispatch_task 获取
```

### Part 2: Resource Catalog (dynamic, from `ResourceManager.get_resource_briefs()`)

```
可用资源：
- skill.weather-query: 天气查询 (工具数: 2)
- skill.text-to-image: 文生图 (工具数: 3)
- mcp.gaode-map: 高德地图 (工具数: 5)
```

Only `active=True` resources included. Format: `id: purpose (工具数: N)`.

### Part 3: Conversation History (dynamic, optional, from Tape)

```
对话摘要: <anchor summary>
用户意图: <user_intent>
近期对话:
  user: ...
  assistant: ...
```

Reuses extracted `_build_history_summary(tape)` function (shared with removed `LLMPlanner`). Capped at last 10 messages.

---

## 6. `OrchestratorAgent` Integration

Minimal change: replace `_build_workflow_engine` + `_answer_with_dag`:

```python
async def _answer_with_dag(self, user_input, tape, heartbeat, media_paths, stream_callback):
    orchestrator = DynamicOrchestrator(
        resource_manager=self.resource_manager,
        gateway=self.gateway,
    )
    context = ExecutionContext(
        session_id="orchestrator",
        state={
            "tape": tape,
            "heartbeat": heartbeat,
            "media_paths": media_paths,
            "context_history_tokens": self.config.system.context_history_tokens,
            # stream_callback added if present
        },
    )
    result = await orchestrator.run(goal=user_input, context=context)
    text = result.conclusion or "任务完成，但没有可返回的结果。"
    collected_media = context.state.get("media_paths_collected", [])
    return text, sorted(set(collected_media))
```

All other `process_task`, `_maybe_handoff`, tape management logic remains unchanged.

---

## 7. Error Handling

| Scenario | Handling |
|----------|----------|
| Sub-task exception | Caught inside asyncio.Task; written to `results[task_id]` as `TaskResult(status="failed")` |
| `wait_for_tasks` with failed task | Returns `"failed: <error>"` in result dict; model decides to retry/skip/inform user |
| Model tool call bad JSON | Returns `"error: invalid arguments: ..."` to model; loop continues |
| `MAX_STEPS` reached | Collect all completed results, generate fallback reply, no exception raised |
| Sub-task timeout | Delegated to `ResourceBridgeExecutor` internal timeout (default 300s) |
| Heartbeat cancellation | Checked each step; raises `HeartbeatError`, caught by `OrchestratorAgent.process_task` |
| `MAX_TASKS` exceeded | `dispatch_task` returns error string `"error: task limit reached"`; model informs user |

---

## 8. Testing

New file: `tests/test_dynamic_orchestrator.py`

| Test | Scenario |
|------|----------|
| `test_direct_reply` | Model calls `reply_to_user` without any dispatch |
| `test_single_task` | dispatch one task, wait, reply with result |
| `test_parallel_tasks` | dispatch two tasks without deps, wait both, reply |
| `test_dependent_tasks` | dispatch A, dispatch B with deps=[A], wait B, reply |
| `test_max_steps_fallback` | model never calls reply_to_user; verify fallback after MAX_STEPS |
| `test_failed_task_handling` | sub-task fails; model receives failure; replies with error info |

Deleted: `tests/test_agent_kernel.py` (tests `WorkflowEngine` which no longer exists).

---

## 9. Out of Scope (Stage 1)

- `ask_user` tool (mid-workflow user interaction) — future stage
- Sub-agent progress streaming — future stage
- Resource catalog enrichment (example_tasks, cost_tier) — future stage
- `SubAgentContext` formalization — future stage
