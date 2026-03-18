# Dynamic Orchestrator — Stage 1 Design

**Date**: 2026-03-18
**Status**: Approved (v2, post-review)
**Scope**: Replace static Plan→Execute→Synthesize pipeline with a dynamic main-agent tool-calling loop.

---

## 1. Context and Motivation

The current system uses a static 3-phase pipeline:

```
LLMPlanner.plan()               ← generates full DAG upfront (static)
WorkflowEngine._execute_plan()  ← executes frozen DAG
LLMSynthesizer.synthesize()     ← post-processes results
```

This prevents:
- Dynamic DAG expansion based on intermediate results (e.g. dispatch one task per discovered item)
- Mid-task decision changes if upstream results are unexpected
- Natural orchestration of complex multi-branch workflows

The goal is a **dynamic orchestration loop** where the main agent uses tool calls to dispatch sub-agents, wait for results, and reply — all driven by model decisions.

**Deliberate trade-off**: Simple conversational questions (e.g. "hi") now go through a full model call + `reply_to_user`, whereas the previous `LLMPlanner` fast-path short-circuited with a heuristic check. This is an accepted cost for architectural simplicity. If latency becomes a problem, a lightweight pre-filter can be added later.

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

### `_build_history_summary` Extraction

Currently a `@staticmethod` on `LLMPlanner` in `dag_ports.py`. Since `LLMPlanner` is deleted, this function must be **extracted to a module-level function** in `dag_ports.py` before the class is removed. `DynamicOrchestrator` will import and call it directly:

```python
# dag_ports.py (module level, not on LLMPlanner)
def build_history_summary(tape: "Tape | None") -> str: ...
```

### Call Chain (New)

```
OrchestratorAgent.process_task(user_input)
  └─ heartbeat.watch(DynamicOrchestrator.run(goal, context))
       ├─ _build_initial_messages(goal, context)
       ├─ main loop (up to MAX_STEPS=30):
       │    heartbeat.beat()
       │    async with heartbeat.keep_alive():
       │        response = await call_model(messages, orchestration_tools)
       │    append assistant message (with all tool_calls)
       │    for each tool_call:
       │        result = await _dispatch_tool(...)
       │        append tool_result message
       │    if reply_to_user was called → return FinalResult
       └─ FinalResult (or fallback after MAX_STEPS)
```

---

## 3. `DynamicOrchestrator` Class

```python
class DynamicOrchestrator:
    MAX_STEPS = 30    # max model calls before forced fallback
    MAX_TASKS = 20    # max sub-agent dispatches per session

    def __init__(self, resource_manager: ResourceManager, gateway: OpenAICompatibleGateway):
        self._rm = resource_manager
        self._gateway = gateway
        self._bridge = ResourceBridgeExecutor(resource_manager, gateway)  # reused per dispatch

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        in_flight: dict[str, asyncio.Task] = {}   # task_id → asyncio.Task (running)
        results:   dict[str, TaskResult]   = {}   # task_id → TaskResult (completed)
        reply_text: str | None = None
        messages = self._build_initial_messages(goal, context)

        for step in range(self.MAX_STEPS):
            heartbeat.beat()
            async with heartbeat.keep_alive():
                response = await self._call_model(messages, context)

            if response.is_final_text():
                # Model responded with plain text (no tool calls)
                return FinalResult(conclusion=response.text)

            # Append assistant message ONCE (contains all tool_calls)
            messages.append(response.as_assistant_message())

            # Process all tool calls; collect result messages
            for tool_call in response.tool_calls:
                result_text = await self._dispatch_tool(tool_call, in_flight, results, context)
                messages.append(make_tool_result_message(tool_call.call_id, result_text))
                if tool_call.name == "reply_to_user":
                    reply_text = result_text  # mark for termination

            if reply_text is not None:
                return FinalResult(conclusion=reply_text, task_results=results)

        # Forced fallback after MAX_STEPS
        return self._build_fallback_result(goal, results)

    def _build_fallback_result(self, goal: str, results: dict[str, TaskResult]) -> FinalResult:
        """Collect completed results and return a best-effort reply."""
        parts = [f"（编排步数已达上限，以下为已完成的任务结果）"]
        for task_id, r in results.items():
            if r.status == "succeeded":
                parts.append(f"- {task_id}: {r.output or '完成'}")
            else:
                parts.append(f"- {task_id}: 失败 — {r.error}")
        return FinalResult(conclusion="\n".join(parts), task_results=results)
```

**`reply_to_user` termination**: The flag is set during tool processing and checked after the full loop. This means if the model calls `reply_to_user` alongside other tool calls, all tool calls still complete before the loop exits. The system prompt instructs the model that `reply_to_user` should be its only and final tool call.

---

## 4. Orchestration Tools

All four tools are registered under the `basic_orchestration` tool group. The main agent's `ToolLease` includes only `basic_orchestration` + `channel_{channel_name}`.

### `dispatch_task` — Non-Blocking

```python
def dispatch_task(
    resource_id: str,     # e.g. "skill.weather-query"
    description: str,     # full task description for the sub-agent
    deps: list[str] = [], # task_ids that must complete before this starts
) -> str:                 # returns task_id immediately (never blocks)
```

**Implementation**:

`dispatch_task` is always non-blocking. It creates an `asyncio.Task` that internally awaits deps before running the sub-agent:

```python
async def _run_with_deps(deps, task_contract, child_context):
    # _run_with_deps is defined inline per dispatch_task call (one closure per call).
    if deps:
        # Wait for deps inside the background task — does NOT block the main loop.
        dep_tasks = [in_flight[d] for d in deps if d in in_flight]
        # Deps already in `results` are already done — no need to await.
        if dep_tasks:
            await asyncio.gather(*dep_tasks)
    # Call as instance method via self._bridge (created in __init__).
    result = await self._bridge.execute(task_contract, child_context)
    # Store result and remove from in_flight BEFORE the coroutine exits.
    # asyncio.gather() resumes only after this coroutine returns, so wait_for_tasks
    # can safely read `results` after gather completes.
    results[task_id] = result
    del in_flight[task_id]

task = asyncio.create_task(_run_with_deps(deps, task_contract, child_context))
in_flight[task_id] = task
return task_id  # returned immediately to model
```

**Validation**:
- `resource_id` not in catalog or is inactive → returns `"error: resource not available: <resource_id>"`
  (Note: `ResourceManager.resolve_resource_scope` returns `None` for both cases;
  implementation does not need to distinguish them — a single error string suffices)
- `deps` contains unknown task_id → returns `"error: unknown dep task_id: <dep>"`
- `MAX_TASKS` exceeded → returns `"error: task limit reached (max 20)"`

### `wait_for_tasks` — Blocking by design

```python
def wait_for_tasks(
    task_ids: list[str],
) -> dict[str, str]:  # {task_id: "succeeded: <output>" | "failed: <error>" | "not_found"}
```

**Lookup order for each `task_id`**:
1. Already in `results` (completed) → return immediately with result
2. In `in_flight` (running) → `await` the asyncio.Task
3. Neither → entry returns `"not_found: <task_id>"`

```python
awaitables = []
for tid in task_ids:
    if tid in results:
        pass  # already done
    elif tid in in_flight:
        awaitables.append(in_flight[tid])
    # else: not_found, handled in result assembly

if awaitables:
    await asyncio.gather(*awaitables, return_exceptions=True)

# Assemble result dict
out = {}
for tid in task_ids:
    if tid in results:
        r = results[tid]
        out[tid] = f"succeeded: {r.output}" if r.status == "succeeded" else f"failed: {r.error}"
    else:
        out[tid] = f"not_found: {tid}"
return out
```

### `get_task_result` — Non-Blocking

```python
def get_task_result(task_id: str) -> str:
```

**Lookup order**:
1. In `results` → return `"succeeded: <output>"` or `"failed: <error>"`
2. In `in_flight` → return `"pending"`
3. Neither → return `"not_found: <task_id>"`

### `reply_to_user` — Terminates loop

```python
def reply_to_user(text: str) -> str:
    """向用户发送最终回复。调用后编排循环结束。
    此工具应作为最后一个工具调用单独使用，不与其他工具混用。
    """
    return text  # stored as reply_text; loop exits after all tool calls processed
```

---

## 5. Main Agent System Prompt

Three sections, dynamically assembled:

### Part 1: Role and Rules (static)

```
你是任务编排Agent。理解用户请求，动态调度子Agent完成任务，最终向用户回复结果。

编排规则：
1. 简单问题（聊天、知识问答）→ 直接调用 reply_to_user，无需创建子任务
2. 需要工具的任务 → dispatch_task 创建子Agent，wait_for_tasks 等待结果，reply_to_user 回复
3. 可并行的任务 → 同时 dispatch 多个（不设deps），再 wait_for_tasks 全部等待
4. 有依赖的任务 → 在 deps 中声明依赖，任务内部自动等待前置任务完成
5. 拿到结果后 → 调用 reply_to_user 汇总并回复用户，reply_to_user 必须单独调用且为最后一步
6. 禁止虚构执行结果；需要外部信息必须通过 dispatch_task 获取
```

### Part 2: Resource Catalog (dynamic)

Source: `ResourceManager.get_resource_briefs()`, filtered to `active=True`.

Format:
```
可用资源：
- skill.weather-query: 天气查询 (工具数: 2)
- skill.text-to-image: 文生图 (工具数: 3)
- mcp.gaode-map: 高德地图 (工具数: 5)
```

### Part 3: Conversation History (dynamic, optional)

Source: `build_history_summary(tape)` (extracted module-level function). Capped at last 10 messages.

```
对话摘要: <anchor summary>
用户意图: <user_intent>
近期对话:
  user: ...
  assistant: ...
```

---

## 6. Heartbeat Integration

`DynamicOrchestrator.run()` is NOT responsible for heartbeat setup. The caller (`OrchestratorAgent._answer_with_dag`) wraps the call:

```python
result = await heartbeat.watch(
    orchestrator.run(goal=user_input, context=context),
    hard_timeout=self.config.system.agent_hard_timeout_s,  # optional
)
```

Inside `run()`:
- `heartbeat.beat()` is called at the start of each step
- `async with heartbeat.keep_alive()` wraps each model call (which may take 10–60s)
- If idle timeout is exceeded, `heartbeat.watch()` raises `asyncio.TimeoutError`, caught by `OrchestratorAgent.process_task`

---

## 7. `OrchestratorAgent` Integration

Minimal change: replace `_build_workflow_engine` with `DynamicOrchestrator` in `_answer_with_dag`:

```python
async def _answer_with_dag(self, user_input, tape, heartbeat, media_paths, stream_callback):
    orchestrator = DynamicOrchestrator(
        resource_manager=self.resource_manager,
        gateway=self.gateway,
    )
    context = ExecutionContext(
        session_id="orchestrator",
        state={k: v for k, v in [
            ("tape", tape),
            ("tape_store", self.tape_store if tape else None),  # required by sub-agents for BM25 recall
            ("heartbeat", heartbeat),
            ("media_paths", media_paths),
            ("context_history_tokens", self.config.system.context_history_tokens),
            # stream_callback intentionally NOT propagated to sub-agent contexts
            # (sub-agents run concurrently; sharing a stream callback would interleave output)
        ] if v is not None},
    )
    result = await heartbeat.watch(
        orchestrator.run(goal=user_input, context=context),
        hard_timeout=self.config.system.agent_hard_timeout_s,  # optional hard ceiling
    )
    text = result.conclusion or "任务完成，但没有可返回的结果。"
    collected_media = context.state.get("media_paths_collected", [])
    return text, sorted(set(collected_media))
```

**`stream_callback`**: Not propagated to sub-agent child contexts. Multiple concurrent sub-agents cannot share a single stream callback without interleaving output. Main agent replies are delivered as complete text via `reply_to_user`. Streaming support for sub-agents is deferred to a future stage.

All other `process_task`, `_maybe_handoff`, tape management logic remains unchanged.

---

## 8. Error Handling

| Scenario | Handling |
|----------|----------|
| Sub-task exception | Caught inside `asyncio.Task`; written to `results[task_id]` as `TaskResult(status="failed", error=str(exc))` |
| `wait_for_tasks` with completed task | Checked in `results` first; no `await` needed |
| `wait_for_tasks` with unknown task_id | Returns `"not_found: <task_id>"` in result dict |
| `wait_for_tasks` with failed task | Returns `"failed: <error>"` in result dict; model decides to retry/skip/inform user |
| `get_task_result` for in-flight task | Returns `"pending"` (checks `in_flight` dict) |
| `get_task_result` for unknown task_id | Returns `"not_found: <task_id>"` |
| Model tool call bad JSON | Returns `"error: invalid arguments: ..."` to model; loop continues |
| `dispatch_task` unknown/inactive resource | Returns `"error: resource not available: <resource_id>"` (single string for both cases; `resolve_resource_scope` returns `None` for both) |
| `dispatch_task` unknown dep | Returns `"error: unknown dep task_id: ..."` |
| `MAX_TASKS` exceeded | Returns `"error: task limit reached (max 20)"` |
| `MAX_STEPS` reached | `_build_fallback_result` collects completed results, returns best-effort reply |
| `reply_to_user` + other tools in same response | All tools execute; loop exits after full batch; system prompt instructs single final call |
| Heartbeat idle timeout | `heartbeat.watch()` raises `asyncio.TimeoutError`; caught by `OrchestratorAgent.process_task` |

---

## 9. Code Cleanup

### Deletions

| File / Symbol | Reason |
|---------------|--------|
| `LLMPlanner` class | Replaced by main agent model decisions |
| `LLMSynthesizer` class | Replaced by `reply_to_user` |
| `PlanOutput`, `PlannedTask` Pydantic models | No longer needed |
| `WorkflowEngine` class | Replaced by `DynamicOrchestrator` |
| `OrchestratorAgent._build_workflow_engine()` | Replaced inline |

### Modified

| Symbol | Change |
|--------|--------|
| `LLMPlanner._build_history_summary` | Extracted to module-level `build_history_summary(tape)` in `dag_ports.py` |
| `OrchestratorAgent._answer_with_dag` | Uses `DynamicOrchestrator` instead of `WorkflowEngine` |
| `agent_kernel/__init__.py` | Update exports: remove `WorkflowEngine`, `LLMPlanner`, `LLMSynthesizer`; add `DynamicOrchestrator` |

### Preserved

`ResourceBridgeExecutor`, `SingleAgentExecutor`, `ResourceManager`, `Tape`/`TapeStore`, all type definitions in `types.py`.

---

## 10. Testing

### New: `tests/test_dynamic_orchestrator.py`

Uses a `DummyGateway` that returns scripted `ModelResponse` objects (tool calls or final text), and a `DummyResourceBridgeExecutor` that returns preset `TaskResult` values.

| Test | Scenario |
|------|----------|
| `test_direct_reply` | Model calls `reply_to_user` without any dispatch |
| `test_single_task` | dispatch → wait → reply with result |
| `test_parallel_tasks` | dispatch A, dispatch B (no deps), wait both, reply |
| `test_dependent_tasks` | dispatch A, dispatch B (deps=[A]), wait B (B auto-waits A internally), reply |
| `test_max_steps_fallback` | Model never calls reply_to_user; verify fallback text after MAX_STEPS |
| `test_failed_task_handling` | Sub-task fails; wait_for_tasks returns failure; model receives and replies with error info |
| `test_unknown_resource` | dispatch_task with invalid resource_id; verify error string returned to model |
| `test_unknown_task_id_in_wait` | wait_for_tasks with unknown task_id; verify `"not_found"` in result |

### Deleted

`tests/test_agent_kernel.py` — tests `WorkflowEngine`, `LLMPlanner`, and the static DAG pipeline. Cycle detection and deadlock protection are no longer structural guarantees; `MAX_TASKS` (20) and `MAX_STEPS` (30) serve as safety bounds instead. The model is responsible for not creating circular dependencies.

### Unchanged

`tests/test_resource_skills.py` — tests `ResourceManager`; unaffected.

---

## 11. Out of Scope (Stage 1)

- `ask_user` tool (mid-workflow user interaction)
- Sub-agent progress streaming / `stream_callback` propagation
- Resource catalog enrichment (`example_tasks`, `cost_tier`, `avg_duration`)
- `SubAgentContext` formalization
- Heuristic pre-filter to avoid model call for trivial messages
