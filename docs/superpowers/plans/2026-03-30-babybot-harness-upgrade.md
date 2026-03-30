# Babybot Harness Upgrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `babybot` from prompt-led orchestration to a harness-led runtime with explicit task contracts, resumable jobs, unified event flow, and machine-readable feedback.

**Architecture:** Add a contract-and-plan layer ahead of orchestration, then route all execution modes through a shared job runtime that emits structured events. Keep channel UX thin: `MessageBus` should render a canonical runtime state machine instead of owning business rules, while `OrchestratorAgent`, interactive sessions, and policy feedback all speak the same task/job vocabulary.

**Tech Stack:** Python, asyncio, sqlite-backed stores already used in `babybot`, pytest

---

### Task 1: Add runtime docs and single-source contracts

**Files:**
- Create: `docs/agent-runtime/interaction-contract.md`
- Create: `docs/agent-runtime/long-running-jobs.md`
- Create: `docs/agent-runtime/debate-and-round-budget.md`
- Create: `docs/agent-runtime/feedback-state-machine.md`
- Modify: `AGENTS.md` (or create if absent)

- [ ] **Step 1: Write the runtime docs as the single source of truth**

Document:

- what a `TaskContract` is
- what a `JobState` is
- which runtime fields are authoritative
- how round budgets and stop conditions map into execution
- which events channels may render

- [ ] **Step 2: Keep `AGENTS.md` as a map, not a rule dump**

Add only:

- repo entry points
- links to the new runtime docs
- rules for when to read each doc

- [ ] **Step 3: Review the docs for overlap and contradictions**

Check:

- `interaction-contract.md` defines the authoritative contract fields
- `debate-and-round-budget.md` references those exact fields
- `feedback-state-machine.md` only renders states defined in `long-running-jobs.md`

### Task 2: Introduce `TaskContract` as the only execution input

**Files:**
- Create: `babybot/task_contract.py`
- Modify: `babybot/agent_kernel/execution_constraints.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_task_contract.py`
- Test: `tests/test_orchestrator_routing.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] **Step 1: Write the failing contract tests**

Add tests that assert:

- “一轮定胜负” becomes `round_budget=1`
- explicit user stop conditions survive planning
- execution defaults are centralized instead of spread across call sites
- runtime refuses to execute when contract and runtime parameters disagree

Suggested skeleton:

```python
def test_contract_keeps_user_round_budget():
    contract = build_task_contract(
        user_input="一轮定胜负，给我最终答案",
        chat_key="feishu:c1",
    )

    assert contract.mode == "debate"
    assert contract.round_budget == 1
    assert contract.termination_rule == "single_round"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/test_task_contract.py tests/test_orchestrator_routing.py -k contract -v`
Expected: FAIL because `TaskContract` does not exist yet

- [ ] **Step 3: Implement the contract model and normalization entry point**

Use a focused dataclass:

```python
@dataclass(frozen=True)
class TaskContract:
    chat_key: str
    goal: str
    mode: str
    deliverable: str
    round_budget: int | None
    termination_rule: str
    allow_clarification: bool
    allowed_tools: tuple[str, ...]
    allowed_agents: tuple[str, ...]
    metadata: dict[str, Any]
```

Implementation requirements:

- `orchestrator.process_task()` builds the contract once
- downstream code reads contract fields, not raw user text
- `execution_constraints.py` becomes an input to contract building, not a parallel path

- [ ] **Step 4: Add a runtime assertion to prevent contract drift**

Add one authoritative helper:

```python
def assert_runtime_matches_contract(contract: TaskContract, *, max_rounds: int | None) -> None:
    ...
```

If contract says `round_budget=1`, runtime cannot silently execute `5`.

- [ ] **Step 5: Run focused tests to verify they pass**

Run: `pytest tests/test_task_contract.py tests/test_orchestrator_routing.py tests/test_dynamic_orchestrator.py -k 'contract or round_budget' -v`
Expected: PASS

### Task 3: Add `ExecutionPlan` so planning and execution are no longer implicit

**Files:**
- Create: `babybot/execution_plan.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/agent_kernel/team.py`
- Test: `tests/test_execution_plan.py`
- Test: `tests/test_agent_team.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] **Step 1: Write the failing plan tests**

Assert that:

- complex requests first become a structured plan
- debate mode records explicit `participants`, `round_budget`, and `stopping_condition`
- single-step tasks bypass debate by plan, not by ad hoc branching

Suggested skeleton:

```python
def test_execution_plan_explicitly_carries_round_budget():
    plan = build_execution_plan(
        TaskContract(
            chat_key="feishu:c1",
            goal="比较两版诗，给出胜者",
            mode="debate",
            deliverable="winner",
            round_budget=1,
            termination_rule="single_round",
            allow_clarification=False,
            allowed_tools=(),
            allowed_agents=("judge_master",),
            metadata={},
        )
    )

    assert plan.round_budget == 1
    assert plan.steps[0].kind == "debate"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/test_execution_plan.py tests/test_agent_team.py -k 'execution_plan or round_budget' -v`
Expected: FAIL because `ExecutionPlan` does not exist yet

- [ ] **Step 3: Implement the plan model**

Use narrow structures:

```python
@dataclass(frozen=True)
class PlanStep:
    step_id: str
    kind: str
    title: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    contract: TaskContract
    steps: tuple[PlanStep, ...]
    round_budget: int | None
    stopping_condition: str
```

- [ ] **Step 4: Make orchestration consume `ExecutionPlan`**

Requirements:

- `OrchestratorAgent` decides once: direct answer, debate, tool workflow, or long-running job
- `TeamRunner` gets `round_budget` from `ExecutionPlan`, not from fallback defaults
- `dynamic_orchestrator` reports plan step IDs in runtime events

- [ ] **Step 5: Run focused tests to verify they pass**

Run: `pytest tests/test_execution_plan.py tests/test_agent_team.py tests/test_dynamic_orchestrator.py -k 'execution_plan or round_budget' -v`
Expected: PASS

### Task 4: Introduce resumable `JobRuntime` for long-running work

**Files:**
- Create: `babybot/runtime_jobs.py`
- Create: `babybot/runtime_job_store.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/message_bus.py`
- Modify: `babybot/config.py`
- Test: `tests/test_runtime_jobs.py`
- Test: `tests/test_message_bus_streaming.py`
- Test: `tests/test_channel_manager_timeout.py`

- [ ] **Step 1: Write the failing job-runtime tests**

Assert that:

- long tasks transition through explicit states
- hard timeout returns a resumable job reference instead of losing all progress
- polling status reads the same persisted job state

Suggested skeleton:

```python
def test_long_running_job_persists_state_transitions(tmp_path):
    store = RuntimeJobStore(tmp_path / "jobs.db")
    job = store.create(chat_key="feishu:c1", goal="long task")

    store.transition(job.job_id, "running", progress_message="开始执行")
    store.transition(job.job_id, "waiting_tool", progress_message="等待外部工具")

    loaded = store.get(job.job_id)

    assert loaded.state == "waiting_tool"
    assert loaded.progress_message == "等待外部工具"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/test_runtime_jobs.py tests/test_channel_manager_timeout.py -k 'job or timeout' -v`
Expected: FAIL because job runtime does not exist yet

- [ ] **Step 3: Implement the job store and runtime model**

Define canonical states only:

- `queued`
- `planning`
- `running`
- `waiting_tool`
- `waiting_user`
- `repairing`
- `completed`
- `failed`
- `cancelled`

Use a narrow store API:

```python
class RuntimeJobStore:
    def create(self, *, chat_key: str, goal: str, plan_id: str = "") -> RuntimeJob: ...
    def transition(self, job_id: str, state: str, **fields: Any) -> RuntimeJob: ...
    def get(self, job_id: str) -> RuntimeJob | None: ...
    def latest_for_chat(self, chat_key: str) -> RuntimeJob | None: ...
```

- [ ] **Step 4: Route long-running work through `JobRuntime`**

Requirements:

- `orchestrator.process_task()` creates a job before long execution starts
- `message_bus` timeout responses include the persisted `job_id`
- follow-up commands can inspect or resume jobs by ID instead of relying on “most recent”

- [ ] **Step 5: Run focused tests to verify they pass**

Run: `pytest tests/test_runtime_jobs.py tests/test_message_bus_streaming.py tests/test_channel_manager_timeout.py -k 'job or timeout or progress' -v`
Expected: PASS

### Task 5: Unify interactive sessions with the normal runtime event flow

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/interactive_sessions/manager.py`
- Modify: `babybot/interactive_sessions/protocols.py`
- Modify: `babybot/interactive_sessions/backends/claude.py`
- Modify: `babybot/context.py`
- Modify: `babybot/memory_store.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`
- Test: `tests/test_interactive_session_manager.py`
- Test: `tests/test_message_bus_streaming.py`

- [ ] **Step 1: Write the failing integration tests**

Add tests that assert:

- session messages preserve `media_paths`
- expired session status is not reported as active
- first message after expiry falls back cleanly instead of being consumed
- interactive sends emit the same runtime events as normal DAG tasks

Suggested skeleton:

```python
@pytest.mark.asyncio
async def test_interactive_session_message_keeps_media_paths():
    agent = make_agent_with_session_manager(active_session=True)

    response = await agent.process_task(
        "看这张图",
        chat_key="feishu:c1",
        media_paths=["/tmp/demo.png"],
    )

    assert response.text == "backend reply"
    assert agent._interactive_sessions.last_media_paths == ["/tmp/demo.png"]
```

- [ ] **Step 2: Run the session-focused tests to verify they fail**

Run: `pytest tests/test_orchestrator_interactive_sessions.py tests/test_interactive_session_manager.py -v`
Expected: FAIL because session routing does not carry media or shared runtime state

- [ ] **Step 3: Extend the interactive protocol so it can participate in the shared runtime**

Change protocol shape from text-only to request/response envelopes:

```python
@dataclass(frozen=True)
class InteractiveRequest:
    text: str
    media_paths: tuple[str, ...]
    job_id: str
    contract_mode: str
```

Manager requirements:

- `has_active_session()` must ignore expired sessions
- `status()` and `summary()` must prune expired sessions first
- `send()` must be able to report `expired` in a structured way so orchestration can fall back

- [ ] **Step 4: Record interactive traffic in tape and memory**

Requirements:

- user session messages append to tape
- backend replies append to tape
- interactive runtime events hit the same callback used by DAG tasks

- [ ] **Step 5: Wrap long interactive sends with heartbeat keep-alive**

Use the existing heartbeat API so session sends cannot be mistaken for idleness.

- [ ] **Step 6: Run session and streaming tests to verify they pass**

Run: `pytest tests/test_orchestrator_interactive_sessions.py tests/test_interactive_session_manager.py tests/test_message_bus_streaming.py -k 'session or runtime' -v`
Expected: PASS

### Task 6: Replace ad hoc text progress with a structured feedback state machine

**Files:**
- Create: `babybot/feedback_events.py`
- Modify: `babybot/message_bus.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Test: `tests/test_feedback_state_machine.py`
- Test: `tests/test_message_bus_streaming.py`

- [ ] **Step 1: Write the failing feedback tests**

Assert that:

- distinct tasks with identical human text do not dedupe incorrectly
- failed events without `error` still surface as failures
- final reply rendering depends on canonical state, not on string comparison

Suggested skeleton:

```python
def test_runtime_feedback_dedupes_by_event_identity_not_text():
    first = RuntimeFeedbackEvent(
        job_id="job-1",
        task_id="task-a",
        state="running",
        message="处理中",
    )
    second = RuntimeFeedbackEvent(
        job_id="job-1",
        task_id="task-b",
        state="running",
        message="处理中",
    )

    assert feedback_dedupe_key(first) != feedback_dedupe_key(second)
```

- [ ] **Step 2: Run the focused feedback tests to verify they fail**

Run: `pytest tests/test_feedback_state_machine.py tests/test_message_bus_streaming.py -k 'feedback or dedupe or failed' -v`
Expected: FAIL because structured feedback events do not exist yet

- [ ] **Step 3: Implement canonical feedback event types**

Recommended shape:

```python
@dataclass(frozen=True)
class RuntimeFeedbackEvent:
    job_id: str
    flow_id: str
    task_id: str
    state: str
    stage: str
    message: str
    error: str
    progress: float | None
```

- [ ] **Step 4: Move all user-facing progress rendering behind one adapter**

Requirements:

- `MessageBus` renders `RuntimeFeedbackEvent` only
- dedupe key includes `job_id`, `task_id`, `stage`, and `state`
- failure rendering does not require non-empty `error`
- final response rendering is distinct from intermediate progress rendering

- [ ] **Step 5: Run focused feedback tests to verify they pass**

Run: `pytest tests/test_feedback_state_machine.py tests/test_message_bus_streaming.py -k 'feedback or dedupe or failed' -v`
Expected: PASS

### Task 7: Make policy feedback target explicit jobs/flows instead of “latest”

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/orchestration_policy_store.py`
- Create: `babybot/runtime_feedback_commands.py`
- Test: `tests/test_orchestrator_policy_feedback.py`
- Test: `tests/test_orchestration_policy_store.py`

- [ ] **Step 1: Write the failing policy feedback tests**

Assert that:

- user can target a specific `job_id` or `flow_id`
- fallback to “latest” is only used when there is exactly one unambiguous candidate
- ambiguous feedback requests return a clear resolution message

Suggested skeleton:

```python
@pytest.mark.asyncio
async def test_policy_feedback_can_target_specific_flow():
    agent = _make_agent()
    agent._recent_flow_ids_by_chat["feishu:c1"] = "flow-new"

    response = await agent.process_task(
        "@policy feedback flow-old bad 轮数失控",
        chat_key="feishu:c1",
    )

    assert "已记录" in response.text
    assert agent._policy_store.feedback_rows[0]["flow_id"] == "flow-old"
```

- [ ] **Step 2: Run the policy feedback tests to verify they fail**

Run: `pytest tests/test_orchestrator_policy_feedback.py tests/test_orchestration_policy_store.py -k 'policy_feedback or flow_id' -v`
Expected: FAIL because explicit targeting is not supported yet

- [ ] **Step 3: Add explicit feedback command parsing**

Support canonical forms:

- `@policy feedback <flow_id> good|bad <reason>`
- `@policy feedback latest good|bad <reason>`
- `@policy inspect <flow_id>`

- [ ] **Step 4: Keep “latest” as a compatibility path, not the primary model**

Requirements:

- if there are multiple recent runs, ask user to target one
- store job/flow references from `JobRuntime` so feedback attaches to the right execution

- [ ] **Step 5: Run policy feedback tests to verify they pass**

Run: `pytest tests/test_orchestrator_policy_feedback.py tests/test_orchestration_policy_store.py -k 'policy_feedback or flow_id' -v`
Expected: PASS

### Task 8: Add governance checks and low-grade entropy cleanup

**Files:**
- Create: `tests/test_agent_runtime_docs.py`
- Create: `tests/test_feedback_contracts.py`
- Modify: `babybot/message_bus.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/interactive_sessions/manager.py`
- Modify: `docs/agent-runtime/*.md`

- [ ] **Step 1: Write failing governance tests**

Assert that:

- runtime docs reference real states only
- feedback renderer handles every canonical job state
- session status and job status stay consistent

- [ ] **Step 2: Run governance tests to verify they fail**

Run: `pytest tests/test_agent_runtime_docs.py tests/test_feedback_contracts.py -v`
Expected: FAIL because governance checks do not exist yet

- [ ] **Step 3: Add a narrow consistency layer**

Examples:

- one exported tuple for valid job states
- one exported tuple for valid feedback states
- doc tests that fail when docs and code drift

- [ ] **Step 4: Add a lightweight cleanup path**

Add one maintenance command or periodic task that can:

- find orphaned jobs older than retention
- find stale interactive sessions
- report flows without matching job IDs

- [ ] **Step 5: Run governance tests to verify they pass**

Run: `pytest tests/test_agent_runtime_docs.py tests/test_feedback_contracts.py -v`
Expected: PASS

### Task 9: Verify the full harness upgrade on touched paths

**Files:**
- Modify: `babybot/task_contract.py`
- Modify: `babybot/execution_plan.py`
- Modify: `babybot/runtime_jobs.py`
- Modify: `babybot/runtime_job_store.py`
- Modify: `babybot/feedback_events.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/message_bus.py`
- Modify: `babybot/interactive_sessions/manager.py`
- Modify: `babybot/interactive_sessions/protocols.py`
- Modify: `babybot/interactive_sessions/backends/claude.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/agent_kernel/team.py`
- Test: `tests/test_task_contract.py`
- Test: `tests/test_execution_plan.py`
- Test: `tests/test_runtime_jobs.py`
- Test: `tests/test_feedback_state_machine.py`
- Test: `tests/test_agent_runtime_docs.py`
- Test: `tests/test_feedback_contracts.py`
- Test: `tests/test_orchestrator_interactive_sessions.py`
- Test: `tests/test_interactive_session_manager.py`
- Test: `tests/test_orchestrator_policy_feedback.py`
- Test: `tests/test_message_bus_streaming.py`
- Test: `tests/test_dynamic_orchestrator.py`
- Test: `tests/test_agent_team.py`

- [ ] **Step 1: Run the focused runtime suite**

Run:

```bash
pytest \
  tests/test_task_contract.py \
  tests/test_execution_plan.py \
  tests/test_runtime_jobs.py \
  tests/test_feedback_state_machine.py \
  tests/test_orchestrator_interactive_sessions.py \
  tests/test_interactive_session_manager.py \
  tests/test_orchestrator_policy_feedback.py \
  tests/test_message_bus_streaming.py \
  tests/test_dynamic_orchestrator.py \
  tests/test_agent_team.py -v
```

Expected: PASS

- [ ] **Step 2: Run lint on touched modules**

Run:

```bash
ruff check \
  babybot/task_contract.py \
  babybot/execution_plan.py \
  babybot/runtime_jobs.py \
  babybot/runtime_job_store.py \
  babybot/feedback_events.py \
  babybot/orchestrator.py \
  babybot/message_bus.py \
  babybot/interactive_sessions \
  babybot/agent_kernel/dynamic_orchestrator.py \
  babybot/agent_kernel/team.py
```

Expected: PASS

- [ ] **Step 3: Run a manual end-to-end smoke checklist**

Validate:

- send “一轮定胜负” and confirm runtime plan says `round_budget=1`
- start an interactive session and send text + media
- let a session expire and confirm the next message falls back cleanly
- trigger a long-running task and confirm a `job_id` is returned with resumable state
- record policy feedback for a specific flow/job

- [ ] **Step 4: Review the diff for scope creep**

Check that the change set is still about:

- contract layer
- plan layer
- job runtime
- unified event flow
- structured feedback

Not:

- unrelated channel rewrites
- model-provider swaps
- speculative UI redesign

- [ ] **Step 5: Commit in narrow slices**

Recommended commit order:

```bash
git commit -m "feat: add task contract and execution plan"
git commit -m "feat: add resumable job runtime"
git commit -m "fix: unify interactive sessions with runtime events"
git commit -m "feat: add structured feedback state machine"
git commit -m "feat: target policy feedback by flow id"
git commit -m "test: add harness governance checks"
```
