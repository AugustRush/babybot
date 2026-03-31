# Conservative Orchestration Policy Learning Implementation Plan

**Status:** Completed. The conservative policy-learning foundation landed and was later extended by `/Users/shike/Desktop/babybot/docs/superpowers/plans/2026-03-27-policy-learning-upgrade.md` and `/Users/shike/Desktop/babybot/docs/superpowers/plans/2026-03-30-lightweight-routing-learning.md`.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BabyBot improve task decomposition and scheduling over time by learning conservative orchestration policies from execution outcomes and explicit user feedback.

**Architecture:** Keep the current `MessageBus -> OrchestratorAgent -> DynamicOrchestrator -> ResourceManager` control plane intact. Add a separate policy-learning layer that records orchestration decisions, computes rewards from execution outcomes, and feeds conservative strategy hints back into planning and worker-dispatch decisions without retraining the underlying LLM.

**Tech Stack:** Python 3.10+, `sqlite3`, dataclasses, existing `DynamicOrchestrator`, `HybridMemoryStore`, `pytest`

---

### Task 1: Add Config Knobs For Conservative Policy Learning

**Files:**
- Modify: `babybot/config.py`
- Modify: `config.json.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
def test_policy_learning_config_fields_load_from_system(tmp_path, monkeypatch):
    monkeypatch.setenv("BABYBOT_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "model": {"api_key": "test"},
                "system": {
                    "policy_learning_enabled": True,
                    "policy_learning_min_samples": 5,
                    "policy_learning_explore_ratio": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(str(config_path))

    assert cfg.system.policy_learning_enabled is True
    assert cfg.system.policy_learning_min_samples == 5
    assert cfg.system.policy_learning_explore_ratio == 0.05
```

- [ ] **Step 2: Run the focused config test to verify it fails**

Run: `uv run pytest -q tests/test_config.py -k policy_learning`
Expected: FAIL because `SystemConfig` does not expose policy-learning fields yet

- [ ] **Step 3: Implement the minimal config fields**

```python
@dataclass
class SystemConfig:
    ...
    policy_learning_enabled: bool = False
    policy_learning_min_samples: int = 8
    policy_learning_explore_ratio: float = 0.05
```

Also thread the fields through:

- `Config.__init__`
- `to_dict()`
- bootstrap config template
- `config.json.example`

- [ ] **Step 4: Run the focused config test**

Run: `uv run pytest -q tests/test_config.py -k policy_learning`
Expected: PASS

- [ ] **Step 5: Run the full config suite**

Run: `uv run pytest -q tests/test_config.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/config.py config.json.example tests/test_config.py
git commit -m "feat: add orchestration policy learning config"
```

### Task 2: Add A Dedicated Policy Store With Conservative-RL Data Model

**Files:**
- Create: `babybot/orchestration_policy_store.py`
- Test: `tests/test_orchestration_policy_store.py`
- Modify: `babybot/memory_store.py`

- [ ] **Step 1: Write the failing policy-store tests**

```python
def test_policy_store_persists_decisions_and_feedback(tmp_path):
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
        decision_kind="decomposition",
        action_name="analyze_then_execute",
        state_features={"task_shape": "multi_step"},
    )
    store.record_feedback(
        flow_id="flow-1",
        rating="good",
        reason="拆分合理",
    )

    row = store.latest_feedback("flow-1")

    assert row["rating"] == "good"
```

```python
def test_policy_store_enables_wal_and_busy_timeout(tmp_path):
    store = OrchestrationPolicyStore(tmp_path / "policy.db")

    assert store.pragma("journal_mode").lower() == "wal"
    assert int(store.pragma("busy_timeout")) >= 3000
```

- [ ] **Step 2: Run the focused policy-store tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py`
Expected: FAIL because the store does not exist yet

- [ ] **Step 3: Implement the minimal store**

Create tables for:

- `policy_decisions`
- `policy_outcomes`
- `policy_feedback`

Suggested minimal schema:

```sql
CREATE TABLE policy_decisions (
  id INTEGER PRIMARY KEY,
  flow_id TEXT NOT NULL,
  chat_key TEXT NOT NULL,
  decision_kind TEXT NOT NULL,
  action_name TEXT NOT NULL,
  state_features_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

Implementation requirements:

- enable WAL on startup
- set `busy_timeout`
- expose small focused methods (`record_decision`, `record_outcome`, `record_feedback`, `summarize_action_stats`)
- keep this store separate from `HybridMemoryStore`

Also fix the lifecycle logging bug in `babybot/memory_store.py` by defining a module-level logger before `close()` uses it.

- [ ] **Step 4: Run the focused store tests**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py`
Expected: PASS

- [ ] **Step 5: Run the memory-store suite**

Run: `uv run pytest -q tests/test_memory_store.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/orchestration_policy_store.py babybot/memory_store.py tests/test_orchestration_policy_store.py
git commit -m "feat: add orchestration policy store"
```

### Task 3: Capture Orchestration Decisions And Outcomes Without Changing The Control Plane

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Create: `babybot/orchestration_policy_types.py`
- Test: `tests/test_orchestration_policy_capture.py`

- [ ] **Step 1: Write the failing capture tests**

```python
def test_orchestrator_records_flow_level_outcome_on_success():
    ...
    assert store.recorded_outcomes[0]["final_status"] == "succeeded"
```

```python
def test_dynamic_orchestrator_records_dispatch_and_wait_events():
    ...
    assert store.recorded_decisions[0]["decision_kind"] == "scheduling"
```

- [ ] **Step 2: Run the focused capture tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy_capture.py`
Expected: FAIL because no policy capture hooks exist yet

- [ ] **Step 3: Implement minimal capture hooks**

Create focused data types:

```python
@dataclass
class PolicyDecisionRecord:
    flow_id: str
    chat_key: str
    decision_kind: str
    action_name: str
    state_features: dict[str, Any]
```

Record at least:

- decomposition recommendation chosen before orchestration starts
- scheduling mode observed when tasks are dispatched
- final outcome after `reply_to_user` or failure
- retry / replan count inferred from runtime events

Integration rules:

- `OrchestratorAgent` owns the policy store lifecycle
- `DynamicOrchestrator` emits structured decision/outcome events; it does not own persistence
- if policy learning is disabled, hooks should be a no-op

- [ ] **Step 4: Run the focused capture tests**

Run: `uv run pytest -q tests/test_orchestration_policy_capture.py`
Expected: PASS

- [ ] **Step 5: Run orchestrator regression tests**

Run: `uv run pytest -q tests/test_orchestrator_routing.py tests/test_runtime_refactor_event_bus.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/orchestrator.py babybot/agent_kernel/dynamic_orchestrator.py babybot/orchestration_policy_types.py tests/test_orchestration_policy_capture.py
git commit -m "feat: capture orchestration decisions and outcomes"
```

### Task 4: Add Conservative Decomposition Policy Selection

**Files:**
- Create: `babybot/orchestration_policy.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Test: `tests/test_orchestration_policy.py`

- [ ] **Step 1: Write the failing decomposition-policy tests**

```python
def test_policy_prefers_historically_safer_decomposition_action():
    store = FakePolicyStore(
        stats={
            "analyze_then_execute": {"mean_reward": 0.91, "samples": 12},
            "direct_execute": {"mean_reward": 0.42, "samples": 12},
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_decomposition(features={"task_shape": "multi_step"})

    assert action.name == "analyze_then_execute"
```

```python
def test_policy_falls_back_to_safe_default_when_data_is_sparse():
    selector = ConservativePolicySelector(FakePolicyStore(stats={}), min_samples=8)

    action = selector.choose_decomposition(features={"task_shape": "unknown"})

    assert action.name == "analyze_then_execute"
```

- [ ] **Step 2: Run the focused policy tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy.py -k decomposition`
Expected: FAIL because the selector does not exist yet

- [ ] **Step 3: Implement the minimal conservative selector**

Support a small fixed action set:

- `direct_execute`
- `analyze_then_execute`
- `retrieve_then_execute`
- `verify_before_finish`

Scoring rules:

- use empirical reward only when `samples >= min_samples`
- otherwise choose safe default
- add a conservative penalty for high retry / failure rate

Feed the chosen action into orchestration by appending a short planning hint to the system prompt or execution context rather than rewriting the whole planner.

- [ ] **Step 4: Run the focused policy tests**

Run: `uv run pytest -q tests/test_orchestration_policy.py -k decomposition`
Expected: PASS

- [ ] **Step 5: Run dynamic-orchestrator regression tests**

Run: `uv run pytest -q tests/test_dynamic_orchestrator.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/orchestration_policy.py babybot/orchestrator.py babybot/agent_kernel/dynamic_orchestrator.py tests/test_orchestration_policy.py
git commit -m "feat: add conservative decomposition policy"
```

### Task 5: Add Conservative Scheduling Policy Selection And Worker Gating

**Files:**
- Modify: `babybot/orchestration_policy.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/builtin_tools/workers.py`
- Test: `tests/test_orchestration_policy.py`
- Test: `tests/test_builtin_tools.py`

- [ ] **Step 1: Write the failing scheduling-policy tests**

```python
def test_policy_prefers_serial_execution_when_parallel_history_is_risky():
    store = FakePolicyStore(
        stats={
            "serial": {"mean_reward": 0.88, "samples": 10},
            "bounded_parallel": {"mean_reward": 0.34, "samples": 10},
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_scheduling(features={"independent_subtasks": 3})

    assert action.name == "serial"
```

```python
def test_worker_tool_is_blocked_when_policy_marks_worker_usage_high_risk():
    ...
    assert "policy denied" in result.lower()
```

- [ ] **Step 2: Run the focused scheduling tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy.py -k scheduling tests/test_builtin_tools.py -k worker`
Expected: FAIL because no scheduling selector or worker gating exists yet

- [ ] **Step 3: Implement the minimal scheduling selector**

Support actions:

- `serial`
- `bounded_parallel`
- `allow_worker`
- `deny_worker`

Integration requirements:

- `DynamicOrchestrator` receives the scheduling recommendation in context and exposes it in prompt hints
- `create_worker` / `dispatch_workers` respect the conservative gate
- default is serial / deny-worker when data is sparse or task shape is unclear

- [ ] **Step 4: Run the focused scheduling tests**

Run: `uv run pytest -q tests/test_orchestration_policy.py -k scheduling tests/test_builtin_tools.py -k worker`
Expected: PASS

- [ ] **Step 5: Run worker/runtime regression tests**

Run: `uv run pytest -q tests/test_worker.py tests/test_resource_skills.py -k worker`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/orchestration_policy.py babybot/agent_kernel/dynamic_orchestrator.py babybot/builtin_tools/workers.py tests/test_orchestration_policy.py tests/test_builtin_tools.py
git commit -m "feat: add conservative scheduling policy"
```

### Task 6: Add Human Feedback Ingestion For Policy Correction

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `README.md`
- Test: `tests/test_orchestrator_policy_feedback.py`

- [ ] **Step 1: Write the failing feedback tests**

```python
@pytest.mark.asyncio
async def test_policy_feedback_command_records_explicit_user_rating():
    response = await agent.process_task(
        "@policy feedback good 拆分合理",
        chat_key="feishu:c1",
    )

    assert "已记录" in response.text
    assert store.feedback_rows[0]["rating"] == "good"
```

- [ ] **Step 2: Run the focused feedback test to verify it fails**

Run: `uv run pytest -q tests/test_orchestrator_policy_feedback.py`
Expected: FAIL because no feedback command exists yet

- [ ] **Step 3: Implement the minimal feedback command**

Add a narrow control command:

- `@policy feedback good <reason>`
- `@policy feedback bad <reason>`

Rules:

- attach feedback to the latest flow for that `chat_key`
- do not expose this command to normal conversation logic
- if there is no recent flow, return a clear error

- [ ] **Step 4: Run the focused feedback tests**

Run: `uv run pytest -q tests/test_orchestrator_policy_feedback.py`
Expected: PASS

- [ ] **Step 5: Run orchestrator regression tests**

Run: `uv run pytest -q tests/test_orchestrator_interactive_sessions.py tests/test_orchestrator_routing.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add babybot/orchestrator.py README.md tests/test_orchestrator_policy_feedback.py
git commit -m "feat: add orchestration policy feedback command"
```

### Task 7: Final Verification And Operator Docs

**Files:**
- Modify: `README.md`
- No new production files expected

- [ ] **Step 1: Document the learning loop**

Add concise operator documentation covering:

- what signals are collected automatically
- how conservative policy selection works
- how to provide `@policy feedback`
- how to inspect stored policy decisions

- [ ] **Step 2: Run the targeted full policy-learning suite**

Run: `uv run pytest -q tests/test_config.py tests/test_memory_store.py tests/test_orchestration_policy_store.py tests/test_orchestration_policy_capture.py tests/test_orchestration_policy.py tests/test_orchestrator_policy_feedback.py tests/test_dynamic_orchestrator.py tests/test_worker.py`
Expected: PASS

- [ ] **Step 3: Run the full default test suite**

Run: `uv run pytest -q`
Expected: PASS except any intentionally opt-in integration cases outside the default path

- [ ] **Step 4: Review the diff**

Run: `git diff --stat`
Expected: only the planned files changed

- [ ] **Step 5: Commit**

```bash
git add README.md babybot tests
git commit -m "docs: describe orchestration policy learning"
```
