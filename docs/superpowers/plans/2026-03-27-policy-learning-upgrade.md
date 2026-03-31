# Policy Learning Upgrade Implementation Plan

**Status:** Completed. The richer contextual policy stats and selector upgrade landed, then routing/reflection work continued in `/Users/shike/Desktop/babybot/docs/superpowers/plans/2026-03-30-lightweight-routing-learning.md`.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade BabyBot's orchestration policy layer from simple global reward sorting to a conservative contextual policy that uses richer risk stats, human feedback shaping, bucketed action history, and lower-confidence action selection.

**Architecture:** Keep the existing policy capture pipeline intact. Extend `OrchestrationPolicyStore` so it can aggregate richer per-action metrics and bucket-specific summaries, then upgrade `ConservativePolicySelector` to consume those summaries via a conservative contextual bandit score. Continue feeding decisions back into prompt hints and worker gating without changing the core control plane.

**Tech Stack:** Python 3.10+, sqlite3, dataclasses, pytest

---

### Task 1: Add Richer Policy Outcome Aggregation

**Files:**
- Modify: `babybot/orchestration_policy_store.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_orchestration_policy_store.py`
- Test: `tests/test_orchestration_policy_capture.py`

- [ ] **Step 1: Write the failing store test for risk metrics**

```python
def test_policy_store_summarizes_risk_metrics_by_action(tmp_path):
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="serial_dispatch",
        state_features={"task_shape": "multi_step"},
    )
    store.record_outcome(
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="failed",
        reward=-1.0,
        outcome={
            "retry_count": 2,
            "dead_letter_count": 1,
            "stalled_count": 0,
        },
    )

    stats = store.summarize_action_stats(decision_kind="scheduling")

    assert stats["serial_dispatch"]["failure_rate"] == 1.0
    assert stats["serial_dispatch"]["retry_rate"] == 2.0
    assert stats["serial_dispatch"]["dead_letter_rate"] == 1.0
```

- [ ] **Step 2: Write the failing capture test for persisted outcome counts**

```python
def test_orchestrator_records_retry_counts_in_policy_outcome():
    ...
    assert store.recorded_outcomes[0]["outcome"]["retry_count"] == 1
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy_capture.py -k policy`
Expected: FAIL because the store does not aggregate risk metrics yet

- [ ] **Step 4: Implement minimal aggregation**

Extend store summaries with:

- `failure_rate`
- `retry_rate`
- `dead_letter_rate`
- `stalled_rate`
- `success_rate`

Also persist retry/dead-letter/stalled counts from `OrchestratorAgent._answer_with_dag()` into `record_outcome()`.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy_capture.py -k policy`
Expected: PASS

### Task 2: Blend Human Feedback Into Action Scoring

**Files:**
- Modify: `babybot/orchestration_policy_store.py`
- Modify: `babybot/orchestration_policy.py`
- Test: `tests/test_orchestration_policy_store.py`
- Test: `tests/test_orchestration_policy.py`

- [ ] **Step 1: Write the failing store test for feedback aggregation**

```python
def test_policy_store_includes_feedback_score_in_action_summary(tmp_path):
    ...
    assert stats["serial_dispatch"]["feedback_score"] > 0
```

- [ ] **Step 2: Write the failing selector test for feedback shaping**

```python
def test_policy_prefers_action_with_stronger_positive_feedback():
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {"mean_reward": 0.7, "samples": 12, "feedback_score": 0.25},
                "bounded_parallel": {"mean_reward": 0.72, "samples": 12, "feedback_score": -0.2},
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy.py -k feedback`
Expected: FAIL because feedback is stored but not used in summaries or scoring

- [ ] **Step 4: Implement minimal feedback shaping**

Add:

- `feedback_good_count`
- `feedback_bad_count`
- `feedback_score`

Blend feedback into selector scoring conservatively instead of overriding reward.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy.py -k feedback`
Expected: PASS

### Task 3: Add Contextual Buckets For Policy Stats

**Files:**
- Modify: `babybot/orchestration_policy_store.py`
- Modify: `babybot/orchestration_policy.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_orchestration_policy_store.py`
- Test: `tests/test_orchestration_policy.py`

- [ ] **Step 1: Write the failing store test for bucket filtering**

```python
def test_policy_store_filters_action_stats_by_bucket(tmp_path):
    ...
    stats = store.summarize_action_stats(
        decision_kind="decomposition",
        state_bucket="task_shape=multi_step|has_media=0|subtasks=1",
    )

    assert set(stats) == {"analyze_then_execute"}
```

- [ ] **Step 2: Write the failing selector test for bucket-specific choice**

```python
def test_policy_uses_bucket_specific_history_before_global_history():
    ...
    assert action.name == "retrieve_then_execute"
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy.py -k bucket`
Expected: FAIL because the selector only looks at global stats

- [ ] **Step 4: Implement contextual buckets**

Bucket on a small stable feature set:

- `task_shape`
- `has_media`
- `independent_subtasks` (bucketed as `1`, `2`, `3plus`)

Prefer bucket stats; if absent, fall back to global stats; then fall back to safe default.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy.py -k bucket`
Expected: PASS

### Task 4: Upgrade To Conservative Contextual Bandit Selection

**Files:**
- Modify: `babybot/orchestration_policy.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_orchestration_policy.py`
- Test: `tests/test_builtin_tools.py`

- [ ] **Step 1: Write the failing selector test for confidence-aware ranking**

```python
def test_policy_prefers_stabler_action_when_high_reward_action_is_under_sampled():
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {"mean_reward": 0.71, "samples": 20, "feedback_score": 0.0},
                "bounded_parallel": {"mean_reward": 0.9, "samples": 8, "feedback_score": 0.0},
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8, explore_ratio=0.1)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"
```

- [ ] **Step 2: Write the failing regression test for safe worker gating fallback**

```python
def test_worker_policy_remains_allow_when_learning_is_disabled():
    ...
    assert payload["action_name"] == "allow_worker"
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `uv run pytest -q tests/test_orchestration_policy.py tests/test_builtin_tools.py -k policy`
Expected: FAIL because selection still uses simple sorting

- [ ] **Step 4: Implement the minimal conservative contextual bandit**

For each action:

- compute adjusted empirical reward
- add feedback shaping
- subtract risk penalties
- subtract a confidence penalty proportional to `sqrt(log(total_samples + 1) / samples)`

Choose the action with the highest conservative lower-confidence score.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/test_orchestration_policy.py tests/test_builtin_tools.py -k policy`
Expected: PASS

### Task 5: Docs And Verification

**Files:**
- Modify: `README.md`
- No new production files expected

- [ ] **Step 1: Update operator docs**

Document:

- richer collected signals
- feedback shaping
- contextual bucket behavior
- conservative bandit selection logic

- [ ] **Step 2: Run targeted policy suite**

Run: `uv run pytest -q tests/test_orchestration_policy_store.py tests/test_orchestration_policy_capture.py tests/test_orchestration_policy.py tests/test_orchestrator_policy_feedback.py tests/test_builtin_tools.py`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -q`
Expected: PASS

- [ ] **Step 4: Review diff**

Run: `git diff --stat`
Expected: only planned files changed

- [ ] **Step 5: Commit**

```bash
git add README.md babybot tests docs/superpowers/plans/2026-03-27-policy-learning-upgrade.md
git commit -m "feat: upgrade orchestration policy learning"
```
