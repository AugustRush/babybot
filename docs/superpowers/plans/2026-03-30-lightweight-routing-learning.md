# Lightweight Routing Learning Implementation Plan

**Status:** Completed. Lightweight routing, async evaluator, reflection hints, and later follow-up refinements all landed on top of this plan.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight context router, async evaluator, and reflection hints loop on top of BabyBot's existing orchestration stack without hurting latency or requiring heavy local compute.

**Architecture:** Reuse the current context stack (`Tape`, anchor summaries, `ContextView`, runtime jobs) to build a compact routing snapshot. Use one structured small-model call to route normal DAG tasks, keep `ConservativePolicySelector` for local action choice, and persist async task evaluations plus small reflection hints for future requests.

**Tech Stack:** Python, asyncio, pytest, sqlite

---

### Task 1: Add config and routing contracts

**Files:**
- Modify: `babybot/config.py`
- Create: `babybot/orchestration_router.py`
- Test: `tests/test_config.py`
- Test: `tests/test_orchestrator_routing.py`

- [ ] Add failing tests for routing model config defaults and fallback behavior.
- [ ] Add lightweight router config fields and routing dataclasses.
- [ ] Run focused routing/config tests.

### Task 2: Build context snapshot and router integration

**Files:**
- Create: `babybot/orchestration_router.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/task_contract.py`
- Modify: `babybot/execution_plan.py`
- Test: `tests/test_orchestrator_routing.py`
- Test: `tests/test_task_contract.py`

- [ ] Add failing tests for snapshot building and router fallback.
- [ ] Build snapshot from Tape anchor, ContextView, memory, and runtime state.
- [ ] Inject router decision into contract/plan/policy hints.
- [ ] Run focused orchestration routing tests.

### Task 3: Add reflection persistence

**Files:**
- Modify: `babybot/orchestration_policy_store.py`
- Test: `tests/test_orchestration_policy_store.py`

- [ ] Add failing tests for recording and querying reflection hints.
- [ ] Add a lightweight reflection table and bounded query API.
- [ ] Run focused policy-store tests.

### Task 4: Add async task evaluator

**Files:**
- Create: `babybot/task_evaluator.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_orchestrator_routing.py`
- Test: `tests/test_runtime_jobs.py`

- [ ] Add failing tests for async evaluator scheduling and reflection write-back.
- [ ] Implement evaluator from existing runtime signals.
- [ ] Run focused evaluator tests.

### Task 5: Use reflection hints in policy selection

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/orchestration_policy.py`
- Test: `tests/test_orchestration_policy.py`
- Test: `tests/test_orchestrator_routing.py`

- [ ] Add failing tests for reflection hints influencing local policy choice.
- [ ] Thread top reflection hints into policy feature selection and prompt hints.
- [ ] Run focused policy tests.

### Task 6: Update docs and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/agent-runtime/interaction-contract.md`
- Verify: focused and full `pytest`
- Verify: `ruff check`

- [ ] Update docs for routing/evaluator/reflection behavior.
- [ ] Run focused verification.
- [ ] Run full verification.
