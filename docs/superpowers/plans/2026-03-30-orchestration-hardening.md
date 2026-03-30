# Orchestration Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden BabyBot's orchestration layer so contracts/plans are authoritative, final replies cannot race unfinished work, runtime feedback is projected consistently, and debate/feedback flows behave predictably.

**Architecture:** Keep the existing `OrchestratorAgent -> DynamicOrchestrator -> child runtime` structure, but move more routing decisions into `TaskContract`/`ExecutionPlan`, add explicit guardrails around reply completion, and centralize runtime-event-to-job-state projection. Improve loop handling and team feedback with minimal changes that preserve current channel integrations.

**Tech Stack:** Python, asyncio, pytest

---

### Task 1: Strengthen contract and execution-plan routing

**Files:**
- Modify: `babybot/task_contract.py`
- Modify: `babybot/execution_plan.py`
- Test: `tests/test_task_contract.py`
- Test: `tests/test_execution_plan.py`

- [ ] Add failing tests for richer contract/plan fields and non-debate workflow kind.
- [ ] Implement contract defaults for orchestration tool allowlist and plan metadata.
- [ ] Make `ExecutionPlan` emit `tool_workflow` for normal orchestrated work and carry allowed tools/agents.
- [ ] Run focused contract/plan tests.

### Task 2: Guard premature final replies

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] Add failing tests showing `reply_to_user` is rejected when child tasks are still running or mixed with other tool calls.
- [ ] Enforce final-reply exclusivity and pending-task checks in the orchestration loop.
- [ ] Run focused dynamic-orchestrator tests.

### Task 3: Centralize runtime event projection

**Files:**
- Modify: `babybot/runtime_jobs.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_runtime_jobs.py`

- [ ] Add failing tests for runtime-event to job-state projection.
- [ ] Move projection logic into runtime job helpers and update orchestrator to use it.
- [ ] Run focused runtime-jobs tests.

### Task 4: Improve loop control and feedback flow

**Files:**
- Modify: `babybot/agent_kernel/loop_guard.py`
- Modify: `babybot/agent_kernel/executor.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Test: `tests/test_loop_guard.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] Add failing tests for non-destructive identical-call blocking and team runtime feedback events.
- [ ] Make identical-call guard advisory before hard-disabling tools.
- [ ] Emit normalized runtime feedback for debate/team progress.
- [ ] Run focused loop-guard/orchestrator tests.

### Task 5: Update docs

**Files:**
- Modify: `README.md`
- Modify: `docs/agent-runtime/interaction-contract.md`
- Modify: `docs/agent-runtime/feedback-state-machine.md`

- [ ] Update README and runtime docs to match the hardened orchestration behavior.

### Task 6: Verify end-to-end

**Files:**
- Verify: `tests/test_task_contract.py`
- Verify: `tests/test_execution_plan.py`
- Verify: `tests/test_dynamic_orchestrator.py`
- Verify: `tests/test_loop_guard.py`
- Verify: `tests/test_runtime_jobs.py`
- Verify: broader `pytest` / `ruff` as needed

- [ ] Run focused verification.
- [ ] Run broader regression suite.
