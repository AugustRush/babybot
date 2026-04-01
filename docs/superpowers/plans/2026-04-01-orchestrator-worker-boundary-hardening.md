# Orchestrator Worker Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a strict boundary where the main orchestrator plans and delegates, normal child workers only execute bounded tasks, and only the main orchestrator may use team mode.

**Architecture:** Tighten authority at the capability layer first, then make task dispatch explicit and structured, and finally simplify worker prompts so they reinforce execution-only behavior instead of open-ended exploration. Keep `dispatch_team` available only to the main orchestrator while removing nested worker orchestration and direct channel delivery from normal child runs.

**Tech Stack:** Python, pytest, BabyBot agent kernel, resource manager, tool leases

---

### Task 1: Lock Normal Workers Out Of Nested Orchestration

**Files:**
- Modify: `babybot/builtin_tools/workers.py`
- Modify: `tests/test_worker.py`

- [x] **Step 1: Add failing tests for nested worker denial**

Verify that `create_worker` and `dispatch_workers` return an explicit denial message when invoked from worker depth `> 0`.

- [x] **Step 2: Implement hard denial in worker tools**

Short-circuit nested worker creation and nested worker dispatch before any policy checks or fallback behavior.

- [x] **Step 3: Run targeted worker tests**

Run: `pytest tests/test_worker.py -q`

### Task 2: Remove Direct User Delivery From Normal Child Tasks

**Files:**
- Modify: `babybot/resource_subagent_runtime.py`
- Modify: `tests/test_resource_skills.py`

- [x] **Step 1: Add failing tests for child-task lease hardening**

Verify that normal child runs exclude channel delivery tools and worker-control tools even when the parent channel context exists.

- [x] **Step 2: Harden child-task lease construction**

Strip `worker_control` and `channel_*` groups from normal child runs. Preserve them only for explicitly marked delivery-style tasks in the future.

- [x] **Step 3: Run targeted lease/runtime tests**

Run: `pytest tests/test_resource_skills.py -q`

### Task 3: Make Main-Orchestrator Dispatch Contracts Structured

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/agent_kernel/dag_ports.py`
- Modify: `tests/test_dynamic_orchestrator.py`

- [x] **Step 1: Add failing tests for structured dispatch enrichment**

Verify that child-task descriptions include a bounded execution brief for maintenance / comparison style tasks and forbid open-ended expansion.

- [x] **Step 2: Implement dispatch-task enrichment**

Before dispatching normal child tasks, transform the raw description into a structured execution brief with objective, inputs, expected output, completion condition, and forbidden actions.

- [x] **Step 3: Keep team mode as main-orchestrator-only**

Do not expose `dispatch_team` to child workers; keep it only in the main orchestrator tool set.

- [x] **Step 4: Run targeted orchestrator tests**

Run: `pytest tests/test_dynamic_orchestrator.py -q`

### Task 4: Convert Worker Prompt From Planner To Executor

**Files:**
- Modify: `babybot/resource_skill_runtime.py`
- Modify: `tests/test_resource_skills.py`

- [x] **Step 1: Add failing tests for execution-only worker prompt**

Verify that the worker prompt explicitly states the worker is not an orchestrator, must stay within budget, and must fail fast on missing inputs.

- [x] **Step 2: Rewrite the worker system prompt**

Replace the generic "finish the task however you can" wording with a short execution checklist that mirrors the new structured contract.

- [x] **Step 3: Run targeted prompt tests**

Run: `pytest tests/test_resource_skills.py -q`

### Task 5: Tighten Exploration Convergence

**Files:**
- Modify: `babybot/agent_kernel/loop_guard.py`
- Modify: `tests/test_loop_guard.py`

- [x] **Step 1: Add failing tests for shared exploration budgets**

Verify that repeated shell/python read-only exploration shares one budget instead of letting the model switch tools to keep wandering.

- [x] **Step 2: Implement shared exploration accounting**

Treat read-only shell/python exploration as one class of exploratory behavior and block it earlier with a clearer stop-and-return message.

- [x] **Step 3: Run targeted loop-guard tests**

Run: `pytest tests/test_loop_guard.py -q`

### Task 6: Verify The Whole Boundary End To End

**Files:**
- Modify: `docs/superpowers/plans/2026-04-01-orchestrator-worker-boundary-hardening.md`

- [x] **Step 1: Run the focused regression suite**

Run: `pytest tests/test_worker.py tests/test_resource_skills.py tests/test_dynamic_orchestrator.py tests/test_loop_guard.py -q`

- [x] **Step 2: Run a broader orchestration smoke suite**

Run: `pytest tests/test_orchestrator_routing.py tests/test_orchestrator_policy_feedback.py -q`

- [x] **Step 3: Update plan checkboxes and summarize residual risk**

Mark completed steps and note any deferred follow-up such as a future raw-URL text fetch tool.

Residual follow-up:
- Add a dedicated raw URL / GitHub blob text reader so documentation-comparison tasks stop paying the browser-navigation cost before they even reach the execution worker.
