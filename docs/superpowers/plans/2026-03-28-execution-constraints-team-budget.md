# Execution Constraints and Team Budget Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make user-stated execution constraints survive planning and enforce runtime budgets for multi-agent debates.

**Architecture:** Add a normalized execution-constraints layer before orchestration, pass it through `ExecutionContext`, and derive an effective team policy inside `dispatch_team`. Extend `TeamRunner` with explicit runtime budgets and partial-summary degradation so long debates stop cleanly before channel hard timeouts.

**Tech Stack:** Python, asyncio, pytest

---

### Task 1: Add execution constraints model and parser

**Files:**
- Create: `babybot/agent_kernel/execution_constraints.py`
- Test: `tests/test_execution_constraints.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal execution constraints model and parser**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Propagate constraints through orchestration and enforce them in dispatch_team

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Test: `tests/test_orchestrator_routing.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Pass normalized constraints into `ExecutionContext` and merge them into `dispatch_team`**
- [ ] **Step 4: Run tests to verify they pass**

### Task 3: Add TeamRunner budget policy and partial-summary degradation

**Files:**
- Modify: `babybot/agent_kernel/team.py`
- Test: `tests/test_agent_team.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Add deadline and per-turn timeout enforcement with partial summaries**
- [ ] **Step 4: Run test to verify it passes**

### Task 4: Verify touched paths

**Files:**
- Modify: `babybot/agent_kernel/execution_constraints.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/agent_kernel/team.py`
- Test: `tests/test_execution_constraints.py`
- Test: `tests/test_orchestrator_routing.py`
- Test: `tests/test_dynamic_orchestrator.py`
- Test: `tests/test_agent_team.py`

- [ ] **Step 1: Run focused pytest targets**
- [ ] **Step 2: Run `ruff check` on touched files**
- [ ] **Step 3: Review diff for accidental scope creep**
