# Resource Subagent Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract subagent runtime orchestration from `resource.py` into a focused helper without changing worker behavior or public API.

**Architecture:** Introduce `resource_subagent_runtime.py` to own runtime lease merging, executor setup, and execution-context plumbing. Keep `resource.py` as a facade with stable wrappers for external callers and tests.

**Tech Stack:** Python 3.11, pytest, existing `babybot` agent kernel and worker executor

---

### Task 1: Lock Runtime Compatibility

**Files:**
- Modify: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Add focused runtime coverage**

Cover:
- `run_subagent_task()` still returns collected media paths
- channel context still reaches worker execution context
- skill tool leases are still merged into the executor task lease
- executor-facing skill packs still have empty tool leases

- [ ] **Step 2: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'run_subagent_task or select_skill_packs or build_worker_prompt'`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_resource_skills.py
git commit -m "Add subagent runtime compatibility tests"
```

### Task 2: Extract Runtime Helper

**Files:**
- Create: `/Users/shike/Desktop/babybot/babybot/resource_subagent_runtime.py`
- Modify: `/Users/shike/Desktop/babybot/babybot/resource.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Create `ResourceSubagentRuntime`**

Move orchestration from `_run_subagent_task()` into the helper, including:
- write-root activation
- merged lease assembly
- executor skill-pack preparation
- execution context creation
- executor invocation
- text/media result normalization

- [ ] **Step 2: Keep facade wrappers in `ResourceManager`**

Preserve method signatures and delegate through a lazily created helper instance.

- [ ] **Step 3: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'run_subagent_task or select_skill_packs or build_worker_prompt'`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add babybot/resource.py babybot/resource_subagent_runtime.py tests/test_resource_skills.py
git commit -m "Extract resource subagent runtime"
```

### Task 3: Run Regression Verification

**Files:**
- Modify: none expected
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_module_exports.py`, `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`, `/Users/shike/Desktop/babybot/tests/test_config.py`, `/Users/shike/Desktop/babybot/tests/test_worker.py`, `/Users/shike/Desktop/babybot/tests/test_orchestrator_routing.py`, `/Users/shike/Desktop/babybot/tests/test_runtime_refactor_event_bus.py`, `/Users/shike/Desktop/babybot/tests/test_message_bus_streaming.py`, `/Users/shike/Desktop/babybot/tests/test_feishu_streaming.py`

- [ ] **Step 1: Run broader regression**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py tests/test_config.py tests/test_worker.py tests/test_orchestrator_routing.py tests/test_runtime_refactor_event_bus.py tests/test_message_bus_streaming.py tests/test_feishu_streaming.py`
Expected: PASS
