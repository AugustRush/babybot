# Resource Skill Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract skill selection and worker prompt/catalog formatting from `resource.py` into a focused helper without changing runtime behavior.

**Architecture:** Add `resource_skill_runtime.py` to own skill-pack selection and worker prompt/catalog assembly. Keep `ResourceManager` wrapper methods stable so current callers and tests continue to use `babybot.resource` unchanged.

**Tech Stack:** Python 3.11, pytest, existing `babybot` skill and lease models

---

### Task 1: Lock Skill Runtime Compatibility

**Files:**
- Modify: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Add focused tests**

Cover:
- active skill selection
- explicit `skill_ids` filtering
- worker prompt still includes skill catalog
- lease-scoped skill catalog still hides inaccessible skills

- [ ] **Step 2: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'select_skill_packs or build_worker_prompt or format_skill_catalog'`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_resource_skills.py
git commit -m "Add resource skill runtime compatibility tests"
```

### Task 2: Extract Skill Runtime Helper

**Files:**
- Create: `/Users/shike/Desktop/babybot/babybot/resource_skill_runtime.py`
- Modify: `/Users/shike/Desktop/babybot/babybot/resource.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Create `ResourceSkillRuntime`**

Move:
- `_select_skill_packs()`
- `_build_worker_sys_prompt()`
- `_format_skill_catalog()`
- `_format_skill_catalog_for_lease()`

- [ ] **Step 2: Preserve facade wrappers**

Keep existing `ResourceManager` method names and signatures, delegating to the helper.

- [ ] **Step 3: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'select_skill_packs or build_worker_prompt or format_skill_catalog'`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add babybot/resource.py babybot/resource_skill_runtime.py tests/test_resource_skills.py
git commit -m "Extract resource skill runtime"
```

### Task 3: Run Regression Verification

**Files:**
- Modify: none expected
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_module_exports.py`, `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`, `/Users/shike/Desktop/babybot/tests/test_config.py`, `/Users/shike/Desktop/babybot/tests/test_worker.py`, `/Users/shike/Desktop/babybot/tests/test_orchestrator_routing.py`, `/Users/shike/Desktop/babybot/tests/test_runtime_refactor_event_bus.py`, `/Users/shike/Desktop/babybot/tests/test_message_bus_streaming.py`, `/Users/shike/Desktop/babybot/tests/test_feishu_streaming.py`

- [ ] **Step 1: Run broader regression**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py tests/test_config.py tests/test_worker.py tests/test_orchestrator_routing.py tests/test_runtime_refactor_event_bus.py tests/test_message_bus_streaming.py tests/test_feishu_streaming.py`
Expected: PASS
