# Resource Scope Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract resource catalog, scope resolution, and task lease logic from `resource.py` into a focused internal module without changing public behavior.

**Architecture:** Add a new `resource_scope.py` helper that encapsulates resource indexing and lease-building rules. Keep `resource.py` as a facade with compatibility wrappers so external imports and tests remain stable.

**Tech Stack:** Python 3.11, pytest, existing `babybot` resource/tool registry model

---

### Task 1: Lock Scope Behavior With Focused Tests

**Files:**
- Modify: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Add additive coverage for resource search and lease filtering**

Cover:
- `search_resources()` still returns matching groups, tools, skills, and MCP server names
- `_build_task_lease()` still drops unknown include tools and keeps orchestration tools excluded by default

- [ ] **Step 2: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'resource_briefs or search_resources or build_task_lease'`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_resource_skills.py
git commit -m "Add resource scope compatibility tests"
```

### Task 2: Extract Resource Scope Helper

**Files:**
- Create: `/Users/shike/Desktop/babybot/babybot/resource_scope.py`
- Modify: `/Users/shike/Desktop/babybot/babybot/resource.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Create `ResourceScopeHelper`**

Move logic for:
- `_preview_tool_names`
- resource id helpers
- `_lease_to_dict`
- `_get_resource_briefs`
- `_resolve_resource_scope`
- `_search_resources`
- `_build_task_lease`

- [ ] **Step 2: Keep facade wrappers in `ResourceManager`**

Preserve method names and signatures by delegating through a lazily created helper instance.

- [ ] **Step 3: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'resource_briefs or search_resources or build_task_lease'`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add babybot/resource_scope.py babybot/resource.py tests/test_resource_skills.py
git commit -m "Extract resource scope helper"
```

### Task 3: Run Regression Verification

**Files:**
- Modify: none expected
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_module_exports.py`, `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`, `/Users/shike/Desktop/babybot/tests/test_config.py`, `/Users/shike/Desktop/babybot/tests/test_worker.py`, `/Users/shike/Desktop/babybot/tests/test_orchestrator_routing.py`, `/Users/shike/Desktop/babybot/tests/test_runtime_refactor_event_bus.py`, `/Users/shike/Desktop/babybot/tests/test_message_bus_streaming.py`, `/Users/shike/Desktop/babybot/tests/test_feishu_streaming.py`

- [ ] **Step 1: Run broader regression**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py tests/test_config.py tests/test_worker.py tests/test_orchestrator_routing.py tests/test_runtime_refactor_event_bus.py tests/test_message_bus_streaming.py tests/test_feishu_streaming.py`
Expected: PASS

- [ ] **Step 2: Commit if code changed during verification**

```bash
git add babybot/resource.py babybot/resource_scope.py tests/test_resource_skills.py
git commit -m "Verify resource scope refactor"
```
