# Resource Tool Loader Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract tool registration and module-loading bootstrap logic from `resource.py` into a focused helper module without changing public behavior.

**Architecture:** Introduce `resource_tool_loader.py` to own registration, workspace import, and callable schema inference. Preserve `ResourceManager` wrapper methods so tests and callers continue to use `babybot.resource` unchanged.

**Tech Stack:** Python 3.11, pytest, existing `babybot` tool registry abstractions

---

### Task 1: Lock Tool Loader Compatibility

**Files:**
- Modify: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Add focused tests**

Cover:
- `_load_tool_module()` raises a proxy-friendly error when a module calls `sys.exit()`
- `_discover_workspace_tools()` registers public functions only
- `_register_custom_tools()` expands `${ENV_VAR}` preset kwargs before invocation

- [ ] **Step 2: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'json_schema_for_callable or load_tool_module or discover_workspace_tools or register_custom_tools'`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_resource_skills.py
git commit -m "Add resource tool loader compatibility tests"
```

### Task 2: Extract Tool Loader Helper

**Files:**
- Create: `/Users/shike/Desktop/babybot/babybot/resource_tool_loader.py`
- Modify: `/Users/shike/Desktop/babybot/babybot/resource.py`
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`

- [ ] **Step 1: Create `ResourceToolLoader`**

Move:
- `register_tool()`
- `_register_custom_tools()`
- `_discover_workspace_tools()`
- `_ensure_workspace_on_pythonpath()`
- `_load_tool_module()`
- `_json_schema_for_callable()`
- `_schema_for_annotation()`

- [ ] **Step 2: Preserve facade wrappers**

Keep the existing `ResourceManager` method names and signatures, delegating to the helper.

- [ ] **Step 3: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'json_schema_for_callable or load_tool_module or discover_workspace_tools or register_custom_tools'`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add babybot/resource.py babybot/resource_tool_loader.py tests/test_resource_skills.py
git commit -m "Extract resource tool loader"
```

### Task 3: Run Regression Verification

**Files:**
- Modify: none expected
- Test: `/Users/shike/Desktop/babybot/tests/test_resource_module_exports.py`, `/Users/shike/Desktop/babybot/tests/test_resource_skills.py`, `/Users/shike/Desktop/babybot/tests/test_config.py`, `/Users/shike/Desktop/babybot/tests/test_worker.py`, `/Users/shike/Desktop/babybot/tests/test_orchestrator_routing.py`, `/Users/shike/Desktop/babybot/tests/test_runtime_refactor_event_bus.py`, `/Users/shike/Desktop/babybot/tests/test_message_bus_streaming.py`, `/Users/shike/Desktop/babybot/tests/test_feishu_streaming.py`

- [ ] **Step 1: Run broader regression**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py tests/test_config.py tests/test_worker.py tests/test_orchestrator_routing.py tests/test_runtime_refactor_event_bus.py tests/test_message_bus_streaming.py tests/test_feishu_streaming.py`
Expected: PASS
