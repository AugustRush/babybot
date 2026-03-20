# Resource Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `babybot/resource.py` into smaller internal modules in one refactor pass while keeping the existing `babybot.resource` public API and current runtime behavior stable.

**Architecture:** Keep `babybot/resource.py` as a facade module and move pure models, external Python execution, workspace tools, and skill loading into dedicated internal modules. Re-export existing public symbols from the facade so callers and tests continue to import from `babybot.resource`.

**Tech Stack:** Python 3.11, pytest, `babybot` agent kernel, existing resource/tool registry conventions

---

### Task 1: Lock Public Compatibility With Failing Tests

**Files:**
- Modify: `tests/test_resource_skills.py`
- Create: `tests/test_resource_module_exports.py`
- Test: `tests/test_resource_module_exports.py`

- [ ] **Step 1: Write failing compatibility tests**

Cover:
- `babybot.resource` still exports `ResourceManager`, `CallableTool`, `ResourceCatalog`, `WorkerRuntime`
- `babybot.resource` still exports `ToolGroup`, `ResourceBrief`, `LoadedSkill`, `SkillRuntimeConfig`
- resource model imports continue to work without changing callers

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py`
Expected: FAIL because the re-export contract is not locked yet

- [ ] **Step 3: Add the minimal tests needed in `tests/test_resource_skills.py`**

Add one focused assertion that existing registration/runtime entry points still resolve through `babybot.resource` after extraction.

- [ ] **Step 4: Run focused tests again**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py -k 'resource_module_exports or cli_script_proxy or external_skill_function'`
Expected: FAIL only for the new compatibility expectations

- [ ] **Step 5: Commit**

```bash
git add tests/test_resource_module_exports.py tests/test_resource_skills.py
git commit -m "add resource facade compatibility tests"
```

### Task 2: Extract Resource Data Models

**Files:**
- Create: `babybot/resource_models.py`
- Modify: `babybot/resource.py`
- Test: `tests/test_resource_module_exports.py`

- [ ] **Step 1: Create `babybot/resource_models.py`**

Move:
- `ToolGroup`
- `ResourceBrief`
- `LoadedSkill`
- `SkillRuntimeConfig`
- `_ScriptFunctionSpec`
- `_CliArgumentSpec`
- `_CliToolSpec`

Keep this module limited to dataclasses and lightweight helpers only.

- [ ] **Step 2: Re-export model symbols from `babybot/resource.py`**

Update imports in `resource.py` so tests and current callers still import these symbols from the facade module.

- [ ] **Step 3: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py -k 'resource_module_exports or parse_frontmatter or register_skill_tools'`
Expected: PASS

- [ ] **Step 4: Inspect for import-cycle regressions**

Run: `PYTHONPATH=. uv run python -c "import babybot.resource; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add babybot/resource_models.py babybot/resource.py tests/test_resource_module_exports.py tests/test_resource_skills.py
git commit -m "extract resource data models"
```

### Task 3: Extract External Python Runner

**Files:**
- Create: `babybot/resource_python_runner.py`
- Modify: `babybot/resource.py`
- Modify: `tests/test_resource_skills.py`
- Test: `tests/test_resource_skills.py`

- [ ] **Step 1: Write or extend focused tests for shared fallback behavior**

Ensure both function-proxy and CLI-proxy skill execution still:
- use the same candidate ordering
- retry on environment-class failure
- stop retrying on business-class failure

- [ ] **Step 2: Create `ExternalPythonRunner` in `babybot/resource_python_runner.py`**

Move logic for:
- candidate discovery
- fallback selection
- module probing
- unhealthy-runtime cache
- environment failure classification
- external function execution
- external CLI execution

- [ ] **Step 3: Rewire `ResourceManager` to delegate**

Keep wrapper methods in `resource.py` only where needed for compatibility with tests or monkeypatching.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'python_candidates or external_skill_function or cli_script_proxy'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/resource_python_runner.py babybot/resource.py tests/test_resource_skills.py
git commit -m "extract external python skill runner"
```

### Task 4: Extract Workspace Tool Suite

**Files:**
- Create: `babybot/resource_workspace_tools.py`
- Modify: `babybot/resource.py`
- Modify: `tests/test_resource_skills.py`
- Test: `tests/test_resource_skills.py`

- [ ] **Step 1: Create `WorkspaceToolSuite`**

Move:
- shell safety checks
- workspace path resolution helpers
- workspace Python execution
- workspace shell execution
- text file read/write/insert helpers

- [ ] **Step 2: Keep `ResourceManager` registration behavior unchanged**

`ResourceManager._register_builtin_tools()` should still register the same tool names into the same groups.

- [ ] **Step 3: Preserve active write-root behavior**

Make sure the extracted suite still resolves paths relative to the current workspace write root and keeps workspace boundary enforcement unchanged.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py -k 'callable_tool or relocates_external_artifact or cli_script_proxy'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/resource_workspace_tools.py babybot/resource.py tests/test_resource_skills.py
git commit -m "extract workspace tool suite"
```

### Task 5: Extract Skill Loader

**Files:**
- Create: `babybot/resource_skill_loader.py`
- Modify: `babybot/resource.py`
- Modify: `tests/test_resource_skills.py`
- Test: `tests/test_resource_skills.py`

- [ ] **Step 1: Create `SkillLoader`**

Move logic for:
- frontmatter parsing
- skill document reading
- configured skill loading
- builtin/workspace skill discovery
- AST function extraction
- argparse schema extraction
- skill tool registration

- [ ] **Step 2: Define narrow dependencies**

Pass into the loader only:
- config
- registry
- tool-group map
- external Python runner
- callable tool registration callback

Avoid importing `ResourceManager` inside the new module.

- [ ] **Step 3: Rewire `ResourceManager._load_config()` and related methods**

`ResourceManager` should delegate:
- `_register_configured_skills()`
- `_discover_skills()`
- `_register_skill_tools()`

Keep thin wrapper methods in `resource.py` if tests or callers rely on them.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add babybot/resource_skill_loader.py babybot/resource.py tests/test_resource_skills.py
git commit -m "extract resource skill loader"
```

### Task 6: Shrink `resource.py` To A Stable Facade

**Files:**
- Modify: `babybot/resource.py`
- Modify: `tests/test_resource_module_exports.py`
- Test: `tests/test_resource_module_exports.py`, `tests/test_resource_skills.py`

- [ ] **Step 1: Remove duplicated implementation left behind in `resource.py`**

After the extractions, delete dead helpers and inline duplication while preserving:
- public exports
- `CallableTool`
- `ResourceCatalog`
- `WorkerRuntime`
- `ResourceManager._run_subagent_task()`

- [ ] **Step 2: Verify facade-level compatibility**

Confirm that existing imports and monkeypatch targets used by tests still resolve.

- [ ] **Step 3: Run focused facade tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py`
Expected: PASS

- [ ] **Step 4: Inspect file size reduction**

Run: `wc -l babybot/resource.py babybot/resource_models.py babybot/resource_python_runner.py babybot/resource_workspace_tools.py babybot/resource_skill_loader.py`
Expected: `babybot/resource.py` is materially smaller and the extracted modules have clear responsibility boundaries

- [ ] **Step 5: Commit**

```bash
git add babybot/resource.py babybot/resource_models.py babybot/resource_python_runner.py babybot/resource_workspace_tools.py babybot/resource_skill_loader.py tests/test_resource_module_exports.py tests/test_resource_skills.py
git commit -m "shrink resource module to facade"
```

### Task 7: Full Regression Verification

**Files:**
- Modify: none expected
- Test: `tests/test_resource_module_exports.py`, `tests/test_resource_skills.py`, `tests/test_config.py`, `tests/test_worker.py`, `tests/test_orchestrator_routing.py`, `tests/test_runtime_refactor_event_bus.py`, `tests/test_message_bus_streaming.py`, `tests/test_feishu_streaming.py`

- [ ] **Step 1: Run the full regression suite for this refactor**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_module_exports.py tests/test_resource_skills.py tests/test_config.py tests/test_worker.py tests/test_orchestrator_routing.py tests/test_runtime_refactor_event_bus.py tests/test_message_bus_streaming.py tests/test_feishu_streaming.py`
Expected: PASS

- [ ] **Step 2: Inspect git diff for scope control**

Run: `git diff --stat`
Expected: only `babybot/resource*.py`, `tests/test_resource*.py`, and any minimal compatibility adjustments directly required by the extraction

- [ ] **Step 3: Record resulting module boundaries**

Summarize:
- what remained in `resource.py`
- what moved into each extracted module
- any intentionally deferred follow-up refactors

- [ ] **Step 4: Prepare final implementation summary**

Include:
- key structural changes
- compatibility guarantees preserved
- exact test evidence
