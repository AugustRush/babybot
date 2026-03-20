# Resource Refactor Design

**Goal:** Split `babybot/resource.py` into smaller runtime-focused modules in a single refactor pass while preserving the current public API and behavior of resource discovery, tool registration, workspace tools, and subagent execution.

**Constraints**
- Keep imports like `from babybot.resource import ResourceManager` working.
- Preserve existing behavior for skill discovery, workspace execution, subagent execution, and channel tool registration.
- Do not expand scope into `babybot/channels/feishu.py` or `babybot/agent_kernel/dynamic_orchestrator.py` in this round.
- Keep the current test surface valid with minimal caller changes.
- Prefer extraction and delegation over semantic redesign.

**Current State**
- `babybot/resource.py` is the largest file in the repository and currently combines:
  - core data models
  - resource catalog logic
  - skill discovery and registration
  - AST and CLI schema extraction
  - host Python interpreter selection and fallback
  - external skill execution
  - workspace shell/python/file tools
  - channel tool registration
  - subagent runtime orchestration
- `ResourceManager` acts as both facade and implementation body, which makes the file difficult to reason about and easy to destabilize when adding features.
- Recent work on host Python fallback and worker runtime settings increased capability, but it also increased the amount of policy and execution logic concentrated in this file.

**Design**

## 1. Preserve `resource.py` as the public facade

`babybot/resource.py` remains the import surface for:
- `ResourceManager`
- `CallableTool`
- `ResourceCatalog`
- `WorkerRuntime`
- data models currently imported from this module

The file should stop owning most implementation details. Instead, it should assemble smaller internal helpers and delegate work to them.

This keeps external imports stable and avoids churn across the rest of the codebase.

## 2. Extract pure data models into `babybot/resource_models.py`

Move the following definitions into a dedicated model module:
- `ToolGroup`
- `ResourceBrief`
- `LoadedSkill`
- `SkillRuntimeConfig`
- `_ScriptFunctionSpec`
- `_CliArgumentSpec`
- `_CliToolSpec`

This module should contain only dataclasses and lightweight serialization helpers. It should not import runtime-heavy modules like `ToolRegistry`, `Config`, channel classes, or subprocess execution helpers.

`resource.py` should re-export these symbols so tests and existing imports remain valid.

## 3. Extract host Python selection and external script execution into `babybot/resource_python_runner.py`

Create a focused runner responsible for:
- host Python candidate selection
- required-module probing
- unhealthy-runtime caching
- environment-class failure detection
- external function-style skill execution
- external CLI-style skill execution

Recommended shape:
- `ExternalPythonRunner`
- initialized with callbacks or lightweight dependencies needed from `ResourceManager`
- exposes methods used by skill registration and workspace execution paths

This module absorbs logic currently centered around:
- interpreter selection and probing
- `_build_external_cli_script_callable`
- `_invoke_external_skill_function`

The key requirement is to unify duplicated fallback behavior so function proxies and CLI proxies share one execution policy.

## 4. Extract skill discovery and registration into `babybot/resource_skill_loader.py`

Create a loader module responsible for:
- `SKILL.md` frontmatter parsing
- skill document reading
- configured skill loading
- builtin/workspace skill discovery
- AST function extraction
- argparse schema extraction
- skill tool registration

Recommended shape:
- `SkillLoader`
- constructed with:
  - config
  - tool registry
  - group registry
  - python runner
  - registration callback for normal in-process tools

This module should register tools and return `LoadedSkill` values, but it should not execute subagent tasks or own workspace file tools.

`ResourceManager` should delegate `_load_config()`, `_register_configured_skills()`, `_discover_skills()`, and `_register_skill_tools()` behavior into this helper.

## 5. Extract workspace execution tools into `babybot/resource_workspace_tools.py`

Create a workspace tool module responsible for:
- shell safety checks
- workspace path resolution
- shell command execution
- Python code execution
- text file read/write/insert helpers

Recommended shape:
- `WorkspaceToolSuite`
- initialized with config and write-root resolver callbacks

This module should encapsulate filesystem and subprocess behavior for the code tool group. It should not know about skills, channel tools, or subagent orchestration.

`ResourceManager` should continue to register these tools into the registry, but the actual implementations should live in the extracted module.

## 6. Keep subagent orchestration inside `resource.py` for this round

`WorkerRuntime`, `ResourceCatalog`, `CallableTool`, and the `ResourceManager._run_subagent_task()` path should remain in `resource.py` for this refactor.

Reason:
- these pieces are tightly coupled to registry state, leases, channel context, and result artifact collection
- extracting them at the same time would turn this refactor from a structural split into a behavioral redesign

This round should optimize the biggest complexity source first: skill loading, external execution, and workspace tool logic.

## 7. Dependency direction

The new dependency flow should be:

- `resource_models.py`
  - no runtime dependencies
- `resource_python_runner.py`
  - depends on `resource_models.py`
- `resource_workspace_tools.py`
  - depends on config and simple helpers only
- `resource_skill_loader.py`
  - depends on `resource_models.py`, `resource_python_runner.py`, registry/group callbacks
- `resource.py`
  - imports all of the above and wires them together

Important restriction:
- extracted modules must not import `ResourceManager`
- they should depend on narrow callbacks or collaborators

This avoids replacing one large file with a circular import graph.

## 8. Backward compatibility strategy

Compatibility requirements for this refactor:
- existing tests importing symbols from `babybot.resource` must continue to pass
- channel and orchestrator code should not need import changes
- the registry tool names and skill group naming must remain unchanged
- current config keys must remain unchanged
- current external skill execution semantics must remain unchanged

Where compatibility would otherwise require a large caller update, `resource.py` should keep thin wrapper methods that delegate to the extracted helper classes.

## 9. Testing strategy

The refactor should be guarded by the current integration-heavy tests first, with a small number of new focused tests added only where extraction creates a new seam.

Verification focus:
- `tests/test_resource_skills.py`
- `tests/test_config.py`
- `tests/test_worker.py`
- `tests/test_orchestrator_routing.py`
- `tests/test_runtime_refactor_event_bus.py`
- `tests/test_message_bus_streaming.py`
- `tests/test_feishu_streaming.py`

New tests are only necessary if extraction changes a previously implicit contract, for example:
- re-export guarantees from `babybot.resource`
- shared fallback behavior between function and CLI proxy execution

## 10. Implementation sequence

The single refactor should still be done in stable internal order:

1. add `resource_models.py`
2. add `resource_python_runner.py`
3. add `resource_workspace_tools.py`
4. add `resource_skill_loader.py`
5. rewire `resource.py` to delegate to the new helpers
6. remove duplicated logic from `resource.py`
7. run the resource, routing, message-bus, and Feishu regression suite

This preserves momentum while keeping each extraction step locally testable.

## 11. Non-goals

This refactor does not:
- redesign the resource model
- redesign the skill frontmatter schema
- split `feishu.py`
- split `dynamic_orchestrator.py`
- replace dict-based runtime payloads with new typed protocol objects

Those are valid follow-up refactors, but combining them here would materially increase risk.

**Expected Outcome**
- `resource.py` becomes a facade/orchestration module rather than a 2900-line implementation dump.
- external Python execution becomes easier to reason about and modify safely.
- skill loading and workspace tool behavior become independently testable.
- future refactors of `feishu.py` and orchestrator code become easier because the resource layer no longer hides unrelated concerns behind one file.
