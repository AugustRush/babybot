# Resource Tool Loader Refactor Design

## Goal

Keep shrinking `/Users/shike/Desktop/babybot/babybot/resource.py` by moving tool registration and module-loading bootstrap logic into a dedicated internal helper, while preserving the existing `ResourceManager` public API and runtime behavior.

## Scope

This slice moves:

- callable tool registration
- custom tool registration from config
- workspace tool discovery and import
- Python module loading from workspace paths
- JSON schema inference for callable registration

This slice does not change:

- builtin tool definitions
- skill loading and external Python execution
- resource scope resolution
- sub-agent runtime flow

## Options Considered

1. Keep all bootstrap logic inside `resource.py`
   - lowest risk
   - does not improve file clarity enough

2. Split discovery only, keep schema and registration in facade
   - smaller patch
   - still leaves bootstrap concerns spread across files

3. Recommended: single `resource_tool_loader.py`
   - one owner for registration, import, and schema-building
   - keeps `resource.py` as facade + orchestration shell
   - easiest boundary to test independently

## Proposed Structure

Add `/Users/shike/Desktop/babybot/babybot/resource_tool_loader.py` with a helper that owns:

- `register_tool()`
- `register_custom_tools()`
- `discover_workspace_tools()`
- `ensure_workspace_on_pythonpath()`
- `load_tool_module()`
- `json_schema_for_callable()`
- `schema_for_annotation()`

`/Users/shike/Desktop/babybot/babybot/resource.py` keeps the old method names and delegates through thin wrappers.

## Compatibility Constraints

- `ResourceManager.register_tool()` signature stays unchanged
- private methods used in tests remain available:
  - `_register_custom_tools`
  - `_discover_workspace_tools`
  - `_load_tool_module`
  - `_json_schema_for_callable`
  - `_schema_for_annotation`
- custom tool env var expansion and workspace module import fallback keep working
