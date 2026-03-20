# Resource Scope Refactor Design

## Goal

Continue shrinking `/Users/shike/Desktop/babybot/babybot/resource.py` by extracting the remaining resource-directory and lease-resolution logic into a dedicated internal module, while preserving the public `babybot.resource` API and current runtime behavior.

## Scope

This slice moves logic for:

- resource brief generation
- resource id normalization and scope resolution
- resource search results
- task lease construction for sub-agent runs

It does not change:

- `ResourceManager` public import path
- `ResourceCatalog` public class name
- sub-agent execution flow in `_run_subagent_task()`
- channel delivery ownership model (main agent only)

## Proposed Structure

Create `/Users/shike/Desktop/babybot/babybot/resource_scope.py` with a focused helper that owns:

- `get_resource_briefs()`
- `resolve_resource_scope()`
- `search_resources()`
- `build_task_lease()`

`/Users/shike/Desktop/babybot/babybot/resource.py` remains the facade:

- keep `ResourceCatalog` and `ResourceManager` entrypoints
- delegate internal behavior through thin wrappers
- preserve private method names used by tests and monkeypatching

## Why This Boundary

These methods all answer the same question: “what resources/tools should a caller see or get access to?” Grouping them together makes the lease logic, resource indexing, and search behavior easier to reason about and test without pulling in the rest of the runtime.

## Compatibility Constraints

- `from babybot.resource import ResourceManager, ResourceCatalog` must keep working
- `ResourceManager._get_resource_briefs`, `_resolve_resource_scope`, `_search_resources`, `_build_task_lease` must keep existing signatures
- existing tests around resource briefs, scope resolution, and nested orchestration tool filtering must still pass unchanged or with only additive assertions
