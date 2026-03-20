# Resource Subagent Runtime Refactor Design

## Goal

Further shrink `/Users/shike/Desktop/babybot/babybot/resource.py` by extracting subagent runtime orchestration into a dedicated internal helper, while preserving the public `ResourceManager` API and current worker behavior.

## Scope

This slice moves the orchestration logic currently inside `_run_subagent_task()`:

- active write-root scoping
- base lease creation and skill lease merging
- selected skill pack preparation for executor
- worker executor construction and invocation
- runtime event context assembly
- final text/media result normalization

This slice keeps in place:

- `ResourceManager.run_subagent_task()`
- `ResourceManager._run_subagent_task()`
- skill selection methods
- worker system prompt construction
- resource/tool registration logic

## Options Considered

1. Leave runtime orchestration inline
   - lowest risk
   - keeps too much orchestration buried in `resource.py`

2. Extract only executor invocation
   - smaller patch
   - still leaves lease merge and context handling spread out

3. Recommended: single `resource_subagent_runtime.py`
   - one owner for runtime orchestration
   - facade remains stable
   - easiest path to further split skill selection/prompt building later

## Proposed Structure

Add `/Users/shike/Desktop/babybot/babybot/resource_subagent_runtime.py` with a helper that owns:

- `run_subagent_task()`
- skill lease merge helper
- executor skill-pack sanitization helper
- execution context creation helper

`/Users/shike/Desktop/babybot/babybot/resource.py` keeps thin wrappers and compatibility-visible method names.

## Compatibility Constraints

- `ResourceManager.run_subagent_task()` and `_run_subagent_task()` signatures must not change
- current context propagation for:
  - `channel_context`
  - `current_task_lease`
  - `current_skill_ids`
  must remain intact
- merged executor task lease must still union skill tool leases
- executor-facing `SkillPack`s must still be stripped of tool leases
- final text/media output behavior must remain unchanged
