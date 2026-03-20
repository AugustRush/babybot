# Resource Skill Runtime Refactor Design

## Goal

Continue shrinking `/Users/shike/Desktop/babybot/babybot/resource.py` by extracting skill selection and worker prompt/catalog formatting into a dedicated helper, while preserving current `ResourceManager` behavior and public API.

## Scope

This slice moves:

- active skill selection
- explicit `skill_ids` filtering
- skill catalog formatting
- lease-scoped skill catalog filtering
- worker system prompt construction

This slice does not change:

- skill loading/discovery
- subagent runtime orchestration
- tool registration/bootstrap
- resource scope resolution

## Recommended Approach

Create `/Users/shike/Desktop/babybot/babybot/resource_skill_runtime.py` with a focused helper responsible for choosing skill packs and formatting worker-facing prompt text. Keep `resource.py` as a facade with wrapper methods that preserve current method names and signatures.

## Compatibility Constraints

- `ResourceManager._select_skill_packs()` stays async and keeps the same signature
- `ResourceManager._build_worker_sys_prompt()` keeps the same signature and wording
- `_format_skill_catalog()` and `_format_skill_catalog_for_lease()` remain callable from tests
- explicit `skill_ids` must still accept both skill names and `skill.<slug>` resource ids
