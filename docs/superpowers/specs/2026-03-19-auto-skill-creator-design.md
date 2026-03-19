# Auto Skill Creator Design

**Goal:** Upgrade `skills/auto_skill_creator` from a minimal instruction-only skill into a practical skill authoring package for `babybot`, with deterministic initialization and validation for both builtin and workspace skills.

**Constraints**
- Keep `babybot` runtime skill discovery unchanged.
- Support both builtin skills and user workspace skills.
- Only create/update local skill files and validate them.
- Do not introduce packaging/distribution behavior in this round.

**Current State**
- `babybot` discovers skills from two roots:
  - builtin: repository `skills/`
  - workspace: `Config.workspace_skills_dir`
- `skills/auto_skill_creator` currently contains only `SKILL.md`.
- There is no deterministic scaffold or validation flow for skill creation.

**Design**

## 1. Keep the skill as a normal builtin skill package

`auto_skill_creator` remains under `skills/auto_skill_creator/` so it stays compatible with current discovery rules.

New files:
- `skills/auto_skill_creator/SKILL.md`
- `skills/auto_skill_creator/scripts/init_skill.py`
- `skills/auto_skill_creator/scripts/quick_validate.py`

No changes to `ResourceManager._discover_skills()` or config path handling are required.

## 2. Add deterministic initialization

`init_skill.py` will:
- normalize skill names to lowercase hyphen-case
- create the target skill directory
- generate a `SKILL.md` template with valid frontmatter
- optionally create `scripts/`, `references/`, `assets/`
- support two targets:
  - `workspace`: default path rooted at `~/.babybot/workspace/skills`
  - `builtin`: path rooted at repository `skills/`
- support explicit `--path` override when needed

The script should be reusable both by agents and developers.

## 3. Add lightweight validation

`quick_validate.py` will validate a skill directory against `babybot`'s current expectations:
- `SKILL.md` must exist
- frontmatter must include `name` and `description`
- `name` must match the folder name and use hyphen-case
- `description` must not be empty or placeholder text
- skill root may only contain:
  - `SKILL.md`
  - `scripts/`
  - `references/`
  - `assets/`

Validation should stay aligned with `babybot`'s current loader and avoid speculative fields or packaging constraints.

## 4. Upgrade the skill instructions

`SKILL.md` will move from a minimal checklist to a fuller creation workflow:
- identify whether the target is builtin or workspace
- collect concrete usage examples and trigger phrases
- decide whether scripts/references/assets are needed
- run `init_skill.py`
- edit `SKILL.md` and resources
- run `quick_validate.py`
- stop only when the skill validates cleanly

The guidance should import the stronger design principles from `nanobot` while staying concise and specific to `babybot`.

## 5. Testing

Add script-level tests that cover:
- workspace target initialization
- builtin target initialization
- invalid resource type rejection
- name normalization
- successful validation of a generated skill
- validation failures for malformed frontmatter or unexpected root files

These tests should verify deterministic behavior directly, not only through the skill prose.
