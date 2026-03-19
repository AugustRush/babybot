---
name: auto-skill-creator
description: Use this skill when creating or updating babybot, Codex, Claude, or AgentScope skills. It scaffolds builtin or workspace skills, guides resource planning, and validates the final structure before completion.
---

# Auto Skill Creator

Use this skill when a user asks to create a new skill or update an existing one.

## Overview

This skill upgrades skill authoring from ad-hoc file writing to a deterministic workflow:
- decide whether the target is a builtin skill or a workspace skill
- plan the skill around concrete user examples
- scaffold the directory with `init_skill.py`
- edit `SKILL.md` and bundled resources
- validate the result with `quick_validate.py`

## Target Selection

Choose one target before writing files:

- `workspace`
  - for user-defined skills
  - default root: `~/.babybot/workspace/skills/<skill-name>/`
- `builtin`
  - for repository-maintained builtin skills
  - default root: repository `skills/<skill-folder>/`

If the user explicitly gives a path, use that path. Otherwise stick to the correct default root for the selected target.

## Workflow

1. Understand the skill from concrete examples.
   - Ask what user requests should trigger the skill.
   - Ask for 2 to 4 representative examples if they are not already clear.
2. Decide what resources are actually needed.
   - `scripts/` for deterministic repeated operations
   - `references/` for long documentation that should be read only when needed
   - `assets/` for templates or output resources
3. Initialize or repair the skill structure with:

```bash
python skills/auto_skill_creator/scripts/init_skill.py "<skill name>" --target workspace --resources scripts,references
```

   For builtin skills:

```bash
python skills/auto_skill_creator/scripts/init_skill.py "<skill name>" --target builtin --resources scripts
```

4. Edit `SKILL.md`.
   - frontmatter must include `name` and `description`
   - `description` must say when to use the skill
   - keep the body procedural and concise
5. Add or update only the resources that are justified by repeated use.
6. Validate before finishing:

```bash
python skills/auto_skill_creator/scripts/quick_validate.py <skill-directory>
```

7. Do not claim completion until validation passes.

## Writing Guidance

- Assume the agent is already smart; only include non-obvious procedural guidance.
- Prefer short workflows over long conceptual explanations.
- Keep the root clean. Only these entries belong in the skill root:
  - `SKILL.md`
  - `scripts/`
  - `references/`
  - `assets/`
- Put detailed reference material in `references/`, not in `SKILL.md`.
- Add scripts when the same fragile logic would otherwise be rewritten repeatedly.

## Constraints

- Do not add `README.md`, changelogs, or process notes unless the user explicitly asks.
- Keep `SKILL.md` focused on triggering conditions, workflow, and expected outputs.
- Do not change `babybot` runtime skill discovery just to support a new skill layout.
- For workspace skills, never write generated files into project root.
- For builtin skills, keep the files inside repository `skills/`.
