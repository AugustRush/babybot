---
name: auto-skill-creator
description: Use this skill when the user asks to create or update an Claude/AgentScope/Codex skill. It provides a deterministic workflow to scaffold a skill directory with a valid SKILL.md frontmatter and minimal runnable structure.
---

# Auto Skill Creator

Use this skill when a user asks to create a new skill or modify an existing skill.

## Goal

Produce a valid skill folder quickly with:
- `SKILL.md` (required)
- concise metadata and instructions
- optional subfolders only when required (`scripts/`, `references/`, `assets/`)

## Workflow

1. Confirm target skill path and intended purpose from user request.
   - User-defined skills must be created under `~/.babybot/workspace/skills/<skill_name>/`.
2. Create/Update `SKILL.md` with YAML frontmatter:
   - `name`
   - `description` (must include when to use the skill)
3. Keep SKILL body concise:
   - Trigger conditions
   - Step-by-step execution
   - Expected outputs
4. Add helper files only if needed:
   - `scripts/` for deterministic repeated tasks
   - `references/` for long domain docs
   - `assets/` for templates or output resources
   - Any generated resources/state files should stay inside this skill directory.
5. Validate:
   - Directory exists
   - `SKILL.md` frontmatter parses
   - Content is action-oriented and under a few hundred lines

## Constraints

- Do not add extra docs like README/CHANGELOG unless user explicitly asks.
- Prefer minimal structure first; expand only for repeated/fragile workflows.
- Keep instructions implementation-focused, not conceptual.
- Never write generated skill files into project root when creating user custom skills.
