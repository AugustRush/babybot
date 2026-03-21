---
name: auto-skill-creator
description: Use when creating or updating babybot, Codex, or Claude skills that need deterministic scaffolding, resource planning, and final validation.
include_groups: code
---

# Auto Skill Creator

Use this skill when a user asks to create a new skill or update an existing one.

## Overview

This skill upgrades skill authoring from ad-hoc file writing to a deterministic workflow:
- decide whether the target is a builtin skill or a workspace skill
- plan the skill around concrete user examples and trigger phrases
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

## Example Requests

- 帮我创建一个技能，用来提取发票图片里的结构化字段
- Create a new babybot skill for querying an internal API with helper scripts
- Update this existing skill so it has better trigger examples and validation
- Make a prompt-only skill for code review checklists without adding scripts
- Turn this rough skill folder into a proper hybrid skill with references and tools

## Workflow

1. Understand the skill from concrete examples.
   - Ask what user requests should trigger the skill.
   - Ask for 2 to 4 representative examples if they are not already clear.
2. Decide what resources are actually needed.
   - `prompt`: mostly guidance and references, usually no scripts needed at first
   - `scripts/` for deterministic repeated operations
   - `references/` for long documentation that should be read only when needed
   - `assets/` for templates or output resources
   - `hybrid`: combine prompt guidance with helper scripts when both are justified
3. Initialize or repair the skill structure with:

```bash
python skills/auto_skill_creator/scripts/init_skill.py "<skill name>" \
  --target workspace \
  --tool-kind hybrid \
  --summary "extracting structured data from receipts and invoices" \
  --example-request "帮我提取这张小票里的金额和日期" \
  --example-request "parse this invoice photo into json"
```

   For builtin skills:

```bash
python skills/auto_skill_creator/scripts/init_skill.py "<skill name>" \
  --target builtin \
  --tool-kind scripts \
  --summary "running a stable image-processing workflow" \
  --resources scripts
```

4. Edit `SKILL.md`.
   - frontmatter must include `name` and `description`
   - `description` must say when to use the skill in current-user language
   - add `Example Requests` early in the document so current babybot routing can tokenize them
   - keep the body procedural and concise
5. Add or update only the resources that are justified by repeated use.
   - Any public Python script in `scripts/` must expose at least one top-level callable function that the agent can invoke directly.
   - Do not leave upstream demo CLIs as the public tool surface. Wrap them in a stable function such as `generate_image(prompt: str, ...) -> str`.
   - Helper modules, pure clients, and demo entrypoints should be named with a leading underscore such as `_client.py` or `_demo.py` so they are not auto-registered as tools.
6. Validate before finishing:

```bash
python skills/auto_skill_creator/scripts/quick_validate.py <skill-directory>
```

7. Smoke-test the actual public tool entrypoints, not just the folder structure.
   - If the skill talks to an external API or CLI, run the public wrapper with representative arguments and capture the real failure or success mode.
8. Do not claim completion until validation passes and the public tool entrypoints have been exercised.

## Skill Shapes

- `prompt`
  - best for workflow guidance, checklists, analysis, or policy-heavy skills
  - do not add `scripts/` unless the same execution logic keeps getting rewritten
- `scripts`
  - best for deterministic helpers that the runtime should expose as callable tools
  - every public script should map to at least one stable top-level function
- `hybrid`
  - best when the agent needs both reasoning guidance and helper tools
  - keep the prompt focused on routing and decision-making; put heavy detail in `references/`

## Trigger Quality

- Write `description` for discovery, not marketing.
- Put realistic user utterances in `Example Requests`; this matters because babybot derives search terms from `description`, skill name, and the early body text.
- Include synonyms, domain nouns, and at least one Chinese or English example if the skill is likely to be used bilingually.
- Add explicit boundaries so the skill does not over-trigger.

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
- Prefer small agent-facing wrappers over copying full upstream demo programs into `scripts/`.
- Keep `Example Requests` high in the document so they land inside the portion indexed by discovery.

## Constraints

- Do not add `README.md`, changelogs, or process notes unless the user explicitly asks.
- Keep `SKILL.md` focused on triggering conditions, workflow, and expected outputs.
- Do not change `babybot` runtime skill discovery just to support a new skill layout.
- For workspace skills, never write generated files into project root.
- For builtin skills, keep the files inside repository `skills/`.
