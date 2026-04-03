---
name: skill-manager
description: Use when creating, updating, installing, enabling, disabling, deleting, or reloading babybot skills. This is the single entry point for all skill lifecycle operations.
include_groups: code
---

# Skill Manager

Use this skill for any request that involves the lifecycle of a babybot skill:
creating a new skill, updating an existing one, enabling/disabling/deleting a skill,
or reloading a skill after a manual edit.

## Example Requests

- 帮我创建一个提取发票字段的技能
- 安装 pdf 解析技能
- 更新 weather 技能的 SKILL.md，加入更多示例
- 禁用 text-to-image 技能
- 启用 weather skill
- 删除我之前创建的 test-skill
- 我刚改了某个技能的脚本，帮我 reload 一下
- Create a new babybot skill for querying an internal API with helper scripts
- Make a prompt-only skill for code review checklists

## Target Selection

Before writing any files, decide where the skill lives:

- `workspace` — user-defined skills, root: `~/.babybot/workspace/skills/<skill-name>/`
- `builtin` — repository-maintained builtin skills, root: `skills/<skill-folder>/`

If the user gives an explicit path, use it. Otherwise use the correct default root.

## Workflow: Creating or Updating a Skill

1. Understand the skill from concrete examples.
   - Ask what user requests should trigger the skill (if not already clear).
   - Aim for 2–4 representative examples.
2. Decide what resources are actually needed.
   - `prompt`: guidance and references, no scripts needed
   - `scripts/`: deterministic repeated operations exposed as callable tools
   - `references/`: long documentation read only when needed
   - `assets/`: templates or output resources
   - `hybrid`: combine prompt guidance with helper scripts when both are justified
3. Initialize or repair the skill structure:

```bash
python skills/skill-manager/scripts/init_skill.py "<skill name>" \
  --target workspace \
  --tool-kind hybrid \
  --summary "extracting structured data from receipts and invoices" \
  --example-request "帮我提取这张小票里的金额和日期" \
  --example-request "parse this invoice photo into json"
```

   For builtin skills:

```bash
python skills/skill-manager/scripts/init_skill.py "<skill name>" \
  --target builtin \
  --tool-kind scripts \
  --summary "running a stable image-processing workflow" \
  --resources scripts
```

4. Edit `SKILL.md`.
   - frontmatter must include `name` and `description`
   - `description` must say when to use the skill in user-facing language
   - add `Example Requests` early so routing can tokenize them
   - keep the body procedural and concise
5. Add or update only justified resources.
   - Public Python scripts in `scripts/` must expose at least one top-level callable function.
   - Do not leave upstream demo CLIs as the public tool surface — wrap them.
   - Helper modules and demo entrypoints should use a leading underscore (`_client.py`, `_demo.py`) to avoid auto-registration.
   - Write files directly into the target skill folder, not into `/workspace/output`.
   - Replace or delete placeholder files created by scaffolding before finishing.
6. Validate:

```bash
python skills/skill-manager/scripts/quick_validate.py <skill-directory>
```

7. Hot-reload so the skill is available immediately:
   - Call the `reload_skill` tool with the skill directory path.
   - Only call this after validation passes.
8. Smoke-test the actual public tool entrypoints with representative arguments.
9. Do not claim completion until validation passes, the skill is hot-reloaded,
   and the public tool entrypoints have been exercised.

## Workflow: Managing an Existing Skill

For enable, disable, delete, and reload operations:

1. Call `list_admin_skills` first — use the returned name or resource id as the source of truth.
   Do not invent skill names or paths.
2. Apply the operation:
   - `enable_skill` / `disable_skill` — after confirming the target from `list_admin_skills`.
   - `delete_skill` — only for workspace custom skills the user explicitly asked to remove.
     Do not use on builtin skills.
   - `reload_skill` — only when a skill directory or `SKILL.md` was intentionally changed
     and the updated skill must become available in the current runtime.
3. Return a concrete result stating what changed and which skill name or path was used.

## Skill Shapes

- `prompt` — workflow guidance, checklists, analysis, policy-heavy skills
- `scripts` — deterministic helpers exposed as callable tools
- `hybrid` — both reasoning guidance and helper tools; keep prompt focused,
  put heavy detail in `references/`

## Trigger Quality

- Write `description` for discovery, not marketing.
- Put realistic user utterances in `Example Requests`.
- Include synonyms, domain nouns, and bilingual examples when applicable.
- Add explicit boundaries so the skill does not over-trigger.

## Writing Guidance

- Assume the agent is smart; only include non-obvious procedural guidance.
- Keep root clean: only `SKILL.md`, `scripts/`, `references/`, `assets/`.
- Put detailed reference material in `references/`, not in `SKILL.md`.

## Constraints

- Do not add `README.md`, changelogs, or process notes unless the user asks.
- Do not invent skill names, resource ids, or file paths. If lookup fails, report it.
- Do not use `delete_skill` for builtin skills or anything outside the workspace skill directory.
- Do not change babybot runtime skill discovery to support a new skill layout.
- For workspace skills, never write generated files into the project root or `/workspace/output`.
- For builtin skills, keep files inside repository `skills/`.
