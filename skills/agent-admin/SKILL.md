---
name: agent-admin
description: Use when the request is to inspect or change the assistant profile, list skills, enable or disable a skill, or reload a skill after an intentional update.
include_groups:
  - admin
  - basic
---

# Agent Admin

## Example Requests

- 查看当前 assistant profile，然后替换成新的版本
- 列出技能并只看激活项，然后禁用 image skill
- 启用 weather skill，如果存在再继续
- 删除一个 workspace 自定义技能
- 我刚改了某个技能的 SKILL.md，帮我 reload

## Workflow

1. Use builtin admin tools first.
   - For profile work, call `get_assistant_profile` before changing anything unless the user explicitly wants a blind overwrite.
   - For skill state work, call `list_admin_skills` first and use the returned name or resource id as the source of truth.
2. Apply the smallest change that matches the request.
   - Use `set_assistant_profile` for profile edits. Default to `mode="replace"` unless the user clearly asks to append.
   - Use `enable_skill` or `disable_skill` only after confirming the target skill from `list_admin_skills`.
   - Use `delete_skill` only for workspace custom skills that the user explicitly asked to remove.
3. Use `reload_skill` only when a skill directory or `SKILL.md` was intentionally changed and the updated skill must become available in the current runtime.
4. Return a concrete result that says what you inspected, what changed, and which skill name or path was used.

## Constraints

- Keep scope limited to assistant profile editing and skill enable, disable, or reload actions.
- Keep scope limited to assistant profile editing and skill enable, disable, delete, or reload actions.
- Do not invent skill names, resource ids, or file paths. If lookup fails, report that directly.
- Do not edit profile files by hand when `set_assistant_profile` can perform the change safely.
- Do not use `delete_skill` for builtin skills or anything outside the workspace skill directory.
- Do not use this skill for broad skill authoring or unrelated repo maintenance.
