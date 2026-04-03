---
name: agent-admin
description: Use when the request is to inspect or change the assistant profile, identity, persona, or system-level configuration. Do not use for skill creation, installation, enabling, disabling, or reloading — use skill-manager for all skill lifecycle operations.
include_groups:
  - admin
  - basic
---

# Agent Admin

## Example Requests

- 查看当前 assistant profile
- 把 assistant profile 替换成新的版本
- 把 assistant 的名字改成小助手
- 更新 assistant 的角色描述，加入新的定位
- Show me the current assistant profile and tell me what role it defines

## Workflow

1. Call `get_assistant_profile` first unless the user explicitly wants a blind overwrite.
2. Apply the smallest change that matches the request.
   - Use `set_assistant_profile` for profile edits.
   - Default to `mode="replace"` unless the user clearly asks to append.
3. Return a concrete result stating what was inspected or changed.

## Constraints

- Scope is limited to assistant profile / identity editing only.
- Do not use this skill for any skill lifecycle operations (create, install, enable, disable, delete, reload). Use skill-manager for those.
- Do not edit profile files by hand when `set_assistant_profile` can perform the change safely.
- Do not invent profile keys or field names.
