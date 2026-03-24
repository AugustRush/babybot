from __future__ import annotations

from typing import Any


def build_reload_skill_tool(owner: Any) -> Any:
    def reload_skill(skill_path: str) -> str:
        """Reload a skill from its directory so it becomes available without restart.

        Call this after creating or updating a skill (e.g. after validate_skill
        reports success) to make the skill immediately usable in the current
        session.

        Args:
            skill_path: Absolute or workspace-relative path to the skill
                directory (the folder that contains SKILL.md).
        """
        return owner.reload_skill(skill_path)

    return reload_skill


def iter_skill_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return ((build_reload_skill_tool(owner), "basic"),)
