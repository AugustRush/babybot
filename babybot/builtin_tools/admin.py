from __future__ import annotations

from typing import Any, Literal


def build_get_assistant_profile_tool(owner: Any) -> Any:
    def get_assistant_profile() -> str:
        """Read the current assistant profile markdown used to steer agent behavior."""
        return owner.get_assistant_profile()

    return get_assistant_profile


def build_set_assistant_profile_tool(owner: Any) -> Any:
    def set_assistant_profile(
        content: str,
        mode: Literal["replace", "append"] = "replace",
    ) -> str:
        """Update the assistant profile markdown with replace or append semantics."""
        return owner.set_assistant_profile(content=content, mode=mode)

    return set_assistant_profile


def build_list_admin_skills_tool(owner: Any) -> Any:
    def list_admin_skills(
        query: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """List skills with active state, source, tool group, and exposed tools."""
        return owner.list_admin_skills(
            query=query,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    return list_admin_skills


def build_enable_skill_tool(owner: Any) -> Any:
    def enable_skill(skill_name: str) -> str:
        """Enable a discovered skill by name or resource id."""
        return owner.enable_skill(skill_name)

    return enable_skill


def build_disable_skill_tool(owner: Any) -> Any:
    def disable_skill(skill_name: str) -> str:
        """Disable a discovered skill by name or resource id."""
        return owner.disable_skill(skill_name)

    return disable_skill


def iter_admin_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        (build_get_assistant_profile_tool(owner), "admin"),
        (build_set_assistant_profile_tool(owner), "admin"),
        (build_list_admin_skills_tool(owner), "admin"),
        (build_enable_skill_tool(owner), "admin"),
        (build_disable_skill_tool(owner), "admin"),
    )
