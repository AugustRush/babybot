from __future__ import annotations

from typing import Any


def build_inspect_runtime_flow_tool(owner: Any) -> Any:
    async def inspect_runtime_flow(flow_id: str = "", chat_key: str = "") -> str:
        """Inspect a runtime flow snapshot with child task states and recent events."""
        return owner._inspect_runtime_flow(flow_id=flow_id, chat_key=chat_key)

    return inspect_runtime_flow


def build_inspect_chat_context_tool(owner: Any) -> Any:
    async def inspect_chat_context(chat_key: str = "", query: str = "") -> str:
        """Inspect the current chat context, including Hot/Warm/Cold layers and memory records."""
        return owner._inspect_chat_context(chat_key=chat_key, query=query)

    return inspect_chat_context


def build_inspect_policy_tool(owner: Any) -> Any:
    async def inspect_policy(chat_key: str = "", decision_kind: str = "") -> str:
        """Inspect current orchestration policy summaries and effective action stats."""
        return owner._inspect_policy(chat_key=chat_key, decision_kind=decision_kind)

    return inspect_policy


def build_inspect_tools_tool(owner: Any) -> Any:
    async def inspect_tools(
        query: str = "",
        group: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Inspect available tools, grouped by tool group, active state, and schema summary."""
        return owner._inspect_tools(
            query=query,
            group=group,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    return inspect_tools


def build_inspect_skills_tool(owner: Any) -> Any:
    async def inspect_skills(
        query: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Inspect discovered skills, their source, active state, tool group, and exposed tools."""
        return owner._inspect_skills(
            query=query,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    return inspect_skills


def build_inspect_skill_load_errors_tool(owner: Any) -> Any:
    async def inspect_skill_load_errors(limit: int = 20) -> str:
        """Inspect recent skill loading failures with path and error details."""
        return owner._inspect_skill_load_errors(limit=limit)

    return inspect_skill_load_errors


def iter_observability_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        (build_inspect_runtime_flow_tool(owner), "basic"),
        (build_inspect_chat_context_tool(owner), "basic"),
        (build_inspect_policy_tool(owner), "basic"),
        (build_inspect_tools_tool(owner), "basic"),
        (build_inspect_skills_tool(owner), "basic"),
        (build_inspect_skill_load_errors_tool(owner), "basic"),
    )
