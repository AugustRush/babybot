from __future__ import annotations

from typing import Any


def build_inspect_runtime_flow_tool(owner: Any) -> Any:
    def inspect_runtime_flow(flow_id: str = "", chat_key: str = "") -> str:
        """Inspect a runtime flow snapshot with child task states and recent events."""
        return owner._inspect_runtime_flow(flow_id=flow_id, chat_key=chat_key)

    return inspect_runtime_flow


def build_inspect_chat_context_tool(owner: Any) -> Any:
    def inspect_chat_context(chat_key: str = "", query: str = "") -> str:
        """Inspect the current chat context, including Hot/Warm/Cold layers and memory records."""
        return owner._inspect_chat_context(chat_key=chat_key, query=query)

    return inspect_chat_context


def iter_observability_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        (build_inspect_runtime_flow_tool(owner), "basic"),
        (build_inspect_chat_context_tool(owner), "basic"),
    )
