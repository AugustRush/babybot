"""Tool contracts and registry for the kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import ToolLease


@dataclass
class ToolContext:
    """Context visible to tools at invocation time."""

    session_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Normalized tool invocation output."""

    ok: bool
    content: str = ""
    error: str = ""
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """Tool protocol."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def schema(self) -> dict[str, Any]: ...

    async def invoke(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult: ...


@dataclass(frozen=True)
class RegisteredTool:
    tool: Tool
    group: str = "basic"


class ToolRegistry:
    """Tool registry with lease-based filtering."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: Tool, group: str = "basic") -> None:
        self._tools[tool.name] = RegisteredTool(tool=tool, group=group)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if the tool existed."""
        return self._tools.pop(name, None) is not None

    def unregister_group(self, group: str) -> list[str]:
        """Remove all tools in *group*. Returns the removed tool names."""
        to_remove = [n for n, r in self._tools.items() if r.group == group]
        for n in to_remove:
            del self._tools[n]
        return to_remove

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def known_names(self) -> set[str]:
        """Return the set of all registered tool names."""
        return set(self._tools.keys())

    def list(self, lease: ToolLease | None = None) -> list[RegisteredTool]:
        lease = lease or ToolLease()
        include_groups = set(lease.include_groups)
        include_tools = set(lease.include_tools)
        exclude_tools = set(lease.exclude_tools)

        selected: list[RegisteredTool] = []
        for name, registered in self._tools.items():
            if name in exclude_tools:
                continue
            if include_tools or include_groups:
                in_tools = name in include_tools if include_tools else False
                in_groups = (
                    registered.group in include_groups if include_groups else False
                )
                if not (in_tools or in_groups):
                    continue
            selected.append(registered)
        return selected

    def tool_schemas(
        self, lease: ToolLease | None = None
    ) -> tuple[dict[str, Any], ...]:
        schemas = []
        for registered in self.list(lease):
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": registered.tool.name,
                        "description": registered.tool.description,
                        "parameters": registered.tool.schema,
                    },
                }
            )
        return tuple(schemas)
