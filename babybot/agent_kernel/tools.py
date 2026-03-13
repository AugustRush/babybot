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
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def schema(self) -> dict[str, Any]:
        ...

    async def invoke(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


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

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list(self, lease: ToolLease | None = None) -> list[RegisteredTool]:
        lease = lease or ToolLease()
        include_groups = set(lease.include_groups)
        include_tools = set(lease.include_tools)
        exclude_tools = set(lease.exclude_tools)

        selected: list[RegisteredTool] = []
        for name, registered in self._tools.items():
            if name in exclude_tools:
                continue
            if include_tools and name not in include_tools:
                continue
            if include_groups and registered.group not in include_groups:
                continue
            selected.append(registered)
        return selected

    def tool_schemas(self, lease: ToolLease | None = None) -> tuple[dict[str, Any], ...]:
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
