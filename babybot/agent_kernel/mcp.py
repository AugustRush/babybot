"""MCP integration as tool adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .tools import ToolContext, ToolRegistry, ToolResult


@dataclass
class MCPToolDescriptor:
    """Tool descriptor fetched from MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClientPort(Protocol):
    """Minimal MCP client contract needed by this kernel."""

    async def list_tools(self) -> list[MCPToolDescriptor]:
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        ...


class MCPToolAdapter:
    """Wrap one MCP tool descriptor as a framework Tool."""

    def __init__(self, client: MCPClientPort, descriptor: MCPToolDescriptor):
        self._client = client
        self._descriptor = descriptor

    @property
    def name(self) -> str:
        return self._descriptor.name

    @property
    def description(self) -> str:
        return self._descriptor.description

    @property
    def schema(self) -> dict[str, Any]:
        return self._descriptor.input_schema

    async def invoke(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            raw = await self._client.call_tool(self.name, args)
            return ToolResult(ok=True, content=str(raw))
        except Exception as exc:  # pragma: no cover - adapter boundary
            return ToolResult(ok=False, error=str(exc))


async def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClientPort,
    group: str = "mcp",
) -> int:
    """Discover and register all MCP tools into the registry."""
    descriptors = await client.list_tools()
    for descriptor in descriptors:
        registry.register(MCPToolAdapter(client, descriptor), group=group)
    return len(descriptors)
