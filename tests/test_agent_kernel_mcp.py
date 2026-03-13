from __future__ import annotations

import asyncio

from babybot.agent_kernel import ToolContext, ToolRegistry, register_mcp_tools
from babybot.agent_kernel.mcp import MCPToolDescriptor


class FakeMCPClient:
    """In-memory MCP client for integration testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[MCPToolDescriptor]:
        return [
            MCPToolDescriptor(
                name="fake_sum",
                description="sum two ints",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            )
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return {"result": int(arguments["a"]) + int(arguments["b"])}


def test_register_mcp_tools_and_invoke_registered_tool() -> None:
    async def _run() -> None:
        registry = ToolRegistry()
        client = FakeMCPClient()

        count = await register_mcp_tools(registry, client, group="mcp_math")
        assert count == 1

        registered = registry.get("fake_sum")
        assert registered is not None
        assert registered.group == "mcp_math"

        result = await registered.tool.invoke({"a": 2, "b": 5}, ToolContext())
        assert result.ok is True
        assert "7" in result.content
        assert client.calls == [("fake_sum", {"a": 2, "b": 5})]

    asyncio.run(_run())
