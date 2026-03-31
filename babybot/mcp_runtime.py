"""MCP runtime adapters for kernel tool registration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client

    _MCP_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    ClientSession = Any  # type: ignore[assignment]
    StdioServerParameters = Any  # type: ignore[assignment]
    stdio_client = None
    streamable_http_client = None
    _MCP_IMPORT_ERROR = exc

from .agent_kernel import MCPToolDescriptor

logger = logging.getLogger(__name__)


def _require_mcp_package() -> None:
    """Raise a clear error only when MCP support is actually used."""
    if _MCP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MCP support requires the optional 'mcp' package. "
            "Install project dependencies again so 'mcp' is available."
        ) from _MCP_IMPORT_ERROR


class BaseMCPRuntimeClient:
    """Shared MCP runtime client interface."""

    async def connect(self) -> None:
        raise NotImplementedError

    async def list_tools(self) -> list[MCPToolDescriptor]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class StdioMCPRuntimeClient(BaseMCPRuntimeClient):
    """MCP client over stdio transport."""

    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        _require_mcp_package()
        self._params = StdioServerParameters(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
        )
        self._stdio_cm: Any | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        self._stdio_cm = stdio_client(self._params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        try:
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            await self._session.initialize()
        except Exception:
            # Clean up the stdio context manager on partial init failure.
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_cm = None
            self._session = None
            raise

    async def list_tools(self) -> list[MCPToolDescriptor]:
        if not self._session:
            return []
        result = await self._session.list_tools()
        tools: list[MCPToolDescriptor] = []
        for tool in result.tools:
            tools.append(
                MCPToolDescriptor(
                    name=tool.name,
                    description=tool.description or tool.title or tool.name,
                    input_schema=tool.inputSchema
                    or {"type": "object", "properties": {}},
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError("MCP session not connected")
        result = await self._session.call_tool(name=name, arguments=arguments)
        if result.structuredContent is not None:
            return result.structuredContent
        texts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts) if texts else str(result)

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_cm = None


class HttpMCPRuntimeClient(BaseMCPRuntimeClient):
    """MCP client over streamable HTTP transport."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        _require_mcp_package()
        self._url = url
        self._headers = dict(headers or {})
        self._http_cm: Any | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        self._http_client = httpx.AsyncClient(headers=self._headers or None)
        self._http_cm = streamable_http_client(
            self._url,
            http_client=self._http_client,
        )
        read_stream, write_stream, _get_session_id = await self._http_cm.__aenter__()
        try:
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            await self._session.initialize()
        except Exception:
            # Clean up the HTTP context manager on partial init failure.
            try:
                await self._http_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._http_cm = None
            if self._http_client is not None:
                try:
                    await self._http_client.aclose()
                except Exception:
                    pass
                self._http_client = None
            self._session = None
            raise

    async def list_tools(self) -> list[MCPToolDescriptor]:
        if not self._session:
            return []
        result = await self._session.list_tools()
        return [
            MCPToolDescriptor(
                name=tool.name,
                description=tool.description or tool.title or tool.name,
                input_schema=tool.inputSchema or {"type": "object", "properties": {}},
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError("MCP session not connected")
        result = await self._session.call_tool(name=name, arguments=arguments)
        if result.structuredContent is not None:
            return result.structuredContent
        texts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts) if texts else str(result)

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._http_cm is not None:
            try:
                await self._http_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._http_cm = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None


def close_clients_best_effort(clients: dict[str, BaseMCPRuntimeClient]) -> None:
    """Close MCP clients from sync context without failing callers."""

    async def _close_all() -> None:
        for client in clients.values():
            try:
                await client.close()
            except Exception as exc:
                logger.warning("Failed to close MCP client: %s", exc)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_close_all())
    except RuntimeError:
        try:
            asyncio.run(_close_all())
        except Exception:
            pass
