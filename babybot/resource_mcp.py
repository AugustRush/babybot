from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent_kernel import register_mcp_tools
from .mcp_runtime import HttpMCPRuntimeClient, StdioMCPRuntimeClient
from .resource_models import ToolGroup

if TYPE_CHECKING:
    from .resource import ResourceManager

logger = logging.getLogger(__name__)


class ResourceMCPHelper:
    """MCP setup and environment normalization for ResourceManager."""

    def __init__(self, owner: ResourceManager) -> None:
        self._owner = owner

    async def register_mcp_servers(self, mcp_servers: dict[str, dict]) -> None:
        for name, server_conf in mcp_servers.items():
            try:
                mcp_type = server_conf.get("type", "http")
                group_name = server_conf.get("group_name", name)
                if mcp_type == "stdio":
                    command, args, cwd, env = self.prepare_mcp_stdio_launch(
                        name=name,
                        command=server_conf["command"],
                        args=server_conf.get("args", []),
                        server_conf=server_conf,
                    )
                    await self.register_mcp_stdio(
                        name=name,
                        command=command,
                        args=args,
                        cwd=cwd,
                        env=env,
                        group_name=group_name,
                    )
                else:
                    transport = server_conf.get("transport", "streamable_http")
                    url, headers = self.prepare_mcp_http_launch(
                        name=name,
                        url=server_conf["url"],
                        server_conf=server_conf,
                    )
                    await self.register_mcp(
                        name=name,
                        url=url,
                        headers=headers,
                        transport=transport,
                        group_name=group_name,
                    )
                if server_conf.get("active", False):
                    self._owner.activate_groups([group_name])
            except Exception as exc:
                logger.warning("Failed to register MCP server %s: %s", name, exc)

    async def register_mcp(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        transport: str = "streamable_http",
        group_name: str = "mcp",
    ) -> None:
        if transport not in {"streamable_http", "http"}:
            logger.warning(
                "MCP transport '%s' is not supported yet for '%s'.",
                transport,
                name,
            )
            return
        client = HttpMCPRuntimeClient(url=url, headers=headers)
        await client.connect()
        self._owner.mcp_clients[name] = client
        self._owner.mcp_server_groups[name] = group_name
        if group_name not in self._owner.groups:
            self._owner.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"MCP tools from {name}",
                active=False,
            )
        await register_mcp_tools(self._owner.registry, client, group=group_name)

    async def register_mcp_stdio(
        self,
        name: str,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        group_name: str = "mcp",
    ) -> None:
        client = StdioMCPRuntimeClient(command=command, args=args, cwd=cwd, env=env)
        await client.connect()
        self._owner.mcp_clients[name] = client
        self._owner.mcp_server_groups[name] = group_name
        if group_name not in self._owner.groups:
            self._owner.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"MCP tools from {name}",
                active=False,
            )
        await register_mcp_tools(self._owner.registry, client, group=group_name)

    def prepare_mcp_stdio_launch(
        self,
        name: str,
        command: str,
        args: list[str],
        server_conf: dict[str, Any],
    ) -> tuple[str, list[str], str | None, dict[str, str] | None]:
        defaults = self.build_mcp_stdio_env(name)
        env = self.normalize_stdio_mcp_env(server_conf.get("env"), defaults=defaults)
        cwd = self.normalize_stdio_mcp_path(server_conf.get("cwd"))
        if cwd is None:
            cwd = env["BABYBOT_MCP_ARTIFACT_ROOT"]
        Path(cwd).mkdir(parents=True, exist_ok=True)
        return command, list(args), cwd, env or None

    def normalize_stdio_mcp_env(
        self,
        raw_env: Any,
        *,
        defaults: dict[str, str] | None = None,
    ) -> dict[str, str]:
        normalized: dict[str, str] = dict(defaults or {})
        if not isinstance(raw_env, dict):
            return normalized
        for key, value in raw_env.items():
            if value is None:
                continue
            normalized[str(key)] = self.normalize_mcp_mapping_value(str(key), value)
        return normalized

    def prepare_mcp_http_launch(
        self,
        name: str,
        url: str,
        server_conf: dict[str, Any],
    ) -> tuple[str, dict[str, str] | None]:
        headers = self.normalize_http_mcp_headers(
            server_conf.get("headers"),
            defaults=self.build_mcp_http_headers(name),
        )
        return url, headers or None

    @staticmethod
    def normalize_stdio_mcp_path(value: Any) -> str | None:
        if value in (None, ""):
            return None
        text = os.path.expandvars(str(value))
        return str(Path(text).expanduser().resolve())

    def normalize_http_mcp_headers(
        self,
        raw_headers: Any,
        *,
        defaults: dict[str, str] | None = None,
    ) -> dict[str, str]:
        normalized: dict[str, str] = dict(defaults or {})
        if not isinstance(raw_headers, dict):
            return normalized
        for key, value in raw_headers.items():
            if value is None:
                continue
            normalized[str(key)] = self.normalize_mcp_mapping_value(str(key), value)
        return normalized

    def build_mcp_stdio_env(self, name: str) -> dict[str, str]:
        metadata = self.build_mcp_runtime_metadata(name)
        return {
            "BABYBOT_MCP_SERVER_NAME": metadata["server_name"],
            "BABYBOT_MCP_WORKSPACE_ROOT": metadata["workspace_root"],
            "BABYBOT_MCP_ARTIFACT_ROOT": metadata["artifact_root"],
        }

    def build_mcp_http_headers(self, name: str) -> dict[str, str]:
        metadata = self.build_mcp_runtime_metadata(name)
        return {
            "X-Babybot-Mcp-Server": metadata["server_name"],
            "X-Babybot-Workspace-Root": metadata["workspace_root"],
            "X-Babybot-Artifact-Root": metadata["artifact_root"],
        }

    def build_mcp_runtime_metadata(self, name: str) -> dict[str, str]:
        workspace_root = str(self._owner.config.workspace_dir.resolve())
        artifact_root = str(self.get_mcp_artifact_root(name))
        return {
            "server_name": name,
            "workspace_root": workspace_root,
            "artifact_root": artifact_root,
        }

    def get_mcp_artifact_root(self, name: str) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()) or "mcp"
        root = self._owner._get_output_dir() / "mcp" / safe_name
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def normalize_mcp_mapping_value(key: str, value: Any) -> str:
        text = os.path.expandvars(str(value))
        if ResourceMCPHelper.is_path_like_mcp_key(key):
            return str(Path(text).expanduser().resolve())
        return text

    @staticmethod
    def is_path_like_mcp_key(key: str) -> bool:
        upper = key.upper().replace("-", "_")
        return upper.endswith(("_ROOT", "_DIR", "_PATH"))
