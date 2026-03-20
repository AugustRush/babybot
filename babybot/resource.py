"""Lightweight resource manager based on the in-repo kernel."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import datetime
import inspect
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from .agent_kernel import (
    SkillPack,
    register_mcp_tools,
    ToolLease,
    ToolRegistry,
    ToolResult,
)
from .config import Config
from .mcp_runtime import (
    BaseMCPRuntimeClient,
    HttpMCPRuntimeClient,
    StdioMCPRuntimeClient,
    close_clients_best_effort,
)
from .resource_models import (
    CliToolSpec as _CliToolSpec,
    LoadedSkill,
    ResourceBrief,
    ScriptFunctionSpec as _ScriptFunctionSpec,
    SkillRuntimeConfig,
    ToolGroup,
)
from .resource_python_runner import ExternalPythonRunner
from .resource_skill_runtime import ResourceSkillRuntime
from .resource_scope import ResourceScopeHelper
from .resource_subagent_runtime import ResourceSubagentRuntime
from .resource_tool_loader import ResourceToolLoader
from .resource_workspace_tools import (
    WorkspaceToolSuite,
    check_shell_safety as _check_shell_safety,
)
from .resource_skill_loader import SkillLoader
from .worker import create_worker_executor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .channels.tools import ChannelTools, ChannelToolContext
    from .context import Tape, TapeStore
    from .cron import ScheduledTaskManager
    from .heartbeat import Heartbeat
    from .model_gateway import OpenAICompatibleGateway


class CallableTool:
    """Wrap a python callable into kernel Tool protocol."""

    def __init__(
        self,
        func: Any,
        name: str,
        description: str,
        schema: dict[str, Any],
        preset_kwargs: dict[str, Any] | None = None,
        resource_manager: ResourceManager | None = None,
    ):
        self._func = func
        self._name = name
        self._description = description
        self._schema = schema
        self._preset_kwargs = dict(preset_kwargs or {})
        self._resource_manager = resource_manager

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    async def invoke(self, args: dict[str, Any], context: Any) -> ToolResult:
        try:
            kwargs = dict(self._preset_kwargs)
            kwargs.update(args or {})
            artifact_base = self._current_write_root()
            if inspect.iscoroutinefunction(self._func):
                value = await self._func(**kwargs)
            else:
                # Run sync callables in a thread to avoid blocking the loop.
                # Use contextvars.copy_context() so channel context etc.
                # are visible inside the thread (Python <3.12 doesn't copy
                # context automatically in run_in_executor).
                # Also chdir to the active write root so relative paths
                # (e.g. result_image.jpg) resolve inside the workspace.
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                if self._resource_manager is not None:
                    write_root = self._resource_manager._get_active_write_root()
                else:
                    write_root = Path.cwd().resolve()
                artifact_base = write_root

                def _run_in_workspace() -> Any:
                    saved_cwd = os.getcwd()
                    try:
                        os.chdir(str(write_root))
                        return self._func(**kwargs)
                    finally:
                        os.chdir(saved_cwd)

                value = await loop.run_in_executor(None, ctx.run, _run_in_workspace)
            return ToolResult(
                ok=True,
                content=self._normalize_result(value),
                artifacts=self._collect_artifacts(value, base_dir=artifact_base),
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    @staticmethod
    def _normalize_result(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def _current_write_root(self) -> Path:
        if self._resource_manager is None:
            return Path.cwd().resolve()
        return self._resource_manager._get_active_write_root()

    @staticmethod
    def _looks_like_path_candidate(candidate: str) -> bool:
        text = candidate.strip()
        if not text:
            return False
        if "\n" in text or "\r" in text:
            return False
        if len(text) > 240:
            return False
        if text.startswith("{") or text.startswith("["):
            return False
        if "://" in text:
            return False
        suffix = Path(text).suffix.lower()
        if suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".pdf",
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".xlsx",
            ".pptx",
            ".docx",
            ".mp4",
            ".mp3",
            ".wav",
        }:
            return True
        return (
            "/" in text or "\\" in text or text.startswith(".") or text.startswith("~")
        )

    def _collect_artifacts(
        self,
        value: Any,
        *,
        base_dir: Path | None = None,
    ) -> list[str]:
        root = (base_dir or self._current_write_root()).resolve()
        found: list[str] = []
        seen: set[str] = set()
        source_seen: set[str] = set()

        def _add_path(raw: str) -> None:
            candidate = raw.strip().strip("\"'")
            if not candidate:
                return
            path = Path(os.path.expanduser(candidate))
            if not path.is_absolute():
                path = root / path
            try:
                resolved = path.resolve()
            except OSError:
                return
            try:
                if not resolved.is_file():
                    return
            except OSError:
                return
            source_key = str(resolved)
            if source_key in source_seen:
                return
            source_seen.add(source_key)
            resolved = self._normalize_artifact_path(resolved)
            resolved_str = str(resolved)
            if resolved_str not in seen:
                seen.add(resolved_str)
                found.append(resolved_str)

        def _walk(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, os.PathLike):
                _add_path(os.fspath(item))
                return
            if isinstance(item, str):
                if self._looks_like_path_candidate(item):
                    _add_path(item)
                for match in ResourceManager._MEDIA_PATH_RE.finditer(item):
                    _add_path(match.group(1))
                return
            if isinstance(item, dict):
                for value in item.values():
                    _walk(value)
                return
            if isinstance(item, (list, tuple, set)):
                for value in item:
                    _walk(value)

        _walk(value)
        return found

    def _normalize_artifact_path(self, resolved: Path) -> Path:
        if self._resource_manager is None:
            return resolved
        try:
            workspace = self._resource_manager.config.workspace_dir.resolve()
        except Exception:
            return resolved
        try:
            resolved.relative_to(workspace)
            return resolved
        except ValueError:
            pass

        output_dir = self._resource_manager._get_output_dir()
        target = output_dir / resolved.name
        if target == resolved:
            return resolved
        stem = resolved.stem
        suffix = resolved.suffix
        counter = 1
        while target.exists():
            try:
                if target.samefile(resolved):
                    return target.resolve()
            except OSError:
                pass
            target = output_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(resolved, target)
        return target.resolve()


class ResourceCatalog:
    """Read-only resource lookup surface exposed to orchestration."""

    def __init__(self, manager: ResourceManager) -> None:
        self._manager = manager

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return self._manager._get_resource_briefs()

    def resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        return self._manager._resolve_resource_scope(
            resource_id=resource_id,
            require_tools=require_tools,
        )

    def search_resources(self, query: str | None = None) -> dict[str, Any]:
        return self._manager._search_resources(query)


class WorkerRuntime:
    """Execution runtime for sub-agent task launches."""

    def __init__(self, manager: ResourceManager) -> None:
        self._manager = manager

    def get_shared_gateway(self) -> "OpenAICompatibleGateway":
        return self._manager._get_shared_gateway()

    async def run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        return await self._manager._run_subagent_task(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
        )


class ResourceManager:
    """Centralized resource manager without external agent frameworks."""

    _orchestration_tools = {"create_worker", "dispatch_workers"}
    _MEDIA_PATH_RE = re.compile(
        r"(?:^|[\s'\"(])((?:/~|[~/])?[\w./]+(?:\.(?:png|jpg|jpeg|gif|bmp|webp|pdf|txt|md|json|yaml|yml|csv|xlsx|pptx|docx|mp4|mp3|wav)))"
    )

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.registry = ToolRegistry()
        self.groups: dict[str, ToolGroup] = {}
        self.mcp_clients: dict[str, BaseMCPRuntimeClient] = {}
        self.mcp_server_groups: dict[str, str] = {}
        self.channel_tools: dict[str, ChannelTools] = {}
        self.skills: dict[str, LoadedSkill] = {}
        self.scheduled_task_manager: ScheduledTaskManager | None = None
        self._shared_gateway: OpenAICompatibleGateway | None = None
        self._active_write_root: contextvars.ContextVar[str] = contextvars.ContextVar(
            "active_write_root",
            default=str(self.config.workspace_dir.resolve()),
        )
        self._current_task_lease: contextvars.ContextVar[ToolLease | None] = (
            contextvars.ContextVar(
                "current_task_lease",
                default=None,
            )
        )
        self._current_skill_ids: contextvars.ContextVar[tuple[str, ...] | None] = (
            contextvars.ContextVar(
                "current_skill_ids",
                default=None,
            )
        )
        self._python_probe_cache: dict[tuple[str, tuple[str, ...], str], str | None] = {}
        self.catalog = ResourceCatalog(self)
        self.runtime = WorkerRuntime(self)

        # Keep task-file management available even in CLI mode; gateway mode
        # later binds the live scheduler onto the same manager instance.
        from .cron import ScheduledTaskManager

        self.scheduled_task_manager = ScheduledTaskManager(self.config)
        self._load_config()
        self._register_builtin_tools()

    async def initialize_async(self) -> None:
        """MCP initialization hook."""
        mcp_servers = {
            k: v
            for k, v in self.config.get_mcp_servers().items()
            if not k.startswith("_")
        }
        if not mcp_servers:
            return None
        await self._register_mcp_servers(mcp_servers)
        return None

    def _catalog_view(self) -> ResourceCatalog:
        catalog = getattr(self, "catalog", None)
        if catalog is None:
            catalog = ResourceCatalog(self)
            self.catalog = catalog
        return catalog

    def _runtime_view(self) -> WorkerRuntime:
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            runtime = WorkerRuntime(self)
            self.runtime = runtime
        return runtime

    def _python_runner_view(self) -> ExternalPythonRunner:
        runner = getattr(self, "_python_runner", None)
        if runner is None:
            runner = ExternalPythonRunner(self.config)
            self._python_runner = runner
        return runner

    def _workspace_tools_view(self) -> WorkspaceToolSuite:
        tools = getattr(self, "_workspace_tools", None)
        if tools is None:
            tools = WorkspaceToolSuite(self)
            self._workspace_tools = tools
        return tools

    def _skill_loader_view(self) -> SkillLoader:
        loader = getattr(self, "_skill_loader", None)
        if loader is None:
            loader = SkillLoader(self)
            self._skill_loader = loader
        return loader

    def _resource_scope_view(self) -> ResourceScopeHelper:
        helper = getattr(self, "_resource_scope", None)
        if helper is None:
            helper = ResourceScopeHelper(self)
            self._resource_scope = helper
        return helper

    def _tool_loader_view(self) -> ResourceToolLoader:
        helper = getattr(self, "_tool_loader", None)
        if helper is None:
            helper = ResourceToolLoader(self)
            self._tool_loader = helper
        return helper

    def _subagent_runtime_view(self) -> ResourceSubagentRuntime:
        helper = getattr(self, "_subagent_runtime", None)
        if helper is None:
            helper = ResourceSubagentRuntime(self)
            self._subagent_runtime = helper
        return helper

    def _skill_runtime_view(self) -> ResourceSkillRuntime:
        helper = getattr(self, "_skill_runtime", None)
        if helper is None:
            helper = ResourceSkillRuntime(self)
            self._skill_runtime = helper
        return helper

    @staticmethod
    def _callable_tool_cls() -> type[CallableTool]:
        return CallableTool

    async def _register_mcp_servers(self, mcp_servers: dict[str, dict]) -> None:
        for name, server_conf in mcp_servers.items():
            try:
                mcp_type = server_conf.get("type", "http")
                group_name = server_conf.get("group_name", name)
                if mcp_type == "stdio":
                    await self.register_mcp_stdio(
                        name=name,
                        command=server_conf["command"],
                        args=server_conf.get("args", []),
                        group_name=group_name,
                    )
                else:
                    transport = server_conf.get("transport", "streamable_http")
                    await self.register_mcp(
                        name=name,
                        url=server_conf["url"],
                        transport=transport,
                        group_name=group_name,
                    )
                if server_conf.get("active", False):
                    self.activate_groups([group_name])
            except Exception as exc:
                logger.warning("Failed to register MCP server %s: %s", name, exc)

    async def register_mcp(
        self,
        name: str,
        url: str,
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
        client = HttpMCPRuntimeClient(url=url)
        await client.connect()
        self.mcp_clients[name] = client
        self.mcp_server_groups[name] = group_name
        if group_name not in self.groups:
            self.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"MCP tools from {name}",
                active=False,
            )
        await register_mcp_tools(self.registry, client, group=group_name)

    async def register_mcp_stdio(
        self,
        name: str,
        command: str,
        args: list[str],
        group_name: str = "mcp",
    ) -> None:
        client = StdioMCPRuntimeClient(command=command, args=args)
        await client.connect()
        self.mcp_clients[name] = client
        self.mcp_server_groups[name] = group_name
        if group_name not in self.groups:
            self.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"MCP tools from {name}",
                active=False,
            )
        await register_mcp_tools(self.registry, client, group=group_name)

    def _load_config(self) -> None:
        tool_groups = {
            k: v
            for k, v in self.config.get_tool_groups().items()
            if not k.startswith("_")
        }
        self._setup_tool_groups(tool_groups)
        custom_tools = {
            k: v
            for k, v in self.config.get_custom_tools().items()
            if not k.startswith("_")
        }
        if custom_tools:
            self._register_custom_tools(custom_tools)
        self._discover_workspace_tools()
        self._register_configured_skills(
            {
                k: v
                for k, v in self.config.get_agent_skills().items()
                if not k.startswith("_")
            }
        )
        self._discover_skills()

    def _register_configured_skills(self, configured: dict[str, dict]) -> None:
        self._skill_loader_view().register_configured_skills(configured)

    def _discover_skills(self) -> None:
        self._skill_loader_view().discover_skills()

    def _upsert_skill(self, skill: LoadedSkill) -> None:
        self.skills[skill.name.strip().lower()] = skill

    def _register_skill_tools(
        self,
        skill_name: str,
        skill_dir: Path,
        runtime: SkillRuntimeConfig | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        return self._skill_loader_view().register_skill_tools(
            skill_name,
            skill_dir,
            runtime=runtime,
            callable_tool_cls=self._callable_tool_cls(),
        )

    @staticmethod
    def _skip_skill_function_name(name: str) -> bool:
        return SkillLoader.skip_skill_function_name(name)

    @classmethod
    def _extract_function_specs_from_script(
        cls,
        script_path: Path,
    ) -> list[_ScriptFunctionSpec]:
        return SkillLoader.extract_function_specs_from_script(script_path)

    @classmethod
    def _extract_cli_tool_spec_from_script(
        cls,
        script_path: Path,
    ) -> _CliToolSpec | None:
        return SkillLoader.extract_cli_tool_spec_from_script(script_path)

    @classmethod
    def _schema_from_argparse_call(
        cls,
        call: ast.Call,
    ) -> tuple[dict[str, Any], bool, str | None]:
        return SkillLoader.schema_from_argparse_call(call)

    @staticmethod
    def _annotation_name_from_ast(node: ast.AST) -> str | None:
        return SkillLoader.annotation_name_from_ast(node)

    def _build_external_cli_script_callable(
        self,
        script_path: Path,
        cli_spec: _CliToolSpec,
        runtime: SkillRuntimeConfig | None = None,
    ) -> Any:
        return self._python_runner_view().build_external_cli_script_callable(
            self,
            script_path,
            cli_spec,
            runtime=runtime,
        )

    @staticmethod
    def _format_cli_argument(value: Any) -> str:
        return ExternalPythonRunner.format_cli_argument(value)

    @classmethod
    def _schema_from_ast_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, Any]:
        return SkillLoader.schema_from_ast_function(node)

    @staticmethod
    def _schema_for_ast_annotation(annotation: ast.AST | None) -> dict[str, Any]:
        return SkillLoader.schema_for_ast_annotation(annotation)

    def _build_external_skill_callable(
        self,
        script_path: Path,
        function_name: str,
        runtime: SkillRuntimeConfig | None = None,
    ) -> Any:
        return self._python_runner_view().build_external_skill_callable(
            self,
            script_path,
            function_name,
            runtime=runtime,
        )

    async def _invoke_external_skill_function(
        self,
        script_path: str,
        function_name: str,
        arguments: dict[str, Any],
        runtime: SkillRuntimeConfig | None = None,
    ) -> str:
        return await self._python_runner_view().invoke_external_skill_function(
            self,
            script_path=script_path,
            function_name=function_name,
            arguments=arguments,
            runtime=runtime,
            result_normalizer=CallableTool._normalize_result,
        )

    @staticmethod
    def _normalize_keywords(
        raw_keywords: Any,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        return SkillLoader.normalize_keywords(raw_keywords, fallback=fallback)

    @staticmethod
    def _normalize_phrases(
        raw_keywords: Any,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        return SkillLoader.normalize_phrases(raw_keywords, fallback=fallback)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return SkillLoader.tokenize(text)

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
        return SkillLoader.parse_frontmatter(text)

    @classmethod
    def _read_skill_document(cls, skill_dir: Path) -> tuple[dict[str, str], str]:
        return SkillLoader.read_skill_document(skill_dir)

    def _setup_tool_groups(self, user_groups: dict[str, dict]) -> None:
        defaults = {
            "basic": ToolGroup(
                "basic",
                "核心工具：定时/延时发送消息、任务调度管理（创建/修改/删除/列出定时任务）",
                active=True,
            ),
            "code": ToolGroup(
                "code",
                "Code execution and file operations",
                active=True,
                notes="Use these tools for coding and filesystem tasks.",
            ),
            "search": ToolGroup("search", "Search tools", active=False),
            "analysis": ToolGroup("analysis", "Analysis tools", active=False),
            "browser": ToolGroup("browser", "Browser automation tools", active=False),
        }
        self.groups = defaults
        for name, conf in user_groups.items():
            old = self.groups.get(name)
            self.groups[name] = ToolGroup(
                name=name,
                description=conf.get(
                    "description", old.description if old else f"{name} tools"
                ),
                notes=conf.get("notes", old.notes if old else ""),
                active=conf.get("active", old.active if old else False),
            )

    def _register_builtin_tools(self) -> None:
        self.register_tool(self.create_worker_tool(), group_name="basic")
        self.register_tool(self.dispatch_workers_tool(), group_name="basic")
        self.register_tool(self.list_scheduled_tasks_tool(), group_name="basic")
        self.register_tool(self.save_scheduled_task_tool(), group_name="basic")
        self.register_tool(self.create_scheduled_task_tool(), group_name="basic")
        self.register_tool(self.update_scheduled_task_tool(), group_name="basic")
        self.register_tool(self.delete_scheduled_task_tool(), group_name="basic")
        self.register_tool(self._workspace_execute_python_code, group_name="code")
        self.register_tool(self._workspace_execute_shell_command, group_name="code")
        self.register_tool(self._workspace_view_text_file, group_name="code")
        self.register_tool(self._workspace_write_text_file, group_name="code")
        self.register_tool(self._workspace_insert_text_file, group_name="code")

    @staticmethod
    def _resource_slug(value: str) -> str:
        return ResourceScopeHelper.resource_slug(value)

    def _skill_resource_id(self, skill: LoadedSkill) -> str:
        return self._resource_scope_view().skill_resource_id(skill)

    def _mcp_resource_id(self, server_name: str) -> str:
        return self._resource_scope_view().mcp_resource_id(server_name)

    def _group_resource_id(self, group_name: str) -> str:
        return self._resource_scope_view().group_resource_id(group_name)

    @staticmethod
    def _lease_to_dict(lease: ToolLease) -> dict[str, Any]:
        return ResourceScopeHelper.lease_to_dict(lease)

    def _get_current_task_lease_var(self) -> contextvars.ContextVar[ToolLease | None]:
        current = getattr(self, "_current_task_lease", None)
        if current is None:
            current = contextvars.ContextVar("current_task_lease", default=None)
            self._current_task_lease = current
        return current

    def _get_current_skill_ids_var(
        self,
    ) -> contextvars.ContextVar[tuple[str, ...] | None]:
        current = getattr(self, "_current_skill_ids", None)
        if current is None:
            current = contextvars.ContextVar("current_skill_ids", default=None)
            self._current_skill_ids = current
        return current

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return self._catalog_view().get_resource_briefs()

    def _get_resource_briefs(self) -> list[dict[str, Any]]:
        return self._resource_scope_view().get_resource_briefs()

    def _preview_tool_names(
        self,
        lease: ToolLease,
        limit: int = 6,
    ) -> tuple[str, ...]:
        return self._resource_scope_view().preview_tool_names(lease, limit=limit)

    def resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        return self._catalog_view().resolve_resource_scope(
            resource_id=resource_id,
            require_tools=require_tools,
        )

    def _resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        return self._resource_scope_view().resolve_resource_scope(
            resource_id,
            require_tools=require_tools,
        )

    def set_scheduled_task_manager(
        self, manager: "ScheduledTaskManager | None"
    ) -> None:
        self.scheduled_task_manager = manager

    def get_shared_gateway(self) -> "OpenAICompatibleGateway":
        return self._runtime_view().get_shared_gateway()

    def _get_shared_gateway(self) -> "OpenAICompatibleGateway":
        """Return a shared gateway instance, creating it lazily."""
        if self._shared_gateway is None:
            from .model_gateway import OpenAICompatibleGateway as _GW

            self._shared_gateway = _GW(self.config)
        return self._shared_gateway

    def register_tool(
        self,
        func: Any,
        group_name: str = "basic",
        preset_kwargs: dict[str, Any] | None = None,
        func_name: str | None = None,
    ) -> None:
        self._tool_loader_view().register_tool(
            func=func,
            group_name=group_name,
            preset_kwargs=preset_kwargs,
            func_name=func_name,
        )

    def _register_custom_tools(self, custom_tools: dict[str, dict]) -> None:
        self._tool_loader_view().register_custom_tools(custom_tools)

    def _discover_workspace_tools(self) -> None:
        self._tool_loader_view().discover_workspace_tools()

    def _ensure_workspace_on_pythonpath(self) -> None:
        self._tool_loader_view().ensure_workspace_on_pythonpath()

    def _load_tool_module(self, module_name: str) -> Any:
        return self._tool_loader_view().load_tool_module(module_name)

    def activate_groups(self, group_names: list[str]) -> str:
        for name in group_names:
            if name in self.groups:
                self.groups[name].active = True
        return self.get_tool_prompt()

    def deactivate_groups(self, group_names: list[str]) -> None:
        for name in group_names:
            if name in self.groups:
                self.groups[name].active = False

    def get_tool_prompt(self) -> str:
        notes = [g.notes for g in self.groups.values() if g.active and g.notes]
        return "\n\n".join(notes)

    def get_available_tools(self) -> list[dict]:
        active_groups = [name for name, g in self.groups.items() if g.active]
        lease = ToolLease(include_groups=tuple(active_groups))
        return list(self.registry.tool_schemas(lease))

    def search_resources(self, query: str | None = None) -> dict[str, Any]:
        return self._catalog_view().search_resources(query)

    def _search_resources(self, query: str | None = None) -> dict[str, Any]:
        return self._resource_scope_view().search_resources(query)

    def get_channel_prompts(self) -> str:
        prompts: list[str] = []
        for tools in self.channel_tools.values():
            prompt = tools.get_prompt()
            if prompt:
                prompts.append(prompt)
        return "\n\n".join(prompts)

    def register_channel_tools(self, channel_tools: "ChannelTools") -> None:
        group_name = channel_tools.get_tool_group_name()
        if group_name not in self.groups:
            self.groups[group_name] = ToolGroup(
                name=group_name,
                description=channel_tools.get_tool_group_description(),
                notes=channel_tools.get_prompt(),
                active=True,
            )
        for func in channel_tools.get_tools():
            self.register_tool(func, group_name=group_name)
        self.channel_tools[channel_tools.channel_name] = channel_tools

    def set_channel_context(self, ctx: "ChannelToolContext | None") -> None:
        from .channels.tools import ChannelToolContext

        ChannelToolContext.set_current(ctx)

    async def run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        return await self._runtime_view().run_subagent_task(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
        )

    async def _run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        return await self._subagent_runtime_view().run_subagent_task(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
        )

    def _build_task_lease(self, lease: dict[str, Any]) -> ToolLease:
        return self._resource_scope_view().build_task_lease(lease)

    def create_worker_tool(self) -> Any:
        async def create_worker(
            task_description: str,
            lease: dict[str, Any] | None = None,
            skill_ids: list[str] | None = None,
        ) -> str:
            inherited_lease = lease
            if inherited_lease is None:
                current_lease = self._get_current_task_lease_var().get()
                if current_lease is not None:
                    inherited_lease = self._lease_to_dict(current_lease)
            inherited_skill_ids = skill_ids
            if inherited_skill_ids is None:
                current_skill_ids = self._get_current_skill_ids_var().get()
                if current_skill_ids is not None:
                    inherited_skill_ids = list(current_skill_ids)
            text, _ = await self.run_subagent_task(
                task_description,
                lease=inherited_lease,
                skill_ids=inherited_skill_ids,
            )
            return text

        return create_worker

    def dispatch_workers_tool(self) -> Any:
        async def dispatch_workers(
            tasks: list[str],
            max_concurrency: int = 3,
            lease: dict[str, Any] | None = None,
            skill_ids: list[str] | None = None,
        ) -> str:
            normalized = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
            if not normalized:
                return "No valid tasks were provided."
            limit = max(1, min(int(max_concurrency), len(normalized), 8))
            semaphore = asyncio.Semaphore(limit)
            inherited_lease = lease
            if inherited_lease is None:
                current_lease = self._get_current_task_lease_var().get()
                if current_lease is not None:
                    inherited_lease = self._lease_to_dict(current_lease)
            inherited_skill_ids = skill_ids
            if inherited_skill_ids is None:
                current_skill_ids = self._get_current_skill_ids_var().get()
                if current_skill_ids is not None:
                    inherited_skill_ids = list(current_skill_ids)

            async def run_one(index: int, task: str) -> dict[str, Any]:
                async with semaphore:
                    try:
                        text, _ = await self.run_subagent_task(
                            task_description=task,
                            lease=inherited_lease,
                            agent_name=f"Worker-{index}",
                            skill_ids=inherited_skill_ids,
                        )
                        return {"index": index, "task": task, "result": text}
                    except Exception as exc:
                        return {"index": index, "task": task, "error": str(exc)}

            results = await asyncio.gather(
                *(run_one(i, task) for i, task in enumerate(normalized, start=1))
            )
            return json.dumps(
                {"max_concurrency": limit, "results": results},
                ensure_ascii=False,
                indent=2,
            )

        return dispatch_workers

    def _require_scheduled_task_manager(self) -> "ScheduledTaskManager":
        if self.scheduled_task_manager is None:
            raise RuntimeError("Scheduled task manager is unavailable in this runtime.")
        return self.scheduled_task_manager

    @staticmethod
    def _resolve_scheduled_task_target(
        channel: str | None,
        chat_id: str | None,
    ) -> tuple[str, str]:
        if channel and chat_id:
            return channel, chat_id

        from .channels.tools import ChannelToolContext

        ctx = ChannelToolContext.get_current()
        resolved_channel = (channel or "").strip()
        resolved_chat_id = (chat_id or "").strip()
        if ctx is not None:
            if not resolved_channel:
                resolved_channel = getattr(ctx, "channel_name", "") or str(
                    (ctx.metadata or {}).get("channel", "")
                )
            if not resolved_chat_id:
                resolved_chat_id = ctx.chat_id

        if not resolved_channel or not resolved_chat_id:
            raise RuntimeError(
                "Scheduled task target is missing. In channel conversations this "
                "should be inherited automatically; otherwise provide channel and chat_id."
            )
        return resolved_channel, resolved_chat_id

    @staticmethod
    def _anchor_delay_to_request_time(
        *,
        delay_seconds: float | None,
        run_at: str | None,
    ) -> tuple[float | None, str | None]:
        if delay_seconds is None or run_at is not None:
            return delay_seconds, run_at

        from .channels.tools import ChannelToolContext

        ctx = ChannelToolContext.get_current()
        if ctx is None:
            return delay_seconds, run_at
        raw_received_at = (ctx.metadata or {}).get("request_received_at")
        if not isinstance(raw_received_at, str) or not raw_received_at.strip():
            return delay_seconds, run_at
        received_at = raw_received_at.strip()
        if received_at.endswith("Z"):
            received_at = received_at[:-1] + "+00:00"
        base = datetime.datetime.fromisoformat(received_at)
        if base.tzinfo is None:
            base = base.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
        anchored = base.astimezone() + datetime.timedelta(seconds=float(delay_seconds))
        return None, anchored.isoformat(timespec="seconds")

    def list_scheduled_tasks_tool(self) -> Any:
        def list_scheduled_tasks() -> str:
            """List all scheduled tasks from the workspace task file."""
            return self._require_scheduled_task_manager().render_tasks()

        return list_scheduled_tasks

    def create_scheduled_task_tool(self) -> Any:
        def create_scheduled_task(
            prompt: str,
            channel: str | None = None,
            chat_id: str | None = None,
            name: str | None = None,
            cron: str | None = None,
            interval_seconds: float | None = None,
            run_at: str | None = None,
            delay_seconds: float | None = None,
            enabled: bool = True,
            require_active_runtime: bool = True,
        ) -> str:
            """Create a scheduled task.

            Scheduling options (provide exactly one):
            - delay_seconds: execute once after N seconds (e.g., 120 for 'in 2 minutes')
            - run_at: execute once at absolute time (ISO format or HH:MM)
            - cron: recurring cron expression
            - interval_seconds: recurring every N seconds
            """
            channel_name, target_chat_id = self._resolve_scheduled_task_target(
                channel, chat_id
            )
            delay_seconds, run_at = self._anchor_delay_to_request_time(
                delay_seconds=delay_seconds,
                run_at=run_at,
            )
            task = self._require_scheduled_task_manager().create_task(
                name=name,
                prompt=prompt,
                channel=channel_name,
                chat_id=target_chat_id,
                cron=cron,
                interval_seconds=interval_seconds,
                run_at=run_at,
                delay_seconds=delay_seconds,
                enabled=enabled,
                require_active_runtime=require_active_runtime,
            )
            return json.dumps(task, ensure_ascii=False, indent=2)

        return create_scheduled_task

    def save_scheduled_task_tool(self) -> Any:
        def save_scheduled_task(
            prompt: str,
            channel: str | None = None,
            chat_id: str | None = None,
            name: str | None = None,
            cron: str | None = None,
            interval_seconds: float | None = None,
            run_at: str | None = None,
            delay_seconds: float | None = None,
            enabled: bool = True,
            require_active_runtime: bool = True,
        ) -> str:
            """Create or update a scheduled task. Prefer this for natural-language task management.

            Scheduling options (provide exactly one):
            - delay_seconds: execute once after N seconds (e.g., 120 for 'in 2 minutes')
            - run_at: execute once at absolute time (ISO format or HH:MM)
            - cron: recurring cron expression
            - interval_seconds: recurring every N seconds
            """
            channel_name, target_chat_id = self._resolve_scheduled_task_target(
                channel, chat_id
            )
            delay_seconds, run_at = self._anchor_delay_to_request_time(
                delay_seconds=delay_seconds,
                run_at=run_at,
            )
            task = self._require_scheduled_task_manager().save_task(
                name=name,
                prompt=prompt,
                channel=channel_name,
                chat_id=target_chat_id,
                cron=cron,
                interval_seconds=interval_seconds,
                run_at=run_at,
                delay_seconds=delay_seconds,
                enabled=enabled,
                require_active_runtime=require_active_runtime,
            )
            return json.dumps(task, ensure_ascii=False, indent=2)

        return save_scheduled_task

    def update_scheduled_task_tool(self) -> Any:
        def update_scheduled_task(
            name: str,
            prompt: str | None = None,
            channel: str | None = None,
            chat_id: str | None = None,
            cron: str | None = None,
            interval_seconds: float | None = None,
            run_at: str | None = None,
            delay_seconds: float | None = None,
            enabled: bool | None = None,
            require_active_runtime: bool = True,
        ) -> str:
            """Update one scheduled task by name.

            Scheduling options (provide exactly one to switch schedule):
            - delay_seconds: execute once after N seconds (e.g., 120 for 'in 2 minutes')
            - run_at: execute once at absolute time (ISO format or HH:MM)
            - cron: recurring cron expression
            - interval_seconds: recurring every N seconds
            """
            if channel is None or chat_id is None:
                try:
                    channel, chat_id = self._resolve_scheduled_task_target(
                        channel, chat_id
                    )
                except RuntimeError:
                    pass
            delay_seconds, run_at = self._anchor_delay_to_request_time(
                delay_seconds=delay_seconds,
                run_at=run_at,
            )
            task = self._require_scheduled_task_manager().update_task(
                name=name,
                prompt=prompt,
                channel=channel,
                chat_id=chat_id,
                cron=cron,
                interval_seconds=interval_seconds,
                run_at=run_at,
                delay_seconds=delay_seconds,
                enabled=enabled,
                require_active_runtime=require_active_runtime,
            )
            return json.dumps(task, ensure_ascii=False, indent=2)

        return update_scheduled_task

    def delete_scheduled_task_tool(self) -> Any:
        def delete_scheduled_task(name: str) -> str:
            """Delete one scheduled task by name."""
            deleted = self._require_scheduled_task_manager().delete_task(name)
            return json.dumps(
                {"name": name, "deleted": deleted}, ensure_ascii=False, indent=2
            )

        return delete_scheduled_task

    def reset(self) -> None:
        close_clients_best_effort(self.mcp_clients)
        self.registry = ToolRegistry()
        self.groups.clear()
        self.channel_tools.clear()
        self.mcp_clients.clear()
        self.mcp_server_groups.clear()
        self.skills.clear()
        self._load_config()
        self._register_builtin_tools()

    async def _select_skill_packs(
        self,
        task_description: str,
        skill_ids: list[str] | None = None,
    ) -> list[SkillPack]:
        return await self._skill_runtime_view().select_skill_packs(
            task_description,
            skill_ids=skill_ids,
        )

    def _build_worker_sys_prompt(
        self,
        agent_name: str,
        task_description: str,
        tools_text: str,
        selected_skill_packs: list[SkillPack],
        merged_lease: "ToolLease | None" = None,
    ) -> str:
        return self._skill_runtime_view().build_worker_sys_prompt(
            agent_name=agent_name,
            task_description=task_description,
            tools_text=tools_text,
            selected_skill_packs=selected_skill_packs,
            merged_lease=merged_lease,
        )

    def _format_skill_catalog_for_lease(
        self, lease: "ToolLease", max_items: int = 20
    ) -> str:
        return self._skill_runtime_view().format_skill_catalog_for_lease(
            lease,
            max_items=max_items,
        )

    def _format_skill_catalog(self, max_items: int = 20) -> str:
        return self._skill_runtime_view().format_skill_catalog(max_items=max_items)

    async def __aenter__(self) -> "ResourceManager":
        await self.initialize_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.reset()

    def _resolve_workspace_file(self, file_path: str) -> tuple[str | None, str | None]:
        ws = self.config.workspace_dir.resolve()
        active_root = self._get_active_write_root()
        target = Path(file_path).expanduser()
        if not target.is_absolute():
            target = active_root / target
        target = target.resolve()
        if not self._is_within(ws, target):
            return None, (
                "PermissionError: Path is outside workspace. "
                f"workspace={ws}, requested={target}"
            )
        return str(target), None

    @staticmethod
    def _is_within(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def _get_active_write_root(self) -> Path:
        ws = self.config.workspace_dir.resolve()
        raw = Path(self._active_write_root.get()).expanduser().resolve()
        if self._is_within(ws, raw):
            return raw
        return ws

    def _get_output_dir(self) -> Path:
        output = self.config.workspace_dir.resolve() / "output"
        output.mkdir(parents=True, exist_ok=True)
        return output

    @staticmethod
    def _normalize_string_list(value: Any) -> tuple[str, ...]:
        return ExternalPythonRunner.normalize_string_list(value)

    @classmethod
    def _build_skill_runtime(cls, raw: dict[str, Any] | None = None) -> SkillRuntimeConfig:
        return ExternalPythonRunner.build_skill_runtime(raw)

    @staticmethod
    def _is_venv_python(path: str) -> bool:
        return ExternalPythonRunner.is_venv_python(path)

    def _get_python_probe_cache(
        self,
    ) -> dict[tuple[str, tuple[str, ...], str], str | None]:
        cache = getattr(self, "_python_probe_cache", None)
        if cache is None:
            cache = {}
            self._python_probe_cache = cache
        return cache

    def _discover_host_python_candidates(self) -> list[str]:
        return self._python_runner_view().discover_host_python_candidates()

    def _get_python_candidates(
        self,
        runtime: SkillRuntimeConfig | None = None,
    ) -> list[dict[str, Any]]:
        return self._python_runner_view().get_python_candidates(runtime)

    def _get_user_python(self) -> str:
        """Return a Python executable for user code, avoiding the project venv."""
        return self._python_runner_view().get_user_python()

    def _probe_python_candidate(self, candidate: dict[str, Any]) -> str | None:
        return ExternalPythonRunner.probe_python_candidate(
            candidate,
            cache=self._get_python_probe_cache(),
            get_active_write_root=self._get_active_write_root,
            clean_env=self._clean_env,
        )

    def _mark_python_candidate_unhealthy(
        self,
        candidate: dict[str, Any],
        detail: str,
    ) -> None:
        ExternalPythonRunner.mark_python_candidate_unhealthy(
            candidate,
            detail,
            cache=self._get_python_probe_cache(),
        )

    @staticmethod
    def _is_environment_failure(
        detail: str,
        *,
        returncode: int | None = None,
        payload_missing: bool = False,
    ) -> bool:
        return ExternalPythonRunner.is_environment_failure(
            detail,
            returncode=returncode,
            payload_missing=payload_missing,
        )

    def _clean_env(self) -> dict[str, str]:
        """Build an env dict with VIRTUAL_ENV removed and .venv/bin stripped from PATH."""
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        path = env.get("PATH", "")
        parts = [p for p in path.split(os.pathsep) if ".venv" not in p]
        env["PATH"] = os.pathsep.join(parts)
        return env

    async def _workspace_execute_python_code(
        self,
        code: str,
        timeout: float | int | str | None = 300,
        **kwargs: Any,
    ) -> str:
        return await self._workspace_tools_view().execute_python_code(
            code,
            timeout=timeout,
            **kwargs,
        )

    async def _workspace_execute_shell_command(
        self,
        command: str,
        timeout: float | int | str | None = 300,
        **kwargs: Any,
    ) -> str:
        return await self._workspace_tools_view().execute_shell_command(
            command,
            timeout=timeout,
            **kwargs,
        )

    async def _workspace_view_text_file(
        self,
        file_path: str,
        ranges: list[int] | None = None,
    ) -> str:
        return await self._workspace_tools_view().view_text_file(
            file_path,
            ranges=ranges,
        )

    async def _workspace_write_text_file(
        self,
        file_path: str,
        content: str,
        ranges: list[int] | None = None,
    ) -> str:
        return await self._workspace_tools_view().write_text_file(
            file_path,
            content,
            ranges=ranges,
        )

    async def _workspace_insert_text_file(
        self,
        file_path: str,
        content: str,
        line_number: int,
    ) -> str:
        return await self._workspace_tools_view().insert_text_file(
            file_path,
            content,
            line_number,
        )

    def _extract_media_from_text(self, text: str) -> list[str]:
        paths: list[str] = []
        write_root = self._get_active_write_root()
        for match in self._MEDIA_PATH_RE.finditer(text or ""):
            raw = match.group(1)
            path = Path(os.path.expanduser(raw))
            if not path.is_absolute():
                path = write_root / path
            if path.is_file():
                paths.append(str(path.resolve()))
        return sorted(set(paths))

    @staticmethod
    def _create_worker_executor(**kwargs: Any) -> Any:
        return create_worker_executor(**kwargs)

    @staticmethod
    def _coerce_timeout(
        timeout: float | int | str | None,
        default: float = 300.0,
    ) -> float:
        if timeout is None:
            return default
        if isinstance(timeout, str):
            timeout = timeout.strip()
            if not timeout:
                return default
        try:
            value = float(timeout)
        except (TypeError, ValueError):
            return default
        if value < 0:
            return default
        return value

    @staticmethod
    def _json_schema_for_callable(func: Any) -> dict[str, Any]:
        return ResourceToolLoader.json_schema_for_callable(func)

    @staticmethod
    def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
        return ResourceToolLoader.schema_for_annotation(annotation)
