"""Lightweight resource manager based on the in-repo kernel."""

from __future__ import annotations

import ast
import asyncio
import contextvars
from contextlib import contextmanager
import inspect
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

from .agent_kernel import (
    SkillPack,
    TaskResult,
    ToolLease,
    ToolRegistry,
    ToolResult,
)
from .builtin_tools import iter_builtin_tool_registrations
from .builtin_tools.scheduled_tasks import (
    build_create_scheduled_task_tool,
    build_delete_scheduled_task_tool,
    build_list_scheduled_tasks_tool,
    build_save_scheduled_task_tool,
    build_update_scheduled_task_tool,
)
from .builtin_tools.workers import (
    build_create_worker_tool,
    build_dispatch_workers_tool,
)
from .config import Config
from .mcp_runtime import (
    BaseMCPRuntimeClient,
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
from .resource_admin import ResourceAdminHelper
from .resource_mcp import ResourceMCPHelper
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
_CURRENT_CALLABLE_TOOL_WRITE_ROOT: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("current_callable_tool_write_root", default=None)
)
_CURRENT_DEFAULT_WRITE_ROOT: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("current_default_write_root", default=None)
)

if TYPE_CHECKING:
    from .channels.tools import ChannelTools, ChannelToolContext
    from .context import Tape, TapeStore
    from .cron import ScheduledTaskManager
    from .heartbeat import Heartbeat
    from .model_gateway import OpenAICompatibleGateway


from .resource_path_utils import (
    _collect_artifact_paths,
    _looks_like_path_candidate,
    _normalize_artifact_path_for_manager,
)


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
        collect_artifacts: bool = True,
    ):
        self._func = func
        self._name = name
        self._description = description
        self._schema = schema
        self._preset_kwargs = dict(preset_kwargs or {})
        self._resource_manager = resource_manager
        self._collect_artifacts_enabled = collect_artifacts

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
        tool_context_token: contextvars.Token[Any | None] | None = None
        write_root_token: contextvars.Token[str | None] | None = None
        try:
            kwargs = dict(self._preset_kwargs)
            kwargs.update(args or {})
            artifact_base = self._current_write_root()
            write_root = artifact_base
            state = getattr(context, "state", None)
            if isinstance(state, dict):
                state.setdefault("write_root", str(write_root))
            if self._resource_manager is not None:
                tool_context_token = (
                    self._resource_manager._get_current_tool_context_var().set(context)
                )
            with override_current_write_root(write_root):
                write_root_token = _CURRENT_CALLABLE_TOOL_WRITE_ROOT.set(
                    str(write_root)
                )
                if inspect.iscoroutinefunction(self._func):
                    value = await self._func(**kwargs)
                else:
                    # Run sync callables in a thread to avoid blocking the loop.
                    # Use contextvars.copy_context() so channel context etc.
                    # are visible inside the thread (Python <3.12 doesn't copy
                    # context automatically in run_in_executor).
                    loop = asyncio.get_running_loop()
                    ctx = contextvars.copy_context()
                    value = await loop.run_in_executor(
                        None,
                        ctx.run,
                        lambda: self._func(**kwargs),
                    )
            return ToolResult(
                ok=True,
                content=self._normalize_result(value),
                artifacts=(
                    self._collect_artifacts(value, base_dir=artifact_base)
                    if self._collect_artifacts_enabled
                    else []
                ),
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
        finally:
            if write_root_token is not None:
                _CURRENT_CALLABLE_TOOL_WRITE_ROOT.reset(write_root_token)
            if tool_context_token is not None and self._resource_manager is not None:
                self._resource_manager._get_current_tool_context_var().reset(
                    tool_context_token
                )

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
            return get_current_write_root()
        return self._resource_manager._get_active_write_root()

    @staticmethod
    def _looks_like_path_candidate(candidate: str) -> bool:
        return _looks_like_path_candidate(candidate)

    def _collect_artifacts(
        self,
        value: Any,
        *,
        base_dir: Path | None = None,
    ) -> list[str]:
        return _collect_artifact_paths(
            value,
            base_dir=(base_dir or self._current_write_root()).resolve(),
            normalize_path=self._normalize_artifact_path,
        )

    def _normalize_artifact_path(self, resolved: Path) -> Path:
        return _normalize_artifact_path_for_manager(self._resource_manager, resolved)


@contextmanager
def override_current_write_root(path: str | os.PathLike[str]) -> Any:
    resolved = Path(path).expanduser().resolve()
    token = _CURRENT_DEFAULT_WRITE_ROOT.set(str(resolved))
    try:
        yield resolved
    finally:
        _CURRENT_DEFAULT_WRITE_ROOT.reset(token)


def get_current_write_root() -> Path:
    raw = _CURRENT_CALLABLE_TOOL_WRITE_ROOT.get()
    if raw:
        return Path(raw).expanduser().resolve()
    fallback = _CURRENT_DEFAULT_WRITE_ROOT.get()
    if fallback:
        return Path(fallback).expanduser().resolve()
    return Path.cwd().resolve()


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

    def recommend_resources(
        self,
        query: str,
        *,
        limit: int = 6,
    ) -> dict[str, Any]:
        return self._manager._recommend_resources(query, limit=limit)


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
        memory_store: Any = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> tuple[str, list[str]]:
        return await self._manager._run_subagent_task(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )

    async def run_subagent_task_result(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        memory_store: Any = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> TaskResult:
        return await self._manager._run_subagent_task_result(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )


class ResourceManager:
    """Centralized resource manager without external agent frameworks."""

    _orchestration_tools = {"create_worker", "dispatch_workers"}
    _EXTERNAL_OUTPUT_PATH_ARG_NAMES = frozenset(
        {"output_path", "output_file", "save_path", "download_path"}
    )
    _MEDIA_PATH_RE = re.compile(
        r"(?:^|[\s'\"`(:：=])((?:/~|[~/])?[-\w./]+(?:\.(?:png|jpg|jpeg|gif|bmp|webp|pdf|txt|md|json|yaml|yml|csv|xlsx|pptx|docx|mp4|mp3|wav)))"
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
        self._current_tool_context: contextvars.ContextVar[Any | None] = (
            contextvars.ContextVar(
                "current_tool_context",
                default=None,
            )
        )
        self._current_worker_depth: contextvars.ContextVar[int] = (
            contextvars.ContextVar("current_worker_depth", default=0)
        )
        self._python_probe_cache: dict[
            tuple[str, tuple[str, ...], str], str | None
        ] = {}
        self._observability_provider: Any = None
        self._skill_load_errors: list[dict[str, str]] = []
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
        return self._lazy("catalog", lambda: ResourceCatalog(self))

    def _runtime_view(self) -> WorkerRuntime:
        return self._lazy("runtime", lambda: WorkerRuntime(self))

    def _lazy(self, attr: str, factory: Any) -> Any:
        """Return a lazily-initialised instance attribute, creating it on first access."""
        obj = getattr(self, attr, None)
        if obj is None:
            obj = factory()
            setattr(self, attr, obj)
        return obj

    def _python_runner_view(self) -> ExternalPythonRunner:
        return self._lazy("_python_runner", lambda: ExternalPythonRunner(self.config))

    def _workspace_tools_view(self) -> WorkspaceToolSuite:
        return self._lazy("_workspace_tools", lambda: WorkspaceToolSuite(self))

    def _skill_loader_view(self) -> SkillLoader:
        return self._lazy("_skill_loader", lambda: SkillLoader(self))

    def _resource_scope_view(self) -> ResourceScopeHelper:
        return self._lazy("_resource_scope", lambda: ResourceScopeHelper(self))

    def _tool_loader_view(self) -> ResourceToolLoader:
        return self._lazy("_tool_loader", lambda: ResourceToolLoader(self))

    def _subagent_runtime_view(self) -> ResourceSubagentRuntime:
        return self._lazy("_subagent_runtime", lambda: ResourceSubagentRuntime(self))

    def _skill_runtime_view(self) -> ResourceSkillRuntime:
        return self._lazy("_skill_runtime", lambda: ResourceSkillRuntime(self))

    def _admin_view(self) -> ResourceAdminHelper:
        return self._lazy("_resource_admin", lambda: ResourceAdminHelper(self))

    def _mcp_view(self) -> ResourceMCPHelper:
        return self._lazy("_resource_mcp", lambda: ResourceMCPHelper(self))

    @staticmethod
    def _callable_tool_cls() -> type[CallableTool]:
        return CallableTool

    async def _register_mcp_servers(self, mcp_servers: dict[str, dict]) -> None:
        await self._mcp_view().register_mcp_servers(mcp_servers)

    async def register_mcp(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        transport: str = "streamable_http",
        group_name: str = "mcp",
    ) -> None:
        await self._mcp_view().register_mcp(
            name=name,
            url=url,
            headers=headers,
            transport=transport,
            group_name=group_name,
        )

    async def register_mcp_stdio(
        self,
        name: str,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        group_name: str = "mcp",
    ) -> None:
        await self._mcp_view().register_mcp_stdio(
            name=name,
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            group_name=group_name,
        )

    def _prepare_mcp_stdio_launch(
        self,
        name: str,
        command: str,
        args: list[str],
        server_conf: dict[str, Any],
    ) -> tuple[str, list[str], str | None, dict[str, str] | None]:
        return self._mcp_view().prepare_mcp_stdio_launch(
            name=name,
            command=command,
            args=args,
            server_conf=server_conf,
        )

    def _normalize_stdio_mcp_env(
        self,
        raw_env: Any,
        *,
        defaults: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return self._mcp_view().normalize_stdio_mcp_env(
            raw_env,
            defaults=defaults,
        )

    def _prepare_mcp_http_launch(
        self,
        name: str,
        url: str,
        server_conf: dict[str, Any],
    ) -> tuple[str, dict[str, str] | None]:
        return self._mcp_view().prepare_mcp_http_launch(
            name=name,
            url=url,
            server_conf=server_conf,
        )

    @staticmethod
    def _normalize_stdio_mcp_path(value: Any) -> str | None:
        return ResourceMCPHelper.normalize_stdio_mcp_path(value)

    def _normalize_http_mcp_headers(
        self,
        raw_headers: Any,
        *,
        defaults: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return self._mcp_view().normalize_http_mcp_headers(
            raw_headers,
            defaults=defaults,
        )

    def _build_mcp_stdio_env(self, name: str) -> dict[str, str]:
        return self._mcp_view().build_mcp_stdio_env(name)

    def _build_mcp_http_headers(self, name: str) -> dict[str, str]:
        return self._mcp_view().build_mcp_http_headers(name)

    def _build_mcp_runtime_metadata(self, name: str) -> dict[str, str]:
        return self._mcp_view().build_mcp_runtime_metadata(name)

    def _get_mcp_artifact_root(self, name: str) -> Path:
        return self._mcp_view().get_mcp_artifact_root(name)

    @staticmethod
    def _normalize_mcp_mapping_value(key: str, value: Any) -> str:
        return ResourceMCPHelper.normalize_mcp_mapping_value(key, value)

    @staticmethod
    def _is_path_like_mcp_key(key: str) -> bool:
        return ResourceMCPHelper.is_path_like_mcp_key(key)

    def _load_config(self) -> None:
        tool_groups = {
            k: v
            for k, v in self.config.get_tool_groups().items()
            if not k.startswith("_")
        }
        self._setup_tool_groups(tool_groups)
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

    def _resolve_skill_record(self, skill_name: str) -> LoadedSkill | None:
        return self._admin_view().resolve_skill_record(skill_name)

    def _resolve_skill_directory_input(self, skill_path: str) -> Path:
        return self._admin_view().resolve_skill_directory_input(skill_path)

    def _record_runtime_hint(self, message: str) -> None:
        self._admin_view().record_runtime_hint(message)

    def reload_skill(self, skill_path: str) -> str:
        return self._admin_view().reload_skill(skill_path)

    def get_assistant_profile(self) -> str:
        return self._admin_view().get_assistant_profile()

    def set_assistant_profile(self, content: str, mode: str = "replace") -> str:
        return self._admin_view().set_assistant_profile(content, mode=mode)

    def list_admin_skills(
        self,
        query: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        return self._admin_view().list_admin_skills(
            query=query,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    def enable_skill(self, skill_name: str) -> str:
        return self._admin_view().enable_skill(skill_name)

    def disable_skill(self, skill_name: str) -> str:
        return self._admin_view().disable_skill(skill_name)

    def delete_skill(self, skill_name: str) -> str:
        return self._admin_view().delete_skill(skill_name)

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
    def _read_skill_document(cls, skill_dir: Path) -> tuple[dict[str, str], str, str]:
        return SkillLoader.read_skill_document(skill_dir)

    def _setup_tool_groups(self, user_groups: dict[str, dict]) -> None:
        defaults = {
            "basic": ToolGroup(
                "basic",
                "核心工具：定时/延时发送消息、任务调度管理（创建/修改/删除/列出定时任务）",
                active=True,
            ),
            "worker_control": ToolGroup(
                "worker_control",
                "内部编排工具：创建或分发子 worker。",
                active=False,
                notes="Only enable when a workflow explicitly needs nested workers.",
            ),
            "code": ToolGroup(
                "code",
                "Code execution and file operations",
                active=True,
                notes="Use these tools for coding and filesystem tasks.",
            ),
            "admin": ToolGroup(
                "admin",
                "Admin tools for assistant profile and skill state management",
                active=False,
                notes="Enable only when explicitly editing BabyBot profile or skill state.",
            ),
            "search": ToolGroup("search", "Search tools", active=False),
            "analysis": ToolGroup("analysis", "Analysis tools", active=False),
            "browser": ToolGroup("browser", "Browser automation tools", active=False),
            "web": ToolGroup(
                "web",
                "Web tools: fetch URLs and search the internet",
                active=True,
                notes="Requires tavily_api_key in config for web_search.",
            ),
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
        for func, group_name in iter_builtin_tool_registrations(self):
            self.register_tool(
                func,
                group_name=group_name,
                collect_artifacts=not func.__name__.startswith("_workspace_"),
            )

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

    def _get_current_worker_depth_var(self) -> contextvars.ContextVar[int]:
        current = getattr(self, "_current_worker_depth", None)
        if current is None:
            current = contextvars.ContextVar("current_worker_depth", default=0)
            self._current_worker_depth = current
        return current

    def _get_current_tool_context_var(self) -> contextvars.ContextVar[Any | None]:
        current = getattr(self, "_current_tool_context", None)
        if current is None:
            current = contextvars.ContextVar("current_tool_context", default=None)
            self._current_tool_context = current
        return current

    def _get_current_task_heartbeat(self) -> Any:
        ctx = self._get_current_tool_context_var().get()
        state = getattr(ctx, "state", None)
        if isinstance(state, dict):
            return state.get("heartbeat")
        return None

    def _report_external_process_output(
        self, line: str, *, stream: str = "stdout"
    ) -> None:
        heartbeat = self._get_current_task_heartbeat()
        if heartbeat is None:
            return
        status, progress = ExternalPythonRunner.parse_progress_line(line)
        payload: dict[str, Any] = {}
        if progress is not None:
            payload["progress"] = progress
        if status:
            payload["status"] = status
        elif stream:
            payload["status"] = f"running:{stream}"
        heartbeat.beat(**payload)

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

    def recommend_resources(
        self,
        query: str,
        *,
        limit: int = 6,
    ) -> dict[str, Any]:
        return self._catalog_view().recommend_resources(query, limit=limit)

    def _recommend_resources(
        self,
        query: str,
        *,
        limit: int = 6,
    ) -> dict[str, Any]:
        return self._resource_scope_view().recommend_resources(query, limit=limit)

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
        collect_artifacts: bool = True,
    ) -> None:
        self._tool_loader_view().register_tool(
            func=func,
            group_name=group_name,
            preset_kwargs=preset_kwargs,
            func_name=func_name,
            collect_artifacts=collect_artifacts,
        )

    def set_observability_provider(self, provider: Any) -> None:
        self._observability_provider = provider

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

    def _default_chat_key(self) -> str:
        return self._admin_view().default_chat_key()

    def _inspect_runtime_flow(self, flow_id: str = "", chat_key: str = "") -> str:
        return self._admin_view().inspect_runtime_flow(
            flow_id=flow_id,
            chat_key=chat_key,
        )

    def _inspect_chat_context(self, chat_key: str = "", query: str = "") -> str:
        return self._admin_view().inspect_chat_context(
            chat_key=chat_key,
            query=query,
        )

    def _fallback_inspect_chat_context(self, chat_key: str, query: str = "") -> str:
        return self._admin_view().fallback_inspect_chat_context(
            chat_key,
            query=query,
        )

    def _inspect_policy(self, chat_key: str = "", decision_kind: str = "") -> str:
        return self._admin_view().inspect_policy(
            chat_key=chat_key,
            decision_kind=decision_kind,
        )

    @staticmethod
    def _schema_type_summary(schema: dict[str, Any]) -> str:
        return ResourceAdminHelper.schema_type_summary(schema)

    def _inspect_tools(
        self,
        query: str = "",
        group: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        return self._admin_view().inspect_tools(
            query=query,
            group=group,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    def _inspect_skills(
        self,
        query: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        return self._admin_view().inspect_skills(
            query=query,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    def _inspect_skill_load_errors(self, limit: int = 20) -> str:
        return self._admin_view().inspect_skill_load_errors(limit=limit)

    def _record_skill_load_error(
        self,
        *,
        skill: str,
        path: str,
        error: str,
        stage: str,
    ) -> None:
        self._admin_view().record_skill_load_error(
            skill=skill,
            path=path,
            error=error,
            stage=stage,
        )

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
        memory_store: Any = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> tuple[str, list[str]]:
        return await self._runtime_view().run_subagent_task(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )

    async def run_subagent_task_result(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        memory_store: Any = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> TaskResult:
        return await self._runtime_view().run_subagent_task_result(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )

    async def _run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        memory_store: Any = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> tuple[str, list[str]]:
        return await self._subagent_runtime_view().run_subagent_task(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )

    async def _run_subagent_task_result(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: "Tape | None" = None,
        tape_store: "TapeStore | None" = None,
        memory_store: Any = None,
        heartbeat: "Heartbeat | None" = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> TaskResult:
        return await self._subagent_runtime_view().run_subagent_task_result(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )

    def _build_task_lease(self, lease: dict[str, Any]) -> ToolLease:
        return self._resource_scope_view().build_task_lease(lease)

    def create_worker_tool(self) -> Any:
        return build_create_worker_tool(self)

    def dispatch_workers_tool(self) -> Any:
        return build_dispatch_workers_tool(self)

    def list_scheduled_tasks_tool(self) -> Any:
        return build_list_scheduled_tasks_tool(self)

    def create_scheduled_task_tool(self) -> Any:
        return build_create_scheduled_task_tool(self)

    def save_scheduled_task_tool(self) -> Any:
        return build_save_scheduled_task_tool(self)

    def update_scheduled_task_tool(self) -> Any:
        return build_update_scheduled_task_tool(self)

    def delete_scheduled_task_tool(self) -> Any:
        return build_delete_scheduled_task_tool(self)

    async def areset(self) -> None:
        for client in self.mcp_clients.values():
            try:
                await client.close()
            except Exception as exc:
                logger.warning("Failed to close MCP client: %s", exc)
        self.registry = ToolRegistry()
        self.groups.clear()
        self.channel_tools.clear()
        self.mcp_clients.clear()
        self.mcp_server_groups.clear()
        self.skills.clear()
        self._load_config()
        self._register_builtin_tools()

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

    @contextmanager
    def _override_current_write_root(self, write_root: Path) -> Any:
        resolved = Path(write_root).expanduser().resolve()
        token = self._active_write_root.set(str(resolved))
        try:
            with override_current_write_root(resolved):
                yield resolved
        finally:
            self._active_write_root.reset(token)

    def _get_output_dir(self) -> Path:
        output = self.config.workspace_dir.resolve() / "output"
        output.mkdir(parents=True, exist_ok=True)
        return output

    @staticmethod
    def _normalize_string_list(value: Any) -> tuple[str, ...]:
        return ExternalPythonRunner.normalize_string_list(value)

    @classmethod
    def _build_skill_runtime(
        cls, raw: dict[str, Any] | None = None
    ) -> SkillRuntimeConfig:
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
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        return await self._workspace_tools_view().view_text_file(
            file_path,
            ranges=ranges,
            offset=offset,
            limit=limit,
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

    async def _workspace_edit_text_file(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        return await self._workspace_tools_view().edit_text_file(
            file_path,
            old_text,
            new_text,
            replace_all=replace_all,
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
        return _collect_artifact_paths(
            text or "",
            base_dir=self._get_active_write_root(),
            normalize_path=self._normalize_artifact_path,
            media_path_re=self._MEDIA_PATH_RE,
        )

    def _normalize_artifact_path(self, resolved: Path) -> Path:
        return _normalize_artifact_path_for_manager(self, resolved)

    def _normalize_external_tool_arguments(
        self,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(arguments or {})
        for key, value in list(normalized.items()):
            if not isinstance(value, str):
                continue
            arg_name = str(key or "").strip().lower()
            if arg_name not in self._EXTERNAL_OUTPUT_PATH_ARG_NAMES:
                continue
            candidate = value.strip()
            if not candidate:
                continue
            resolved, error = self._resolve_workspace_file(candidate)
            if resolved is None or error:
                continue
            normalized[key] = resolved
        return normalized

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
