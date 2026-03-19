"""Lightweight resource manager based on the in-repo kernel."""

from __future__ import annotations

import asyncio
import ast
import contextlib
import contextvars
import datetime
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field

from .agent_kernel import (
    ExecutionContext,
    SkillPack,
    register_mcp_tools,
    TaskContract,
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
from .worker import create_worker_executor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .channels.tools import ChannelTools, ChannelToolContext
    from .context import Tape, TapeStore
    from .cron import ScheduledTaskManager
    from .heartbeat import Heartbeat
    from .model_gateway import OpenAICompatibleGateway


_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+(-[^\s]*\s+)*-[^\s]*[rR]", "recursive delete"),
    (r"rm\s+-rf\s+/", "recursive delete from root"),
    (r"mkfs", "filesystem format"),
    (r"dd\s+if=", "disk overwrite"),
    (r">\s*/dev/sd", "device overwrite"),
    (r":()\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"chmod\s+-R\s+777\s+/", "recursive permission change on root"),
    (r"curl[^|]*\|\s*(sudo\s+)?bash", "pipe to shell"),
    (r"wget[^|]*\|\s*(sudo\s+)?bash", "pipe to shell"),
]


def _check_shell_safety(command: str) -> str | None:
    """Return an error string if *command* matches a dangerous pattern, else None."""
    for pattern, label in _DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Blocked: command matches dangerous pattern ({label})"
    return None


@dataclass
class ToolGroup:
    name: str
    description: str
    notes: str = ""
    active: bool = False


@dataclass
class ResourceBrief:
    """One concise resource record exposed to the main routing agent."""

    resource_id: str
    resource_type: str
    name: str
    purpose: str
    group: str = ""
    tool_count: int = 0
    tools_preview: tuple[str, ...] = ()
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.resource_id,
            "type": self.resource_type,
            "name": self.name,
            "purpose": self.purpose,
            "group": self.group,
            "tool_count": self.tool_count,
            "tools_preview": list(self.tools_preview),
            "active": self.active,
        }


@dataclass
class LoadedSkill:
    """Resolved skill metadata for runtime routing."""

    name: str
    description: str
    directory: str
    prompt: str = ""
    keywords: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    lease: ToolLease = ToolLease()
    source: str = "auto"
    active: bool = True
    tool_group: str = ""
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ScriptFunctionSpec:
    name: str
    description: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class _CliArgumentSpec:
    name: str
    flag: str
    schema: dict[str, Any]
    required: bool = False
    action: str | None = None


@dataclass(frozen=True)
class _CliToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    arguments: tuple[_CliArgumentSpec, ...]


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

                value = await loop.run_in_executor(
                    None, ctx.run, _run_in_workspace
                )
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
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
            ".pdf", ".txt", ".md", ".json", ".yaml", ".yml",
            ".csv", ".xlsx", ".pptx", ".docx", ".mp4", ".mp3", ".wav",
        }:
            return True
        return (
            "/" in text
            or "\\" in text
            or text.startswith(".")
            or text.startswith("~")
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
        self._current_task_lease: contextvars.ContextVar[ToolLease | None] = contextvars.ContextVar(
            "current_task_lease",
            default=None,
        )
        self._current_skill_ids: contextvars.ContextVar[tuple[str, ...] | None] = contextvars.ContextVar(
            "current_skill_ids",
            default=None,
        )
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
        for name, conf in configured.items():
            try:
                directory = conf.get("directory", "")
                if not directory:
                    continue
                skill_dir = Path(self.config.resolve_workspace_path(directory))
                if not skill_dir.exists() or not skill_dir.is_dir():
                    logger.warning("Configured skill not found: %s", skill_dir)
                    continue
                meta, prompt = self._read_skill_document(skill_dir)
                resolved_name = meta.get("name", name)
                tool_group, tool_names = self._register_skill_tools(
                    skill_name=resolved_name,
                    skill_dir=skill_dir,
                )
                description = conf.get("description") or meta.get("description", "")
                keywords = self._normalize_keywords(
                    conf.get("keywords"),
                    fallback=(description, name),
                )
                phrases = self._normalize_phrases(
                    conf.get("keywords"),
                    fallback=(description, name),
                )
                self._upsert_skill(
                    LoadedSkill(
                        name=meta.get("name", name),
                        description=description or f"Skill: {name}",
                        directory=str(skill_dir.resolve()),
                        prompt=prompt,
                        keywords=keywords,
                        phrases=phrases,
                        lease=ToolLease(
                            include_groups=tuple(
                                set(conf.get("include_groups") or ()) | ({tool_group} if tool_group else set())
                            ),
                            include_tools=tuple(conf.get("include_tools") or ()),
                            exclude_tools=tuple(conf.get("exclude_tools") or ()),
                        ),
                        source="config",
                        active=bool(conf.get("active", True)),
                        tool_group=tool_group,
                        tools=tool_names,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to load configured skill %s: %s", name, exc)

    def _discover_skills(self) -> None:
        roots = [self.config.builtin_skills_dir, self.config.workspace_skills_dir]
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if not (child / "SKILL.md").exists():
                    continue
                try:
                    meta, prompt = self._read_skill_document(child)
                    name = meta.get("name", child.name)
                    tool_group, tool_names = self._register_skill_tools(
                        skill_name=name,
                        skill_dir=child,
                    )
                    key = name.strip().lower()
                    if key in self.skills:
                        continue
                    description = meta.get("description", f"Skill: {name}")
                    keywords = self._normalize_keywords(
                        None,
                        fallback=(description, name, prompt[:400]),
                    )
                    phrases = self._normalize_phrases(
                        None,
                        fallback=(description, name),
                    )
                    self._upsert_skill(
                        LoadedSkill(
                            name=name,
                            description=description,
                            directory=str(child.resolve()),
                            prompt=prompt,
                            keywords=keywords,
                            phrases=phrases,
                            source="auto",
                            active=True,
                            lease=ToolLease(
                                include_groups=(tool_group,) if tool_group else (),
                            ),
                            tool_group=tool_group,
                            tools=tool_names,
                        )
                    )
                except Exception as exc:
                    logger.warning("Failed to auto-load skill %s: %s", child, exc)

    def _upsert_skill(self, skill: LoadedSkill) -> None:
        self.skills[skill.name.strip().lower()] = skill

    def _register_skill_tools(
        self,
        skill_name: str,
        skill_dir: Path,
    ) -> tuple[str, tuple[str, ...]]:
        """Register callable tools from `<skill_dir>/scripts/*.py`.

        Returns `(group_name, tool_names)`.
        """
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.exists() or not scripts_dir.is_dir():
            return "", ()
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", skill_name.strip().lower()).strip("_")
        if not slug:
            slug = "skill"
        group_name = f"skill_{slug}"
        if group_name not in self.groups:
            self.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"Tools from skill {skill_name}",
                notes=f"Skill tools for {skill_name}",
                active=False,
            )
        tool_names: list[str] = []
        for py_file in sorted(scripts_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            before_count = len(tool_names)
            try:
                module = self._load_tool_module(str(py_file))
                for func_name, func in inspect.getmembers(module, inspect.isfunction):
                    if (
                        func.__module__ != module.__name__
                        or self._skip_skill_function_name(func_name)
                    ):
                        continue
                    tool_name = f"{slug}__{func_name}"
                    self.register_tool(
                        func=func,
                        group_name=group_name,
                        func_name=tool_name,
                    )
                    tool_names.append(tool_name)
            except Exception as exc:
                logger.warning(
                    "Failed to import skill script %s: %s; using host-python proxy registration.",
                    py_file,
                    exc,
                )
                specs = self._extract_function_specs_from_script(py_file)
                if not specs:
                    continue
                for spec in specs:
                    tool_name = f"{slug}__{spec.name}"
                    proxy = self._build_external_skill_callable(
                        script_path=py_file,
                        function_name=spec.name,
                    )
                    self.registry.register(
                        CallableTool(
                            func=proxy,
                            name=tool_name,
                            description=spec.description,
                            schema=spec.schema,
                            resource_manager=self,
                        ),
                        group=group_name,
                    )
                    tool_names.append(tool_name)
            if len(tool_names) == before_count:
                cli_spec = self._extract_cli_tool_spec_from_script(py_file)
                if cli_spec is not None:
                    tool_name = f"{slug}__{cli_spec.name}"
                    self.registry.register(
                        CallableTool(
                            func=self._build_external_cli_script_callable(py_file, cli_spec),
                            name=tool_name,
                            description=cli_spec.description,
                            schema=cli_spec.schema,
                            resource_manager=self,
                        ),
                        group=group_name,
                    )
                    tool_names.append(tool_name)
        return group_name, tuple(sorted(set(tool_names)))

    @staticmethod
    def _skip_skill_function_name(name: str) -> bool:
        return name.startswith("_") or name in {"main", "parse_arguments", "create_client"}

    @classmethod
    def _extract_function_specs_from_script(
        cls,
        script_path: Path,
    ) -> list[_ScriptFunctionSpec]:
        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except Exception as exc:
            logger.warning("Failed to parse skill script %s: %s", script_path, exc)
            return []

        specs: list[_ScriptFunctionSpec] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if cls._skip_skill_function_name(node.name):
                continue
            schema = cls._schema_from_ast_function(node)
            doc = ast.get_docstring(node) or node.name
            description = doc.splitlines()[0].strip() if doc else node.name
            specs.append(
                _ScriptFunctionSpec(
                    name=node.name,
                    description=description or node.name,
                    schema=schema,
                )
            )
        return specs

    @classmethod
    def _extract_cli_tool_spec_from_script(
        cls,
        script_path: Path,
    ) -> _CliToolSpec | None:
        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except Exception as exc:
            logger.warning("Failed to parse CLI skill script %s: %s", script_path, exc)
            return None

        parse_func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        has_main = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "parse_arguments":
                    parse_func = node
                elif node.name == "main":
                    has_main = True
        if parse_func is None or not has_main:
            return None

        parser_names: set[str] = set()
        args: list[_CliArgumentSpec] = []
        required: list[str] = []
        properties: dict[str, Any] = {}

        for node in ast.walk(parse_func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                func = call.func
                if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            parser_names.add(target.id)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "add_argument"
                and isinstance(func.value, ast.Name)
                and func.value.id in parser_names
            ):
                continue
            option_strings = [
                arg.value for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if not option_strings:
                continue
            flag = next((item for item in option_strings if item.startswith("--")), option_strings[-1])
            name = flag.lstrip("-").replace("-", "_")
            schema, is_required, action = cls._schema_from_argparse_call(node)
            properties[name] = schema
            args.append(
                _CliArgumentSpec(
                    name=name,
                    flag=flag,
                    schema=schema,
                    required=is_required,
                    action=action,
                )
            )
            if is_required:
                required.append(name)

        if not args:
            return None

        return _CliToolSpec(
            name=script_path.stem,
            description=f"Run CLI script {script_path.stem}",
            schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            arguments=tuple(args),
        )

    @classmethod
    def _schema_from_argparse_call(
        cls,
        call: ast.Call,
    ) -> tuple[dict[str, Any], bool, str | None]:
        action: str | None = None
        arg_type: Any = None
        required = False
        for keyword in call.keywords:
            if keyword.arg == "action" and isinstance(keyword.value, ast.Constant):
                action = str(keyword.value.value)
            elif keyword.arg == "type":
                arg_type = cls._annotation_name_from_ast(keyword.value)
            elif keyword.arg == "required" and isinstance(keyword.value, ast.Constant):
                required = bool(keyword.value.value)
        if action == "store_true":
            return {"type": "boolean"}, False, action
        if arg_type == "int":
            schema = {"type": "integer"}
        elif arg_type == "float":
            schema = {"type": "number"}
        elif arg_type == "bool":
            schema = {"type": "boolean"}
        else:
            schema = {"type": "string"}
        return schema, required, action

    @staticmethod
    def _annotation_name_from_ast(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _build_external_cli_script_callable(
        self,
        script_path: Path,
        cli_spec: _CliToolSpec,
    ) -> Any:
        resolved = str(script_path.resolve())
        arguments = cli_spec.arguments

        async def _runner(**kwargs: Any) -> str:
            argv = [self._get_user_python(), resolved]
            for spec in arguments:
                if spec.name not in kwargs:
                    continue
                value = kwargs.get(spec.name)
                if spec.action == "store_true":
                    if value:
                        argv.append(spec.flag)
                    continue
                if value is None:
                    continue
                argv.extend([spec.flag, self._format_cli_argument(value)])
            timeout_s = self._coerce_timeout(kwargs.get("timeout"), default=300.0)
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self._get_active_write_root()),
                env=self._clean_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.communicate()
                raise RuntimeError(f"CLI tool timeout after {timeout_s}s")
            out_text = (stdout or b"").decode("utf-8", errors="ignore").strip()
            err_text = (stderr or b"").decode("utf-8", errors="ignore").strip()
            if proc.returncode != 0:
                detail = err_text or out_text or f"exit code {proc.returncode}"
                raise RuntimeError(detail)
            return out_text or err_text

        return _runner

    @staticmethod
    def _format_cli_argument(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @classmethod
    def _schema_from_ast_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        defaults = list(node.args.defaults or [])
        plain_args = list(node.args.args or [])
        default_offset = len(plain_args) - len(defaults)
        for idx, arg in enumerate(plain_args):
            name = arg.arg
            if name in {"self", "context"}:
                continue
            properties[name] = cls._schema_for_ast_annotation(arg.annotation)
            if idx < default_offset:
                required.append(name)
        for kw_arg, kw_default in zip(node.args.kwonlyargs or [], node.args.kw_defaults or []):
            name = kw_arg.arg
            if name in {"self", "context"}:
                continue
            properties[name] = cls._schema_for_ast_annotation(kw_arg.annotation)
            if kw_default is None:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def _schema_for_ast_annotation(annotation: ast.AST | None) -> dict[str, Any]:
        if annotation is None:
            return {"type": "string"}
        text = ast.unparse(annotation).strip().lower()
        if "bool" in text:
            return {"type": "boolean"}
        if any(token in text for token in {"int", "long"}):
            return {"type": "integer"}
        if any(token in text for token in {"float", "decimal"}):
            return {"type": "number"}
        if any(token in text for token in {"dict", "mapping"}):
            return {"type": "object"}
        if any(token in text for token in {"list", "tuple", "set", "sequence"}):
            return {"type": "array"}
        return {"type": "string"}

    def _build_external_skill_callable(
        self,
        script_path: Path,
        function_name: str,
    ) -> Any:
        resolved = str(script_path.resolve())

        async def _runner(**kwargs: Any) -> str:
            return await self._invoke_external_skill_function(
                script_path=resolved,
                function_name=function_name,
                arguments=kwargs,
            )

        return _runner

    async def _invoke_external_skill_function(
        self,
        script_path: str,
        function_name: str,
        arguments: dict[str, Any],
    ) -> str:
        runner = (
            "import asyncio, importlib.util, inspect, json, sys, traceback\n"
            "MARK='__BABYBOT_RESULT__'\n"
            "script, fn, raw = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "try:\n"
            "    spec = importlib.util.spec_from_file_location('babybot_skill_proxy', script)\n"
            "    if spec is None or spec.loader is None:\n"
            "        raise RuntimeError(f'cannot load script: {script}')\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(mod)\n"
            "    if not hasattr(mod, fn):\n"
            "        raise AttributeError(f'function not found: {fn}')\n"
            "    func = getattr(mod, fn)\n"
            "    kwargs = json.loads(raw) if raw else {}\n"
            "    if inspect.iscoroutinefunction(func):\n"
            "        out = asyncio.run(func(**kwargs))\n"
            "    else:\n"
            "        out = func(**kwargs)\n"
            "    print(MARK + json.dumps({'ok': True, 'result': out}, ensure_ascii=False, default=str))\n"
            "except Exception as exc:\n"
            "    traceback.print_exc()\n"
            "    print(MARK + json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False))\n"
        )
        args_json = json.dumps(arguments or {}, ensure_ascii=False)
        timeout_s = self._coerce_timeout(arguments.get("timeout"), default=300.0)
        proc = await asyncio.create_subprocess_exec(
            self._get_user_python(),
            "-c",
            runner,
            script_path,
            function_name,
            args_json,
            cwd=str(self._get_active_write_root()),
            env=self._clean_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate_coro = proc.communicate()
        try:
            if timeout_s and timeout_s > 0:
                stdout, stderr = await asyncio.wait_for(communicate_coro, timeout=timeout_s)
            else:
                stdout, stderr = await communicate_coro
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
            return f"Tool error: execution timeout after {timeout_s}s."
        except Exception:
            try:
                communicate_coro.close()
            except Exception:
                pass
            raise

        out_text = (stdout or b"").decode("utf-8", errors="ignore")
        err_text = (stderr or b"").decode("utf-8", errors="ignore")
        marker = "__BABYBOT_RESULT__"
        payload_line = ""
        for line in reversed(out_text.splitlines()):
            if line.startswith(marker):
                payload_line = line[len(marker) :].strip()
                break
        if not payload_line:
            combined = (out_text.strip() + ("\n" + err_text.strip() if err_text.strip() else "")).strip()
            return combined or "Tool error: no result returned."
        try:
            payload = json.loads(payload_line)
        except json.JSONDecodeError:
            return payload_line
        if not payload.get("ok", False):
            detail = payload.get("error", "external execution failed")
            return f"Tool error: {detail}"
        return CallableTool._normalize_result(payload.get("result"))

    @staticmethod
    def _normalize_keywords(
        raw_keywords: Any,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        values: list[str] = []
        if isinstance(raw_keywords, (list, tuple)):
            values.extend(str(x) for x in raw_keywords if str(x).strip())
        elif isinstance(raw_keywords, str) and raw_keywords.strip():
            values.extend([w for w in re.split(r"[,\n]+", raw_keywords) if w.strip()])
        values.extend(str(item) for item in fallback if str(item).strip())
        terms: set[str] = set()
        for value in values:
            terms.update(ResourceManager._tokenize(str(value)))
        return tuple(sorted(terms))

    @staticmethod
    def _normalize_phrases(
        raw_keywords: Any,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        phrases: list[str] = []
        if isinstance(raw_keywords, (list, tuple)):
            phrases.extend(str(x).strip().lower() for x in raw_keywords if str(x).strip())
        elif isinstance(raw_keywords, str) and raw_keywords.strip():
            phrases.extend(
                p.strip().lower()
                for p in re.split(r"[,\n]+", raw_keywords)
                if p.strip()
            )
        phrases.extend(str(x).strip().lower() for x in fallback if str(x).strip())
        normalized = []
        for phrase in phrases:
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase and len(phrase) >= 2:
                normalized.append(phrase)
        return tuple(sorted(set(normalized)))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Language-agnostic tokenizer for skill retrieval.

        - English/latin: lowercase words and 3-gram word chunks for fuzzy matching.
        - CJK: contiguous Han segments + bi-grams.
        """
        text = (text or "").lower()
        tokens: set[str] = set()
        # Latin words
        words = re.findall(r"[a-z0-9_]{2,}", text)
        tokens.update(words)
        for word in words:
            if len(word) >= 5:
                for i in range(0, len(word) - 2):
                    tokens.add(word[i : i + 3])
        # CJK segments + bi-grams
        for seg in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            tokens.add(seg)
            if len(seg) >= 2:
                for i in range(0, len(seg) - 1):
                    tokens.add(seg[i : i + 2])
        return {t for t in tokens if t}

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
        if not text.startswith("---\n"):
            return {}, text.strip()
        end = text.find("\n---", 4)
        if end == -1:
            return {}, text.strip()
        header = text[4:end].strip()
        body = text[end + 4 :].strip()
        meta: dict[str, str] = {}
        for line in header.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("'\"")
        return meta, body

    @classmethod
    def _read_skill_document(cls, skill_dir: Path) -> tuple[dict[str, str], str]:
        skill_md = skill_dir / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        meta, body = cls._parse_frontmatter(text)
        prompt = body.strip()
        if len(prompt) > 4000:
            prompt = prompt[:4000]
        return meta, prompt

    def _setup_tool_groups(self, user_groups: dict[str, dict]) -> None:
        defaults = {
            "basic": ToolGroup("basic", "核心工具：定时/延时发送消息、任务调度管理（创建/修改/删除/列出定时任务）", active=True),
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
                description=conf.get("description", old.description if old else f"{name} tools"),
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
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower()).strip("-")
        return slug or "resource"

    def _skill_resource_id(self, skill: LoadedSkill) -> str:
        return f"skill.{self._resource_slug(skill.name)}"

    def _mcp_resource_id(self, server_name: str) -> str:
        return f"mcp.{self._resource_slug(server_name)}"

    def _group_resource_id(self, group_name: str) -> str:
        return f"group.{self._resource_slug(group_name)}"

    @staticmethod
    def _lease_to_dict(lease: ToolLease) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if lease.include_groups:
            payload["include_groups"] = list(lease.include_groups)
        if lease.include_tools:
            payload["include_tools"] = list(lease.include_tools)
        if lease.exclude_tools:
            payload["exclude_tools"] = list(lease.exclude_tools)
        return payload

    def _get_current_task_lease_var(self) -> contextvars.ContextVar[ToolLease | None]:
        current = getattr(self, "_current_task_lease", None)
        if current is None:
            current = contextvars.ContextVar("current_task_lease", default=None)
            self._current_task_lease = current
        return current

    def _get_current_skill_ids_var(self) -> contextvars.ContextVar[tuple[str, ...] | None]:
        current = getattr(self, "_current_skill_ids", None)
        if current is None:
            current = contextvars.ContextVar("current_skill_ids", default=None)
            self._current_skill_ids = current
        return current

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return self._catalog_view().get_resource_briefs()

    def _get_resource_briefs(self) -> list[dict[str, Any]]:
        """Build concise resource descriptors for top-level routing."""
        briefs: list[ResourceBrief] = []
        mcp_groups = set(self.mcp_server_groups.values())

        for server_name, group_name in sorted(self.mcp_server_groups.items()):
            group = self.groups.get(group_name)
            tools = self.registry.list(ToolLease(include_groups=(group_name,)))
            tool_count = len(tools)
            briefs.append(
                ResourceBrief(
                    resource_id=self._mcp_resource_id(server_name),
                    resource_type="mcp",
                    name=server_name,
                    purpose=(group.description if group else f"MCP tools from {server_name}"),
                    group=group_name,
                    tool_count=tool_count,
                    tools_preview=self._preview_tool_names(ToolLease(include_groups=(group_name,))),
                    active=(bool(group.active) if group else True) and tool_count > 0,
                )
            )

        for skill in sorted(self.skills.values(), key=lambda s: s.name.lower()):
            if not skill.active:
                continue
            skill_lease = skill.lease or ToolLease()
            tools = self.registry.list(skill_lease) if skill_lease.include_groups or skill_lease.include_tools else []
            tool_count = len(tools)
            briefs.append(
                ResourceBrief(
                    resource_id=self._skill_resource_id(skill),
                    resource_type="skill",
                    name=skill.name,
                    purpose=skill.description or f"Skill: {skill.name}",
                    group=skill.tool_group,
                    tool_count=tool_count,
                    tools_preview=self._preview_tool_names(skill_lease),
                    active=skill.active and tool_count > 0,
                )
            )

        for group_name, group in sorted(self.groups.items()):
            if group_name.startswith("skill_"):
                continue
            if group_name in mcp_groups:
                continue
            tools = self.registry.list(ToolLease(include_groups=(group_name,)))
            tool_count = len(tools)
            briefs.append(
                ResourceBrief(
                    resource_id=self._group_resource_id(group_name),
                    resource_type="tool_group",
                    name=group_name,
                    purpose=group.description,
                    group=group_name,
                    tool_count=tool_count,
                    tools_preview=self._preview_tool_names(ToolLease(include_groups=(group_name,))),
                    active=group.active and tool_count > 0,
                )
            )

        return [brief.to_dict() for brief in briefs]

    def _preview_tool_names(
        self,
        lease: ToolLease,
        limit: int = 6,
    ) -> tuple[str, ...]:
        names = sorted(registered.tool.name for registered in self.registry.list(lease))
        return tuple(names[:limit])

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
        """Map resource id to a least-privilege lease and optional skill ids."""
        normalized = (resource_id or "").strip().lower()
        if not normalized:
            return None

        for skill in self.skills.values():
            if not skill.active:
                continue
            if normalized == self._skill_resource_id(skill):
                lease = skill.lease or ToolLease()
                if require_tools and not self.registry.list(lease):
                    return None
                return self._lease_to_dict(lease), (skill.name.strip().lower(),)

        for server_name, group_name in self.mcp_server_groups.items():
            if normalized == self._mcp_resource_id(server_name):
                lease = ToolLease(include_groups=(group_name,))
                if require_tools and not self.registry.list(lease):
                    return None
                return self._lease_to_dict(lease), ()

        for group_name in self.groups:
            if normalized == self._group_resource_id(group_name):
                lease = ToolLease(include_groups=(group_name,))
                if require_tools and not self.registry.list(lease):
                    return None
                return self._lease_to_dict(lease), ()

        return None

    def set_scheduled_task_manager(self, manager: "ScheduledTaskManager | None") -> None:
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
        if group_name not in self.groups:
            self.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"{group_name} tools",
                active=False,
            )
        name = func_name or getattr(func, "__name__", "tool")
        self.registry.register(
            CallableTool(
                func=func,
                name=name,
                description=(inspect.getdoc(func) or "").splitlines()[0] if inspect.getdoc(func) else name,
                schema=self._json_schema_for_callable(func),
                preset_kwargs=preset_kwargs,
                resource_manager=self,
            ),
            group=group_name,
        )

    def _register_custom_tools(self, custom_tools: dict[str, dict]) -> None:
        self._ensure_workspace_on_pythonpath()
        for name, tool_conf in custom_tools.items():
            try:
                module = self._load_tool_module(tool_conf["module"])
                func_name = tool_conf.get("function", name)
                func = getattr(module, func_name, None)
                if func is None:
                    continue
                preset_kwargs = dict(tool_conf.get("preset_kwargs", {}))
                for key, value in preset_kwargs.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        preset_kwargs[key] = os.getenv(value[2:-1], value)
                self.register_tool(
                    func=func,
                    group_name=tool_conf.get("group_name", "basic"),
                    preset_kwargs=preset_kwargs,
                    func_name=func_name,
                )
            except Exception as exc:
                logger.warning("Failed to register custom tool %s: %s", name, exc)

    def _discover_workspace_tools(self) -> None:
        tools_root = self.config.workspace_tools_dir
        if not tools_root.exists():
            return
        self._ensure_workspace_on_pythonpath()
        for py_file in sorted(tools_root.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            rel = py_file.relative_to(tools_root)
            group_name = rel.parts[0] if len(rel.parts) > 1 else "basic"
            try:
                module = self._load_tool_module(str(Path("tools") / rel))
                for func_name, func in inspect.getmembers(module, inspect.isfunction):
                    if func.__module__ != module.__name__ or func_name.startswith("_"):
                        continue
                    self.register_tool(func, group_name=group_name, func_name=func_name)
            except Exception as exc:
                logger.warning("Failed to load tools from %s: %s", py_file, exc)

    def _ensure_workspace_on_pythonpath(self) -> None:
        workspace = str(self.config.workspace_dir.resolve())
        if workspace not in sys.path:
            sys.path.insert(0, workspace)

    def _load_tool_module(self, module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
        resolved = self.config.resolve_workspace_path(module_name)
        spec = importlib.util.spec_from_file_location(
            f"babybot_custom_{abs(hash(resolved))}",
            resolved,
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        raise ModuleNotFoundError(f"Cannot import custom module: {module_name}")

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
        keyword = (query or "").strip().lower()
        groups = [
            {
                "name": g.name,
                "active": g.active,
                "description": g.description,
            }
            for g in self.groups.values()
            if not keyword or keyword in g.name.lower() or keyword in g.description.lower()
        ]
        tools = []
        for registered in self.registry.list():
            if keyword and keyword not in registered.tool.name.lower():
                continue
            tools.append({"name": registered.tool.name, "group": registered.group})
        skills = []
        for skill in self.skills.values():
            if keyword and keyword not in skill.name.lower() and keyword not in skill.description.lower():
                continue
            skills.append(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                    "active": skill.active,
                }
            )
        return {
            "query": query or "",
            "groups": groups,
            "tools": tools,
            "mcp_servers": list(self.mcp_clients.keys()),
            "skills": skills,
        }

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
        from .channels.tools import ChannelToolContext

        write_root = self._get_output_dir()
        token = self._active_write_root.set(str(write_root))
        started = time.perf_counter()
        scope_token: contextvars.Token[ToolLease | None] | None = None
        skill_ids_token: contextvars.Token[tuple[str, ...] | None] | None = None
        try:
            merged_lease = self._build_task_lease(lease or {})
            skill_packs = await self._select_skill_packs(
                task_description,
                skill_ids=skill_ids,
            )
            for skill in skill_packs:
                merged_lease = ToolLease(
                    include_groups=tuple(
                        sorted(
                            set(merged_lease.include_groups) | set(skill.tool_lease.include_groups)
                        )
                    ),
                    include_tools=tuple(
                        sorted(
                            set(merged_lease.include_tools) | set(skill.tool_lease.include_tools)
                        )
                    ),
                    exclude_tools=tuple(
                        sorted(
                            set(merged_lease.exclude_tools) | set(skill.tool_lease.exclude_tools)
                        )
                    ),
                )
            scope_token = self._get_current_task_lease_var().set(merged_lease)
            skill_ids_token = self._get_current_skill_ids_var().set(
                tuple(skill_ids) if skill_ids is not None else None
            )
            tools_text = ", ".join(
                sorted(t.tool.name for t in self.registry.list(merged_lease))
            ) or "无"
            logger.info(
                "Run subagent agent=%s write_root=%s selected_skills=%s tools=%s include_groups=%s include_tools=%s exclude_tools=%s",
                agent_name,
                write_root,
                [s.name for s in skill_packs],
                tools_text,
                list(merged_lease.include_groups),
                list(merged_lease.include_tools),
                list(merged_lease.exclude_tools),
            )
            sys_prompt = self._build_worker_sys_prompt(
                agent_name=agent_name,
                task_description=task_description,
                tools_text=tools_text,
                selected_skill_packs=skill_packs,
                merged_lease=merged_lease,
            )
            # Strip tool_leases from skill packs before passing to executor:
            # run_subagent_task already merged skill groups into merged_lease
            # via UNION.  If we pass the original skill packs, the executor's
            # merge_leases() would INTERSECT them again, dropping basic/code/
            # channel groups and potentially leaving tools=[].
            executor_skills = [
                SkillPack(name=sp.name, system_prompt=sp.system_prompt)
                for sp in skill_packs
            ]
            executor = create_worker_executor(
                config=self.config,
                tools=self.registry,
                sys_prompt=sys_prompt,
                skill_packs=executor_skills,
                gateway=self._get_shared_gateway(),
            )
            exec_context = ExecutionContext(
                session_id=agent_name,
                state={
                    k: v for k, v in [
                        ("heartbeat", heartbeat),
                        ("tape", tape),
                        ("tape_store", tape_store),
                        ("context_history_tokens", self.config.system.context_history_tokens),
                        ("media_paths", media_paths),
                        ("channel_context", ChannelToolContext.get_current()),
                    ] if v is not None
                },
            )
            result = await executor.execute(
                TaskContract(
                    task_id=agent_name,
                    description=task_description,
                    lease=merged_lease,
                    retries=0,
                ),
                exec_context,
            )
            text = result.output if result.status == "succeeded" else result.error
            if result.status != "succeeded":
                logger.error(
                    "Subagent failed agent=%s status=%s error=%s metadata=%s",
                    agent_name,
                    result.status,
                    result.error,
                    (result.metadata or {}),
                )
            logger.info(
                "Run subagent done agent=%s status=%s elapsed=%.2fs output_len=%d",
                agent_name,
                result.status,
                time.perf_counter() - started,
                len(text or ""),
            )
            collected_media = list(exec_context.state.get("media_paths_collected", []))
            fallback_media = self._extract_media_from_text(text)
            merged_media = list(dict.fromkeys(collected_media + fallback_media))
            return text.strip() or "任务完成但没有文本输出。", merged_media
        except Exception:
            logger.exception("Run subagent crashed agent=%s", agent_name)
            raise
        finally:
            if skill_ids_token is not None:
                self._get_current_skill_ids_var().reset(skill_ids_token)
            if scope_token is not None:
                self._get_current_task_lease_var().reset(scope_token)
            self._active_write_root.reset(token)

    def _build_task_lease(self, lease: dict[str, Any]) -> ToolLease:
        include_groups = lease.get("include_groups")
        if include_groups is None:
            # Fallback: all active groups EXCEPT channel tool groups.
            # Sub-agents must not get channel send-message tools by default;
            # those must be explicitly requested in the task lease.
            include_groups = [
                name for name, group in self.groups.items()
                if group.active and not name.startswith("channel_")
            ]
        else:
            # Always include the "basic" group so fundamental tools
            # (scheduled tasks, worker dispatch, etc.) are available.
            # "basic" must never contain channel_* tools — channel groups use
            # the "channel_{name}" naming convention and are registered
            # separately via register_channel_tools().
            explicit_channel_groups = {
                g for g in include_groups if g.startswith("channel_")
            }
            include_groups = list(include_groups)
            if "basic" not in include_groups:
                include_groups.append("basic")
            # Defensive: strip any channel_* groups that were not part of the
            # original explicit request (guards against future misconfiguration
            # where a non-channel group name starts with "channel_" by mistake).
            include_groups = [
                g for g in include_groups
                if not g.startswith("channel_") or g in explicit_channel_groups
            ]

        # Validate include_tools against registered tool names.
        # The LLM planner may hallucinate tool names; drop any that don't exist
        # so they don't accidentally filter out all real tools.
        raw_include_tools = lease.get("include_tools") or ()
        known_tools = set(self.registry._tools.keys())
        valid_include_tools = [t for t in raw_include_tools if t in known_tools]
        if raw_include_tools and not valid_include_tools:
            logger.warning(
                "Lease include_tools contained no valid names, ignoring: %s",
                list(raw_include_tools),
            )

        return ToolLease(
            include_groups=tuple(include_groups or ()),
            include_tools=tuple(valid_include_tools),
            exclude_tools=tuple(lease.get("exclude_tools") or ()),
        )

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
            raise RuntimeError(
                "Scheduled task manager is unavailable in this runtime."
            )
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
            return json.dumps({"name": name, "deleted": deleted}, ensure_ascii=False, indent=2)

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
        active = [s for s in self.skills.values() if s.active]
        if not active:
            return []
        if skill_ids is not None:
            wanted = {
                item.strip().lower()
                for item in skill_ids
                if isinstance(item, str) and item.strip()
            }
            selected = [
                s for s in active
                if s.name.strip().lower() in wanted
                or self._skill_resource_id(s) in wanted
            ]
            return [
                SkillPack(name=s.name, system_prompt=s.prompt, tool_lease=s.lease)
                for s in selected
            ]
        return [
            SkillPack(name=s.name, system_prompt=s.prompt, tool_lease=s.lease)
            for s in active
        ]


    def _build_worker_sys_prompt(
        self,
        agent_name: str,
        task_description: str,
        tools_text: str,
        selected_skill_packs: list[SkillPack],
        merged_lease: "ToolLease | None" = None,
    ) -> str:
        selected_names = ", ".join(skill.name for skill in selected_skill_packs) or "无"
        if merged_lease is not None:
            skill_catalog = self._format_skill_catalog_for_lease(merged_lease, max_items=24)
        else:
            skill_catalog = self._format_skill_catalog(max_items=24)
        lines = [
            "你是 %s，请完成任务并输出最终答案。" % agent_name,
            "任务：%s" % task_description,
            "已激活技能（本次强相关）：%s" % selected_names,
            "可用技能目录（按需选择）：\n%s" % skill_catalog,
            "可用工具：%s" % tools_text,
            "要求：",
            "1. 当任务需要生成图片、查询信息、发送消息等操作时，必须调用对应工具，不能仅用文字描述。",
            "2. 禁止编造工具执行结果或虚构文件路径。",
            "3. 用户询问你能做什么或能力范围时，必须优先介绍可用技能目录与工具能力。",
            "4. 如需某技能但本次未激活，可在回答中明确指出可切换到该技能处理。",
        ]
        return "\n".join(lines)

    def _format_skill_catalog_for_lease(self, lease: "ToolLease", max_items: int = 20) -> str:
        """Format skill catalog showing only skills accessible under the given lease."""
        lease_groups = set(lease.include_groups)
        lease_tools = set(lease.include_tools)

        # No group/tool restrictions on the lease → show all (same as _format_skill_catalog)
        if not lease_groups and not lease_tools:
            return self._format_skill_catalog(max_items=max_items)

        accessible: list["LoadedSkill"] = []
        for skill in sorted(self.skills.values(), key=lambda s: s.name.lower()):
            if not skill.active:
                continue
            skill_lease = skill.lease or ToolLease()
            skill_groups = set(skill_lease.include_groups)
            skill_tools = set(skill_lease.include_tools)
            if not skill_groups and not skill_tools:
                # Prompt-only skill with no tool requirements → always accessible
                accessible.append(skill)
            elif skill_groups & lease_groups:
                accessible.append(skill)
            elif skill_tools and (skill_tools & lease_tools):
                accessible.append(skill)

        if not accessible:
            return "- 无"
        lines: list[str] = []
        for idx, skill in enumerate(accessible, start=1):
            if idx > max_items:
                lines.append(f"- ... 还有 {len(accessible) - max_items} 个技能")
                break
            desc = skill.description.strip() or "无描述"
            lines.append(f"- {skill.name}: {desc}")
        return "\n".join(lines)

    def _format_skill_catalog(self, max_items: int = 20) -> str:
        skills = [s for s in self.skills.values() if s.active]
        if not skills:
            return "- 无"
        lines: list[str] = []
        for idx, skill in enumerate(sorted(skills, key=lambda s: s.name.lower()), start=1):
            if idx > max_items:
                lines.append(f"- ... 还有 {len(skills) - max_items} 个技能")
                break
            desc = skill.description.strip() or "无描述"
            lines.append(f"- {skill.name}: {desc}")
        return "\n".join(lines)

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

    def _get_user_python(self) -> str:
        """Return a Python executable for user code, avoiding the project venv."""
        configured = self.config.system.python_executable
        if configured:
            return configured

        def _is_venv_path(path: str) -> bool:
            normalized = path.replace("\\", "/").lower()
            return "/.venv/" in normalized or "/venv/" in normalized

        found = shutil.which("python3")
        if found and not _is_venv_path(found):
            return found
        found_py = shutil.which("python")
        if found_py and not _is_venv_path(found_py):
            return found_py

        preferred = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
        ]
        for candidate in preferred:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        return sys.executable

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
        ws = str(self._get_active_write_root())
        proc = await asyncio.create_subprocess_exec(
            self._get_user_python(),
            "-c",
            f"import os\nos.chdir({ws!r})\n{code}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._clean_env(),
        )
        timeout_s = self._coerce_timeout(timeout, default=300.0)
        communicate_coro = proc.communicate()
        try:
            if timeout_s and timeout_s > 0:
                stdout, stderr = await asyncio.wait_for(communicate_coro, timeout=timeout_s)
            else:
                stdout, stderr = await communicate_coro
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
            return f"Timeout: python execution exceeded {timeout_s}s."
        except Exception:
            try:
                communicate_coro.close()
            except Exception:
                pass
            raise
        out = (stdout or b"").decode("utf-8", errors="ignore")
        err = (stderr or b"").decode("utf-8", errors="ignore")
        text = out.strip()
        if err.strip():
            text = f"{text}\n{err.strip()}".strip()
        return text

    async def _workspace_execute_shell_command(
        self,
        command: str,
        timeout: float | int | str | None = 300,
        **kwargs: Any,
    ) -> str:
        safety_error = _check_shell_safety(command)
        if safety_error:
            return safety_error
        ws = shlex.quote(str(self._get_active_write_root()))
        guarded = f"cd {ws} && {command}"
        proc = await asyncio.create_subprocess_shell(
            guarded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._clean_env(),
        )
        timeout_s = self._coerce_timeout(timeout, default=300.0)
        communicate_coro = proc.communicate()
        try:
            if timeout_s and timeout_s > 0:
                stdout, stderr = await asyncio.wait_for(communicate_coro, timeout=timeout_s)
            else:
                stdout, stderr = await communicate_coro
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
            return f"Timeout: shell command exceeded {timeout_s}s."
        except Exception:
            try:
                communicate_coro.close()
            except Exception:
                pass
            raise
        out = (stdout or b"").decode("utf-8", errors="ignore")
        err = (stderr or b"").decode("utf-8", errors="ignore")
        text = out.strip()
        if err.strip():
            text = f"{text}\n{err.strip()}".strip()
        return text

    async def _workspace_view_text_file(
        self,
        file_path: str,
        ranges: list[int] | None = None,
    ) -> str:
        resolved, err = self._resolve_workspace_file(file_path)
        if err:
            return err
        with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if not ranges:
            return "".join(lines)
        chunks: list[str] = []
        for i in range(0, len(ranges), 2):
            start = max(1, int(ranges[i]))
            end = int(ranges[i + 1]) if i + 1 < len(ranges) else start
            chunks.extend(lines[start - 1 : end])
        return "".join(chunks)

    async def _workspace_write_text_file(
        self,
        file_path: str,
        content: str,
        ranges: list[int] | None = None,
    ) -> str:
        resolved, err = self._resolve_workspace_file(file_path)
        if err:
            return err
        target = Path(resolved)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not ranges:
            target.write_text(content, encoding="utf-8")
            return f"Wrote file: {target}"
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True) if target.exists() else []
        start = max(1, int(ranges[0])) if ranges else 1
        end = int(ranges[1]) if ranges and len(ranges) > 1 else start
        replacement = content.splitlines(keepends=True)
        lines[start - 1 : end] = replacement
        target.write_text("".join(lines), encoding="utf-8")
        return f"Updated file range in: {target}"

    async def _workspace_insert_text_file(
        self,
        file_path: str,
        content: str,
        line_number: int,
    ) -> str:
        resolved, err = self._resolve_workspace_file(file_path)
        if err:
            return err
        target = Path(resolved)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True) if target.exists() else []
        idx = max(0, min(len(lines), int(line_number) - 1))
        lines[idx:idx] = content.splitlines(keepends=True)
        target.write_text("".join(lines), encoding="utf-8")
        return f"Inserted text into: {target}"

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
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if name in {"self", "context"}:
                continue
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            anno = hints.get(name, param.annotation)
            properties[name] = ResourceManager._schema_for_annotation(anno)
            if param.default is inspect._empty:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
        if annotation is inspect._empty:
            return {"type": "string"}

        origin = get_origin(annotation)
        args = [arg for arg in get_args(annotation) if arg is not type(None)]

        if origin is None:
            if annotation in {str}:
                return {"type": "string"}
            if annotation in {bool}:
                return {"type": "boolean"}
            if annotation in {int}:
                return {"type": "integer"}
            if annotation in {float}:
                return {"type": "number"}
            if annotation in {dict}:
                return {"type": "object"}
            if annotation in {list, tuple, set}:
                return {"type": "array"}
            return {"type": "string"}

        if origin in {list, tuple, set}:
            item_schema = (
                ResourceManager._schema_for_annotation(args[0])
                if args else {"type": "string"}
            )
            return {"type": "array", "items": item_schema}

        if origin is dict:
            return {"type": "object"}

        if origin is Literal:
            values = [x for x in args if isinstance(x, (str, int, float, bool))]
            if not values:
                return {"type": "string"}
            first = values[0]
            if isinstance(first, bool):
                field_type = "boolean"
            elif isinstance(first, int):
                field_type = "integer"
            elif isinstance(first, float):
                field_type = "number"
            else:
                field_type = "string"
            return {"type": field_type, "enum": values}

        if origin in {Union, UnionType}:
            if len(args) == 1:
                return ResourceManager._schema_for_annotation(args[0])
            return {"anyOf": [ResourceManager._schema_for_annotation(arg) for arg in args]}

        return {"type": "string"}
