"""Lightweight resource manager based on the in-repo kernel."""

from __future__ import annotations

import asyncio
import contextvars
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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    from .context import Tape
    from .heartbeat import Heartbeat


@dataclass
class ToolGroup:
    name: str
    description: str
    notes: str = ""
    active: bool = False


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




class CallableTool:
    """Wrap a python callable into kernel Tool protocol."""

    def __init__(
        self,
        func: Any,
        name: str,
        description: str,
        schema: dict[str, Any],
        preset_kwargs: dict[str, Any] | None = None,
    ):
        self._func = func
        self._name = name
        self._description = description
        self._schema = schema
        self._preset_kwargs = dict(preset_kwargs or {})

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
                mgr = ResourceManager.get_instance()
                write_root = str(mgr._get_active_write_root())

                def _run_in_workspace() -> Any:
                    saved_cwd = os.getcwd()
                    try:
                        os.chdir(write_root)
                        return self._func(**kwargs)
                    finally:
                        os.chdir(saved_cwd)

                value = await loop.run_in_executor(
                    None, ctx.run, _run_in_workspace
                )
            return ToolResult(ok=True, content=self._normalize_result(value))
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


class ResourceManager:
    """Centralized resource manager without external agent frameworks."""

    _instance: "ResourceManager | None" = None
    _initialized: bool = False
    _lock = threading.Lock()
    _orchestration_tools = {"create_worker", "dispatch_workers"}
    _MEDIA_PATH_RE = re.compile(
        r"(?:^|[\s'\"(])((?:/~|[~/])?[\w./]+(?:\.(?:png|jpg|jpeg|gif|bmp|webp|pdf|txt|md|json|yaml|yml|csv|xlsx|pptx|docx|mp4|mp3|wav)))"
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> "ResourceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Config | None = None):
        if ResourceManager._initialized:
            return
        self.config = config or Config()
        self.registry = ToolRegistry()
        self.groups: dict[str, ToolGroup] = {}
        self.mcp_clients: dict[str, BaseMCPRuntimeClient] = {}
        self.channel_tools: dict[str, ChannelTools] = {}
        self.skills: dict[str, LoadedSkill] = {}
        self._active_write_root: contextvars.ContextVar[str] = contextvars.ContextVar(
            "active_write_root",
            default=str(self.config.workspace_dir.resolve()),
        )

        self._load_config()
        self._register_builtin_tools()
        ResourceManager._initialized = True

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
            try:
                module = self._load_tool_module(str(py_file))
            except Exception as exc:
                logger.warning("Failed to import skill script %s: %s", py_file, exc)
                continue
            for func_name, func in inspect.getmembers(module, inspect.isfunction):
                if func.__module__ != module.__name__ or func_name.startswith("_"):
                    continue
                tool_name = f"{slug}__{func_name}"
                self.register_tool(
                    func=func,
                    group_name=group_name,
                    func_name=tool_name,
                )
                tool_names.append(tool_name)
        return group_name, tuple(sorted(set(tool_names)))

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
            "basic": ToolGroup("basic", "Core orchestration tools", active=True),
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
        self.register_tool(self._workspace_execute_python_code, group_name="code")
        self.register_tool(self._workspace_execute_shell_command, group_name="code")
        self.register_tool(self._workspace_view_text_file, group_name="code")
        self.register_tool(self._workspace_write_text_file, group_name="code")
        self.register_tool(self._workspace_insert_text_file, group_name="code")

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
        heartbeat: "Heartbeat | None" = None,
    ) -> tuple[str, list[str]]:
        write_root = self._get_output_dir()
        token = self._active_write_root.set(str(write_root))
        started = time.perf_counter()
        try:
            merged_lease = self._build_task_lease(lease or {})
            skill_packs = await self._select_skill_packs(task_description)
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
            )
            result = await executor.execute(
                TaskContract(
                    task_id=agent_name,
                    description=task_description,
                    lease=merged_lease,
                    retries=0,
                ),
                ExecutionContext(
                    session_id=agent_name,
                    state={
                        k: v for k, v in [
                            ("heartbeat", heartbeat),
                            ("tape", tape),
                            ("context_history_tokens", self.config.system.context_history_tokens),
                        ] if v
                    },
                ),
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
            return text.strip() or "任务完成但没有文本输出。", self._extract_media_from_text(text)
        except Exception:
            logger.exception("Run subagent crashed agent=%s", agent_name)
            raise
        finally:
            self._active_write_root.reset(token)

    def _build_task_lease(self, lease: dict[str, Any]) -> ToolLease:
        include_groups = lease.get("include_groups")
        if include_groups is None:
            include_groups = [name for name, group in self.groups.items() if group.active]

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
        async def create_worker(task_description: str) -> str:
            text, _ = await self.run_subagent_task(task_description)
            return text

        return create_worker

    def dispatch_workers_tool(self) -> Any:
        async def dispatch_workers(
            tasks: list[str],
            max_concurrency: int = 3,
            lease: dict[str, Any] | None = None,
        ) -> str:
            normalized = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
            if not normalized:
                return "No valid tasks were provided."
            limit = max(1, min(int(max_concurrency), len(normalized), 8))
            semaphore = asyncio.Semaphore(limit)

            async def run_one(index: int, task: str) -> dict[str, Any]:
                async with semaphore:
                    try:
                        text, _ = await self.run_subagent_task(
                            task_description=task,
                            lease=lease,
                            agent_name=f"Worker-{index}",
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

    def reset(self) -> None:
        close_clients_best_effort(self.mcp_clients)
        self.registry = ToolRegistry()
        self.groups.clear()
        self.channel_tools.clear()
        self.mcp_clients.clear()
        self.skills.clear()
        self._load_config()
        self._register_builtin_tools()

    async def _select_skill_packs(self, task_description: str) -> list[SkillPack]:
        active = [s for s in self.skills.values() if s.active]
        if not active:
            return []
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
    ) -> str:
        selected_names = ", ".join(skill.name for skill in selected_skill_packs) or "无"
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

    @classmethod
    def get_instance(cls) -> "ResourceManager":
        if cls._instance is None:
            raise RuntimeError("ResourceManager not initialized yet")
        return cls._instance

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
        found = shutil.which("python3")
        if found and "venv" not in found:
            return found
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
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if name in {"self", "context"}:
                continue
            anno = param.annotation
            field_type = "string"
            if anno in {int}:
                field_type = "integer"
            elif anno in {float}:
                field_type = "number"
            elif anno in {bool}:
                field_type = "boolean"
            elif getattr(anno, "__origin__", None) in {list, tuple}:
                field_type = "array"
            elif getattr(anno, "__origin__", None) is dict:
                field_type = "object"
            properties[name] = {"type": field_type}
            if param.default is inspect._empty:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
