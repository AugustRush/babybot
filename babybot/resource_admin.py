from __future__ import annotations

import contextvars
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent_kernel import ToolLease
from .context_views import build_context_view
from .resource_models import LoadedSkill
from .resource_scope import ResourceScopeHelper

if TYPE_CHECKING:
    from .resource import ResourceManager

logger = logging.getLogger(__name__)


class ResourceAdminHelper:
    """Admin, diagnostics, and skill-state helpers for ResourceManager."""

    def __init__(self, owner: ResourceManager) -> None:
        self._owner = owner

    def resolve_skill_record(self, skill_name: str) -> LoadedSkill | None:
        normalized = (skill_name or "").strip().lower()
        if not normalized:
            return None
        direct = self._owner.skills.get(normalized)
        if direct is not None:
            return direct
        for skill in self._owner.skills.values():
            if normalized in {
                skill.name.strip().lower(),
                self._owner._skill_resource_id(skill).strip().lower(),
                Path(skill.directory).name.strip().lower(),
            }:
                return skill
        return None

    def resolve_skill_directory_input(self, skill_path: str) -> Path:
        raw = (skill_path or "").strip()
        if not raw:
            return Path(raw).expanduser().resolve()

        normalized = raw.lower()
        for skill in getattr(self._owner, "skills", {}).values():
            if normalized in {
                skill.name.strip().lower(),
                self._owner._skill_resource_id(skill).strip().lower(),
            }:
                return Path(skill.directory).expanduser().resolve()

        candidates: list[Path] = []
        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(Path(self._owner.config.resolve_workspace_path(raw)))
            candidates.append(
                (self._owner.config.workspace_skills_dir / raw_path).expanduser()
            )
            candidates.append(
                (self._owner.config.builtin_skills_dir / raw_path).expanduser()
            )
            candidates.append(raw_path)

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.name == "SKILL.md":
                return resolved.parent
            if resolved.is_dir() and (resolved / "SKILL.md").exists():
                return resolved

        fallback = candidates[0] if candidates else raw_path
        resolved = fallback.resolve()
        if resolved.name == "SKILL.md":
            return resolved.parent
        return resolved

    def record_runtime_hint(self, message: str) -> None:
        text = (message or "").strip()
        if not text:
            return
        ctx = self._owner._get_current_tool_context_var().get()
        state = getattr(ctx, "state", None)
        if not isinstance(state, dict):
            return
        hints = state.setdefault("pending_runtime_hints", [])
        if isinstance(hints, list):
            hints.append(text)

    def reload_skill(self, skill_path: str) -> str:
        skill_dir = self.resolve_skill_directory_input(skill_path)
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
            return f"Not a valid skill directory: {skill_dir}"

        loader = self._owner._skill_loader_view()
        meta, prompt_summary, _prompt_body = loader.read_skill_document(skill_dir)
        name = meta.get("name", skill_dir.name)
        key = name.strip().lower()

        old = self._owner.skills.get(key)
        old_active = old.active if old is not None else True
        if old and old.tool_group:
            removed = self._owner.registry.unregister_group(old.tool_group)
            self._owner.groups.pop(old.tool_group, None)
            logger.info(
                "Unregistered %d old tools for skill %s",
                len(removed),
                name,
            )

        runtime = self._owner._build_skill_runtime(meta)
        tool_group, tool_names = self._owner._register_skill_tools(
            name,
            skill_dir,
            runtime=runtime,
        )

        meta_include_groups = loader._parse_frontmatter_list(meta.get("include_groups"))
        meta_include_tools = loader._parse_frontmatter_list(meta.get("include_tools"))
        meta_exclude_tools = loader._parse_frontmatter_list(meta.get("exclude_tools"))

        description = meta.get("description", f"Skill: {name}")
        keywords = loader.normalize_keywords(
            None, fallback=(description, name, prompt_summary[:400])
        )
        phrases = loader.normalize_phrases(None, fallback=(description, name))

        self._owner._upsert_skill(
            LoadedSkill(
                name=name,
                description=description,
                directory=str(skill_dir),
                prompt=prompt_summary,
                prompt_body="",
                prompt_body_path=str((skill_dir / "SKILL.md").resolve()),
                keywords=keywords,
                phrases=phrases,
                source="hot-reload",
                active=old_active,
                lease=ToolLease(
                    include_groups=tuple(
                        dict.fromkeys(
                            [
                                *meta_include_groups,
                                *([tool_group] if tool_group else []),
                            ]
                        )
                    ),
                    include_tools=tuple(meta_include_tools),
                    exclude_tools=tuple(meta_exclude_tools),
                ),
                tool_group=tool_group,
                tools=tool_names,
                runtime=runtime,
            )
        )
        skill_md = str((skill_dir / "SKILL.md").resolve())
        self.record_runtime_hint(
            "技能已热重载："
            f"{name}\n"
            f"skill_dir={skill_dir}\n"
            f"SKILL.md={skill_md}\n"
            "当前运行中的 agent 不会自动重建技能快照或扩展 lease。"
        )
        logger.info("Hot-reloaded skill %s with %d tools", name, len(tool_names))
        parts = [f"Skill '{name}' reloaded successfully."]
        if tool_names:
            parts.append(f"Registered tools: {', '.join(tool_names)}")
        return " ".join(parts)

    def get_assistant_profile(self) -> str:
        memory_store = getattr(self._owner, "memory_store", None)
        if memory_store is None:
            return "暂无 assistant profile：memory_store 未初始化。"
        load_profile = getattr(memory_store, "load_assistant_profile", None)
        if not callable(load_profile):
            return "暂无 assistant profile：memory_store 不支持读取。"
        text = str(load_profile() or "").strip()
        return text or "assistant profile 为空。"

    def set_assistant_profile(self, content: str, mode: str = "replace") -> str:
        memory_store = getattr(self._owner, "memory_store", None)
        if memory_store is None:
            return "无法修改 assistant profile：memory_store 未初始化。"
        save_profile = getattr(memory_store, "save_assistant_profile", None)
        if not callable(save_profile):
            return "无法修改 assistant profile：memory_store 不支持写入。"
        normalized_mode = str(mode or "replace").strip().lower()
        if normalized_mode not in {"replace", "append"}:
            return "Unsupported mode. Use replace or append."
        incoming = str(content or "").strip()
        if normalized_mode == "append":
            current = self.get_assistant_profile()
            base = (
                ""
                if current.startswith("暂无 assistant profile")
                or current == "assistant profile 为空。"
                else current
            )
            incoming = f"{base}\n\n{incoming}".strip() if base else incoming
        save_profile(incoming)
        return "Assistant profile updated successfully."

    def list_admin_skills(
        self,
        query: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        return self.inspect_skills(
            query=query,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    def enable_skill(self, skill_name: str) -> str:
        skill = self.resolve_skill_record(skill_name)
        if skill is None:
            return f"Skill not found: {skill_name}"
        if skill.active:
            return f"Skill '{skill.name}' is already enabled."
        skill.active = True
        return f"Skill '{skill.name}' enabled."

    def disable_skill(self, skill_name: str) -> str:
        skill = self.resolve_skill_record(skill_name)
        if skill is None:
            return f"Skill not found: {skill_name}"
        if not skill.active:
            return f"Skill '{skill.name}' is already disabled."
        skill.active = False
        return f"Skill '{skill.name}' disabled."

    def delete_skill(self, skill_name: str) -> str:
        skill = self.resolve_skill_record(skill_name)
        if skill is None:
            return f"Skill not found: {skill_name}"

        skill_dir = Path(skill.directory).expanduser().resolve()
        workspace_root = self._owner.config.workspace_skills_dir.expanduser().resolve()
        try:
            skill_dir.relative_to(workspace_root)
        except ValueError:
            return (
                "Refusing to delete non-workspace skill. "
                "Only workspace custom skills can be deleted."
            )

        if skill.tool_group:
            removed = self._owner.registry.unregister_group(skill.tool_group)
            self._owner.groups.pop(skill.tool_group, None)
            logger.info(
                "Deleted skill %s and unregistered %d tools",
                skill.name,
                len(removed),
            )

        self._owner.skills.pop(skill.name.strip().lower(), None)
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
        return f"Skill '{skill.name}' deleted from workspace."

    def default_chat_key(self) -> str:
        from .channels.tools import ChannelToolContext

        ctx = ChannelToolContext.get_current()
        if ctx is None or not ctx.chat_id:
            return ""
        channel_name = (ctx.channel_name or "").strip()
        if channel_name:
            return f"{channel_name}:{ctx.chat_id}"
        return str(ctx.chat_id)

    def inspect_runtime_flow(self, flow_id: str = "", chat_key: str = "") -> str:
        provider = getattr(self._owner, "_observability_provider", None)
        resolved_chat_key = chat_key.strip() or self.default_chat_key()
        if provider is not None and hasattr(provider, "inspect_runtime_flow"):
            return str(
                provider.inspect_runtime_flow(
                    flow_id=flow_id.strip(),
                    chat_key=resolved_chat_key,
                )
            )
        if not flow_id and not resolved_chat_key:
            return "暂无可观测的运行中 flow。"
        return f"flow={flow_id.strip()};chat={resolved_chat_key}"

    def inspect_chat_context(self, chat_key: str = "", query: str = "") -> str:
        provider = getattr(self._owner, "_observability_provider", None)
        resolved_chat_key = chat_key.strip() or self.default_chat_key()
        if provider is not None and hasattr(provider, "inspect_chat_context"):
            return str(
                provider.inspect_chat_context(
                    chat_key=resolved_chat_key,
                    query=query.strip(),
                )
            )
        if not resolved_chat_key:
            return "缺少 chat_key，且当前上下文没有可推断的会话。"
        return self.fallback_inspect_chat_context(
            resolved_chat_key,
            query=query.strip(),
        )

    def fallback_inspect_chat_context(self, chat_key: str, query: str = "") -> str:
        memory_store = getattr(self._owner, "memory_store", None)
        if memory_store is None:
            return f"chat={chat_key}\n暂无 memory store。"
        view = build_context_view(
            memory_store=memory_store,
            chat_id=chat_key,
            query=query,
        )
        records = memory_store.list_memories(chat_id=chat_key)
        parts = ["[Chat Context]", f"chat_key={chat_key}"]
        if query:
            parts.append(f"query={query}")
        if view.hot:
            parts.append("[Hot Context]\n- " + "\n- ".join(view.hot))
        if view.warm:
            parts.append("[Warm Context]\n- " + "\n- ".join(view.warm))
        if view.cold:
            parts.append("[Cold Context]\n- " + "\n- ".join(view.cold))
        if records:
            lines = [
                f"- memory_type={record.memory_type} key={record.key} tier={record.tier} status={record.status} summary={record.summary}"
                for record in records[:12]
            ]
            parts.append("[Memory Records]\n" + "\n".join(lines))
        return "\n".join(parts)

    def inspect_policy(self, chat_key: str = "", decision_kind: str = "") -> str:
        provider = getattr(self._owner, "_observability_provider", None)
        resolved_chat_key = chat_key.strip() or self.default_chat_key()
        if provider is not None and hasattr(provider, "inspect_policy"):
            return str(
                provider.inspect_policy(
                    chat_key=resolved_chat_key,
                    decision_kind=decision_kind.strip(),
                )
            )
        return decision_kind.strip() or resolved_chat_key or "policy"

    @staticmethod
    def schema_type_summary(schema: dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return "unknown"
        schema_type = schema.get("type")
        if schema_type == "array":
            item_summary = ResourceAdminHelper.schema_type_summary(
                schema.get("items", {})
            )
            return f"array[{item_summary}]"
        if schema_type == "object" and isinstance(
            schema.get("additionalProperties"), dict
        ):
            nested = ResourceAdminHelper.schema_type_summary(
                schema["additionalProperties"]
            )
            return f"object[{nested}]"
        if schema.get("enum"):
            values = ",".join(str(value) for value in schema["enum"][:4])
            return f"{schema_type or 'enum'}({values})"
        if schema.get("anyOf"):
            variants = [
                ResourceAdminHelper.schema_type_summary(item)
                for item in schema["anyOf"]
                if isinstance(item, dict)
            ]
            return "|".join(variants) if variants else "any"
        return str(schema_type or "any")

    def inspect_tools(
        self,
        query: str = "",
        group: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        query_text = query.strip().lower()
        group_filter = group.strip().lower()
        normalized_limit = max(1, min(int(limit or 50), 200))
        normalized_offset = max(0, int(offset or 0))
        rows: list[str] = []
        for registered in sorted(self._owner.registry.list(), key=lambda item: item.tool.name):
            tool_name = registered.tool.name
            tool_group = registered.group
            group_state = self._owner.groups.get(tool_group)
            is_active = bool(group_state.active) if group_state is not None else False
            if active_only and not is_active:
                continue
            if group_filter and tool_group.lower() != group_filter:
                continue
            haystack = f"{tool_name} {tool_group}".lower()
            if query_text and query_text not in haystack:
                continue
            properties = registered.tool.schema.get("properties", {})
            schema_parts = [
                f"{name}:{self.schema_type_summary(prop)}"
                for name, prop in properties.items()
                if isinstance(prop, dict)
            ]
            schema_summary = ", ".join(schema_parts) if schema_parts else "no-args"
            rows.append(
                f"- tool={tool_name} group={tool_group} active={is_active} schema={schema_summary}"
            )
        total = len(rows)
        window = rows[normalized_offset : normalized_offset + normalized_limit]
        lines = [
            "[Tools]",
            (
                f"- total={total} returned={len(window)} "
                f"offset={normalized_offset} limit={normalized_limit}"
            ),
        ]
        if not window:
            lines.append("- no matching tools")
            return "\n".join(lines)
        lines.extend(window)
        if normalized_offset + len(window) < total:
            lines.append(
                f"[Truncated. Use offset={normalized_offset + len(window)} "
                f"limit={normalized_limit} to read more.]"
            )
        return "\n".join(lines)

    def inspect_skills(
        self,
        query: str = "",
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        query_text = query.strip().lower()
        normalized_limit = max(1, min(int(limit or 50), 200))
        normalized_offset = max(0, int(offset or 0))
        rows: list[str] = []
        for skill in sorted(self._owner.skills.values(), key=lambda item: item.name.lower()):
            if active_only and not skill.active:
                continue
            haystack = f"{skill.name} {skill.description} {skill.source}".lower()
            if query_text and query_text not in haystack:
                continue
            tool_list = ", ".join(skill.tools) if skill.tools else "-"
            rows.append(
                f"- skill={skill.name} active={skill.active} source={skill.source} "
                f"group={skill.tool_group or '-'} tools={tool_list}"
            )
        total = len(rows)
        window = rows[normalized_offset : normalized_offset + normalized_limit]
        lines = [
            "[Skills]",
            (
                f"- total={total} returned={len(window)} "
                f"offset={normalized_offset} limit={normalized_limit}"
            ),
        ]
        if not window:
            lines.append("- no matching skills")
            return "\n".join(lines)
        lines.extend(window)
        if normalized_offset + len(window) < total:
            lines.append(
                f"[Truncated. Use offset={normalized_offset + len(window)} "
                f"limit={normalized_limit} to read more.]"
            )
        return "\n".join(lines)

    def inspect_skill_load_errors(self, limit: int = 20) -> str:
        capped = max(1, int(limit or 20))
        entries = list(self._owner._skill_load_errors[-capped:])
        if not entries:
            return "[Skill Load Errors]\n- none"
        lines = ["[Skill Load Errors]"]
        for item in entries:
            lines.append(
                f"- skill={item.get('skill', '')} path={item.get('path', '')} "
                f"stage={item.get('stage', '')} error={item.get('error', '')}"
            )
        return "\n".join(lines)

    def record_skill_load_error(
        self,
        *,
        skill: str,
        path: str,
        error: str,
        stage: str,
    ) -> None:
        entry = {
            "skill": str(skill),
            "path": str(path),
            "error": str(error),
            "stage": str(stage),
        }
        self._owner._skill_load_errors.append(entry)
        if len(self._owner._skill_load_errors) > 50:
            self._owner._skill_load_errors = self._owner._skill_load_errors[-50:]

