from __future__ import annotations

import logging
import re
from typing import Any

from .agent_kernel import ToolLease
from .resource_models import LoadedSkill, ResourceBrief

logger = logging.getLogger(__name__)


class ResourceScopeHelper:
    def __init__(self, owner: Any) -> None:
        self._owner = owner

    @staticmethod
    def resource_slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower()).strip("-")
        return slug or "resource"

    @classmethod
    def skill_resource_id(cls, skill: LoadedSkill) -> str:
        return f"skill.{cls.resource_slug(skill.name)}"

    @classmethod
    def mcp_resource_id(cls, server_name: str) -> str:
        return f"mcp.{cls.resource_slug(server_name)}"

    @classmethod
    def group_resource_id(cls, group_name: str) -> str:
        return f"group.{cls.resource_slug(group_name)}"

    @staticmethod
    def lease_to_dict(lease: ToolLease) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if lease.include_groups:
            payload["include_groups"] = list(lease.include_groups)
        if lease.include_tools:
            payload["include_tools"] = list(lease.include_tools)
        if lease.exclude_tools:
            payload["exclude_tools"] = list(lease.exclude_tools)
        return payload

    def preview_tool_names(
        self,
        lease: ToolLease,
        limit: int = 6,
    ) -> tuple[str, ...]:
        names = sorted(
            registered.tool.name for registered in self._owner.registry.list(lease)
        )
        return tuple(names[:limit])

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        briefs: list[ResourceBrief] = []
        mcp_groups = set(self._owner.mcp_server_groups.values())

        for server_name, group_name in sorted(self._owner.mcp_server_groups.items()):
            group = self._owner.groups.get(group_name)
            tool_count = len(self._owner.registry.list(ToolLease(include_groups=(group_name,))))
            briefs.append(
                ResourceBrief(
                    resource_id=self.mcp_resource_id(server_name),
                    resource_type="mcp",
                    name=server_name,
                    purpose=group.description if group else f"MCP tools from {server_name}",
                    group=group_name,
                    tool_count=tool_count,
                    tools_preview=self.preview_tool_names(
                        ToolLease(include_groups=(group_name,))
                    ),
                    active=(bool(group.active) if group else True) and tool_count > 0,
                )
            )

        for skill in sorted(self._owner.skills.values(), key=lambda item: item.name.lower()):
            if not skill.active:
                continue
            skill_lease = skill.lease or ToolLease()
            tools = (
                self._owner.registry.list(skill_lease)
                if skill_lease.include_groups or skill_lease.include_tools
                else []
            )
            tool_count = len(tools)
            briefs.append(
                ResourceBrief(
                    resource_id=self.skill_resource_id(skill),
                    resource_type="skill",
                    name=skill.name,
                    purpose=skill.description or f"Skill: {skill.name}",
                    group=skill.tool_group,
                    tool_count=tool_count,
                    tools_preview=self.preview_tool_names(skill_lease),
                    active=skill.active and tool_count > 0,
                )
            )

        for group_name, group in sorted(self._owner.groups.items()):
            if group_name.startswith("skill_") or group_name in mcp_groups:
                continue
            tool_count = len(
                self._owner.registry.list(ToolLease(include_groups=(group_name,)))
            )
            briefs.append(
                ResourceBrief(
                    resource_id=self.group_resource_id(group_name),
                    resource_type="tool_group",
                    name=group_name,
                    purpose=group.description,
                    group=group_name,
                    tool_count=tool_count,
                    tools_preview=self.preview_tool_names(
                        ToolLease(include_groups=(group_name,))
                    ),
                    active=group.active and tool_count > 0,
                )
            )

        return [brief.to_dict() for brief in briefs]

    def resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        normalized = (resource_id or "").strip().lower()
        if not normalized:
            return None

        for skill in self._owner.skills.values():
            if not skill.active:
                continue
            if normalized == self.skill_resource_id(skill):
                lease = skill.lease or ToolLease()
                if require_tools and not self._owner.registry.list(lease):
                    return None
                return self.lease_to_dict(lease), (skill.name.strip().lower(),)

        for server_name, group_name in self._owner.mcp_server_groups.items():
            if normalized == self.mcp_resource_id(server_name):
                lease = ToolLease(include_groups=(group_name,))
                if require_tools and not self._owner.registry.list(lease):
                    return None
                return self.lease_to_dict(lease), ()

        for group_name in self._owner.groups:
            if normalized == self.group_resource_id(group_name):
                lease = ToolLease(include_groups=(group_name,))
                if require_tools and not self._owner.registry.list(lease):
                    return None
                return self.lease_to_dict(lease), ()

        return None

    def search_resources(self, query: str | None = None) -> dict[str, Any]:
        keyword = (query or "").strip().lower()
        groups = [
            {
                "name": group.name,
                "active": group.active,
                "description": group.description,
            }
            for group in self._owner.groups.values()
            if not keyword
            or keyword in group.name.lower()
            or keyword in group.description.lower()
        ]
        tools = []
        for registered in self._owner.registry.list():
            if keyword and keyword not in registered.tool.name.lower():
                continue
            tools.append({"name": registered.tool.name, "group": registered.group})
        skills = []
        for skill in self._owner.skills.values():
            if (
                keyword
                and keyword not in skill.name.lower()
                and keyword not in skill.description.lower()
            ):
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
            "mcp_servers": list(self._owner.mcp_clients.keys()),
            "skills": skills,
        }

    def build_task_lease(self, lease: dict[str, Any]) -> ToolLease:
        include_groups = lease.get("include_groups")
        if include_groups is None:
            include_groups = [
                name
                for name, group in self._owner.groups.items()
                if group.active and not name.startswith("channel_")
            ]
        else:
            explicit_channel_groups = {
                group for group in include_groups if group.startswith("channel_")
            }
            include_groups = list(include_groups)
            if "basic" not in include_groups:
                include_groups.append("basic")
            include_groups = [
                group
                for group in include_groups
                if not group.startswith("channel_") or group in explicit_channel_groups
            ]

        raw_include_tools = lease.get("include_tools") or ()
        known_tools = set(self._owner.registry._tools.keys())
        valid_include_tools = [tool for tool in raw_include_tools if tool in known_tools]
        if raw_include_tools and not valid_include_tools:
            logger.warning(
                "Lease include_tools contained no valid names, ignoring: %s",
                list(raw_include_tools),
            )

        exclude_tools = set(lease.get("exclude_tools") or ())
        if not set(valid_include_tools) & self._owner._orchestration_tools:
            exclude_tools.update(self._owner._orchestration_tools)

        return ToolLease(
            include_groups=tuple(include_groups or ()),
            include_tools=tuple(valid_include_tools),
            exclude_tools=tuple(sorted(exclude_tools)),
        )
