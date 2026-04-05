from __future__ import annotations

import logging
import re
from typing import Any

from .agent_kernel import ToolLease
from .resource_models import LoadedSkill, ResourceBrief

logger = logging.getLogger(__name__)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_NETWORK_QUERY_MARKERS = (
    "http://",
    "https://",
    "www.",
    "github.com",
    "raw.githubusercontent.com",
    "仓库",
    "网页",
    "网页链接",
    "网址",
    "链接",
    "文档",
    "readme",
)
_NETWORK_RESOURCE_MARKERS = (
    "web",
    "http",
    "fetch",
    "search",
    "browser",
    "crawl",
    "url",
    "网页",
    "浏览",
    "仓库",
    "文档",
)


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

    @staticmethod
    def _filter_registered_tools(
        tools: list[Any],
        lease: ToolLease | None = None,
    ) -> list[Any]:
        lease = lease or ToolLease()
        include_groups = set(lease.include_groups)
        include_tools = set(lease.include_tools)
        exclude_tools = set(lease.exclude_tools)

        selected: list[Any] = []
        for registered in tools:
            name = registered.tool.name
            if name in exclude_tools:
                continue
            if include_tools or include_groups:
                in_tools = name in include_tools if include_tools else False
                in_groups = registered.group in include_groups if include_groups else False
                if not (in_tools or in_groups):
                    continue
            selected.append(registered)
        return selected

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        briefs: list[ResourceBrief] = []
        mcp_groups = set(self._owner.mcp_server_groups.values())
        registry_snapshot = list(self._owner.registry.list())

        def _tool_summary(lease: ToolLease) -> tuple[int, tuple[str, ...]]:
            selected = self._filter_registered_tools(registry_snapshot, lease)
            names = tuple(sorted(registered.tool.name for registered in selected))
            return len(names), names[:6]

        for server_name, group_name in sorted(self._owner.mcp_server_groups.items()):
            group = self._owner.groups.get(group_name)
            tool_count, tools_preview = _tool_summary(ToolLease(include_groups=(group_name,)))
            briefs.append(
                ResourceBrief(
                    resource_id=self.mcp_resource_id(server_name),
                    resource_type="mcp",
                    name=server_name,
                    purpose=group.description if group else f"MCP tools from {server_name}",
                    group=group_name,
                    tool_count=tool_count,
                    tools_preview=tools_preview,
                    active=(bool(group.active) if group else True) and tool_count > 0,
                )
            )

        for skill in sorted(self._owner.skills.values(), key=lambda item: item.name.lower()):
            if not skill.active:
                continue
            skill_lease = skill.lease or ToolLease()
            has_explicit_scope = bool(
                skill_lease.include_groups or skill_lease.include_tools
            )
            tool_count, tools_preview = _tool_summary(skill_lease) if has_explicit_scope else (0, ())
            briefs.append(
                ResourceBrief(
                    resource_id=self.skill_resource_id(skill),
                    resource_type="skill",
                    name=skill.name,
                    purpose=skill.description or f"Skill: {skill.name}",
                    group=skill.tool_group,
                    tool_count=tool_count,
                    tools_preview=tools_preview if has_explicit_scope else (),
                    active=skill.active and (tool_count > 0 or not has_explicit_scope),
                )
            )

        for group_name, group in sorted(self._owner.groups.items()):
            if group_name.startswith("skill_") or group_name in mcp_groups:
                continue
            tool_count, tools_preview = _tool_summary(ToolLease(include_groups=(group_name,)))
            briefs.append(
                ResourceBrief(
                    resource_id=self.group_resource_id(group_name),
                    resource_type="tool_group",
                    name=group_name,
                    purpose=group.description,
                    group=group_name,
                    tool_count=tool_count,
                    tools_preview=tools_preview,
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

    @staticmethod
    def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    @staticmethod
    def _strip_urls(text: str) -> str:
        return _URL_RE.sub(" ", str(text or ""))

    def _query_requires_network(self, query: str) -> bool:
        normalized = (query or "").strip().lower()
        if not normalized:
            return False
        return self._contains_any_marker(normalized, _NETWORK_QUERY_MARKERS)

    def _resource_looks_network_capable(
        self,
        *,
        name: str,
        description: str,
        tool_names: tuple[str, ...] = (),
    ) -> bool:
        haystack = " ".join([name, description, *tool_names]).lower()
        return self._contains_any_marker(haystack, _NETWORK_RESOURCE_MARKERS)

    def _score_text_match(
        self,
        *,
        query: str,
        query_terms: set[str],
        resource_id: str,
        name: str,
        keywords: tuple[str, ...] = (),
        phrases: tuple[str, ...] = (),
        fallback_text: tuple[str, ...] = (),
    ) -> tuple[int, list[str]]:
        normalized_query = (query or "").strip().lower()
        normalized_name = (name or "").strip().lower()
        normalized_resource_id = (resource_id or "").strip().lower()
        reasons: list[str] = []
        score = 0

        if normalized_resource_id and normalized_resource_id in normalized_query:
            score += 120
            reasons.append("explicit_resource_id")
        elif normalized_name and len(normalized_name) >= 3 and normalized_name in normalized_query:
            score += 80
            reasons.append("explicit_name")

        matched_phrases = {
            phrase.strip().lower()
            for phrase in phrases
            if str(phrase).strip() and str(phrase).strip().lower() in normalized_query
        }
        if matched_phrases:
            score += 24 * len(matched_phrases)
            reasons.append(f"phrase_matches={len(matched_phrases)}")

        candidate_terms = set()
        for keyword in keywords:
            text = str(keyword).strip().lower()
            if text:
                candidate_terms.add(text)
        tokenizer = getattr(self._owner, "_tokenize", None)
        if callable(tokenizer):
            for value in fallback_text:
                candidate_terms.update(tokenizer(str(value or "")))
        matched_terms = sorted(term for term in candidate_terms if term in query_terms)
        if matched_terms:
            score += 6 * len(matched_terms)
            reasons.append(f"term_matches={len(matched_terms)}")

        return score, reasons

    def recommend_resources(
        self,
        query: str,
        *,
        limit: int = 6,
    ) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {
                "query": "",
                "primary_resource_ids": [],
                "supporting_resource_ids": [],
                "resource_ids": [],
                "matches": [],
            }

        semantic_query = self._strip_urls(normalized_query).strip()
        tokenizer = getattr(self._owner, "_tokenize", None)
        query_terms = (
            tokenizer(semantic_query) if callable(tokenizer) else set()
        )
        require_network = self._query_requires_network(normalized_query)
        matches: list[dict[str, Any]] = []

        for skill in sorted(self._owner.skills.values(), key=lambda item: item.name.lower()):
            if not skill.active:
                continue
            resource_id = self.skill_resource_id(skill)
            score, reasons = self._score_text_match(
                query=semantic_query,
                query_terms=query_terms,
                resource_id=resource_id,
                name=skill.name,
                keywords=tuple(skill.keywords or ()),
                phrases=tuple(skill.phrases or ()),
                fallback_text=(
                    skill.name,
                    skill.description,
                    skill.prompt,
                ),
            )
            if score <= 0:
                continue
            matches.append(
                {
                    "resource_id": resource_id,
                    "resource_type": "skill",
                    "name": skill.name,
                    "score": score,
                    "role": "primary",
                    "reasons": reasons,
                }
            )

        for server_name, group_name in sorted(self._owner.mcp_server_groups.items()):
            group = self._owner.groups.get(group_name)
            description = (
                group.description if group is not None else f"MCP tools from {server_name}"
            )
            resource_id = self.mcp_resource_id(server_name)
            score, reasons = self._score_text_match(
                query=semantic_query,
                query_terms=query_terms,
                resource_id=resource_id,
                name=server_name,
                fallback_text=(server_name, description),
            )
            if score <= 0:
                continue
            matches.append(
                {
                    "resource_id": resource_id,
                    "resource_type": "mcp",
                    "name": server_name,
                    "score": score,
                    "role": "primary",
                    "reasons": reasons,
                }
            )

        for group_name, group in sorted(self._owner.groups.items()):
            if not group.active or group_name.startswith("skill_") or group_name in set(
                self._owner.mcp_server_groups.values()
            ):
                continue
            lease = ToolLease(include_groups=(group_name,))
            tool_names = tuple(
                sorted(registered.tool.name for registered in self._owner.registry.list(lease))
            )
            resource_id = self.group_resource_id(group_name)
            score, reasons = self._score_text_match(
                query=semantic_query,
                query_terms=query_terms,
                resource_id=resource_id,
                name=group_name,
                fallback_text=(group_name, group.description, *tool_names),
            )
            if require_network and self._resource_looks_network_capable(
                name=group_name,
                description=group.description,
                tool_names=tool_names,
            ):
                score += 60
                reasons.append("capability:network")
            if score <= 0:
                continue
            matches.append(
                {
                    "resource_id": resource_id,
                    "resource_type": "tool_group",
                    "name": group_name,
                    "score": score,
                    "role": "supporting",
                    "reasons": reasons,
                }
            )

        matches.sort(
            key=lambda item: (
                0 if item["role"] == "primary" else 1,
                -int(item["score"]),
                str(item["resource_id"]),
            )
        )
        primary_resource_ids = [
            item["resource_id"] for item in matches if item["role"] == "primary"
        ][:1]
        supporting_resource_ids = [
            item["resource_id"]
            for item in matches
            if item["role"] == "supporting"
        ][: max(0, limit - len(primary_resource_ids))]
        resource_ids = list(
            dict.fromkeys([*primary_resource_ids, *supporting_resource_ids])
        )[:limit]
        filtered_matches = [
            item for item in matches if item["resource_id"] in resource_ids
        ]
        return {
            "query": normalized_query,
            "primary_resource_ids": primary_resource_ids,
            "supporting_resource_ids": supporting_resource_ids,
            "resource_ids": resource_ids,
            "matches": filtered_matches,
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
            if "code" in self._owner.groups and "code" not in include_groups:
                include_groups.append("code")
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
