"""Shared helpers for tool-lease composition and normalization."""

from __future__ import annotations

from typing import Any, Iterable

from .types import ToolLease


def lease_to_dict(lease: ToolLease) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if lease.include_groups:
        payload["include_groups"] = list(lease.include_groups)
    if lease.include_tools:
        payload["include_tools"] = list(lease.include_tools)
    if lease.exclude_tools:
        payload["exclude_tools"] = list(lease.exclude_tools)
    return payload


def merge_tool_leases(*leases: ToolLease) -> ToolLease:
    """Merge leases with additive include semantics and deny-wins excludes."""
    include_groups: set[str] = set()
    include_tools: set[str] = set()
    exclude_tools: set[str] = set()
    for lease in leases:
        include_groups.update(lease.include_groups)
        include_tools.update(lease.include_tools)
        exclude_tools.update(lease.exclude_tools)
    return ToolLease(
        include_groups=tuple(sorted(include_groups)),
        include_tools=tuple(sorted(include_tools)),
        exclude_tools=tuple(sorted(exclude_tools)),
    )


def filter_tool_lease(
    lease: ToolLease,
    *,
    drop_groups: Iterable[str] = (),
    drop_group_prefixes: Iterable[str] = (),
    drop_tools: Iterable[str] = (),
    extra_exclude_tools: Iterable[str] = (),
) -> ToolLease:
    """Return a filtered lease while preserving explicit deny semantics."""
    forbidden_groups = {str(item) for item in drop_groups if str(item)}
    forbidden_prefixes = tuple(
        str(item) for item in drop_group_prefixes if str(item)
    )
    forbidden_tools = {str(item) for item in drop_tools if str(item)}
    merged_excludes = set(lease.exclude_tools)
    merged_excludes.update(str(item) for item in extra_exclude_tools if str(item))
    return ToolLease(
        include_groups=tuple(
            group
            for group in lease.include_groups
            if group not in forbidden_groups
            and not any(group.startswith(prefix) for prefix in forbidden_prefixes)
        ),
        include_tools=tuple(
            tool for tool in lease.include_tools if tool not in forbidden_tools
        ),
        exclude_tools=tuple(sorted(merged_excludes | forbidden_tools)),
    )
