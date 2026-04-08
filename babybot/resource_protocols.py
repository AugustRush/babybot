"""Protocol interfaces for ResourceManager helper classes.

Each Protocol defines the minimum interface that a helper class requires from
its owner (ResourceManager), replacing the previous ``owner: Any`` annotations
with narrow, statically-checkable contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "WorkspaceHost",
    "SkillHost",
    "CatalogHost",
    "SubagentHost",
    "AdminHost",
]


class WorkspaceHost(Protocol):
    """Minimum interface required by WorkspaceToolSuite."""

    def _get_active_write_root(self) -> Path: ...
    def _get_user_python(self) -> str: ...
    def _clean_env(self) -> dict[str, str]: ...
    def _coerce_timeout(self, value: Any, default: float = 300.0) -> float: ...
    def _resolve_workspace_file(self, path: str) -> tuple[str, str | None]: ...


class SkillHost(Protocol):
    """Minimum interface required by SkillLoader and ResourceToolLoader."""

    config: Any
    skills: dict[str, Any]
    groups: dict[str, Any]
    _skill_load_errors: list[dict]
    registry: Any

    def _callable_tool_cls(self) -> type: ...
    def _build_skill_runtime(self, meta: dict) -> Any: ...
    def _upsert_skill(self, skill: Any) -> None: ...
    def _build_external_skill_callable(self, **kw: Any) -> Any: ...
    def _build_external_cli_script_callable(self, **kw: Any) -> Any: ...


class CatalogHost(Protocol):
    """Minimum interface required by ResourceScopeHelper."""

    registry: Any
    groups: dict[str, Any]
    skills: dict[str, Any]
    mcp_server_groups: dict[str, str]
    mcp_clients: dict[str, Any]
    _orchestration_tools: frozenset[str]

    def _tokenize(self, text: str) -> set[str]: ...


class SubagentHost(Protocol):
    """Minimum interface required by ResourceSubagentRuntime."""

    config: Any
    registry: Any

    def _get_output_dir(self) -> Path: ...
    def _override_current_write_root(self, root: Path) -> Any: ...
    def _build_task_lease(self, raw: dict) -> Any: ...
    def _select_skill_packs(self, desc: str, **kw: Any) -> list: ...
    def _build_worker_sys_prompt(self, **kw: Any) -> str: ...
    def _create_worker_executor(self, **kw: Any) -> Any: ...
    def _get_shared_gateway(self) -> Any: ...
    def _extract_media_from_text(self, text: str) -> list[str]: ...
    def _get_current_task_lease_var(self) -> Any: ...
    def _get_current_skill_ids_var(self) -> Any: ...
    def _get_current_worker_depth_var(self) -> Any: ...


class AdminHost(Protocol):
    """Minimum interface required by ResourceAdminHelper."""

    config: Any
    skills: dict[str, Any]
    groups: dict[str, Any]
    registry: Any
    _skill_load_errors: list[dict]
    memory_store: Any
    _observability_provider: Any

    def _skill_resource_id(self, skill: Any) -> str: ...
    def _get_current_tool_context_var(self) -> Any: ...
    def _skill_loader_view(self) -> Any: ...
    def _build_skill_runtime(self, meta: dict) -> Any: ...
    def _register_skill_tools(self, name: str, dir: Any, **kw: Any) -> tuple: ...
    def _upsert_skill(self, skill: Any) -> None: ...
