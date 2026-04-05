"""Typed helpers for accessing shared execution state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .plan_notebook import PlanNotebook
from .types import ExecutionContext


@dataclass(frozen=True)
class NotebookBinding:
    notebook: PlanNotebook | None
    node_id: str = ""

    @property
    def active(self) -> bool:
        return self.notebook is not None and bool(self.node_id)


class RuntimeState:
    """Thin accessor over ExecutionContext.state."""

    def __init__(self, context: ExecutionContext) -> None:
        self._context = context

    @property
    def raw(self) -> dict[str, Any]:
        return self._context.state

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self.raw.setdefault(key, default)

    def media_paths(self) -> tuple[str, ...]:
        raw = self.raw.get("media_paths") or ()
        return tuple(str(item) for item in raw if str(item))

    def collected_media_bucket(self) -> list[str]:
        bucket = self.raw.setdefault("media_paths_collected", [])
        return bucket if isinstance(bucket, list) else []

    def pending_runtime_hints(self) -> list[Any]:
        hints = self.raw.setdefault("pending_runtime_hints", [])
        return hints if isinstance(hints, list) else []

    def append_runtime_hint(self, text: Any) -> None:
        rendered = str(text or "").strip()
        if rendered:
            self.pending_runtime_hints().append(rendered)

    def notebook_binding(self, preferred_node_id: str = "") -> NotebookBinding:
        notebook = self.raw.get("plan_notebook")
        if not isinstance(notebook, PlanNotebook):
            return NotebookBinding(None, "")
        node_id = str(preferred_node_id or "").strip()
        if not node_id:
            node_id = str(self.raw.get("current_notebook_node_id", "") or "").strip()
        if (not node_id or node_id not in notebook.nodes) and notebook.root_node_id:
            node_id = str(notebook.root_node_id).strip()
        if not node_id or node_id not in notebook.nodes:
            return NotebookBinding(None, "")
        return NotebookBinding(notebook, node_id)

    def notebook_context_budget(self, default: int = 2400) -> int:
        try:
            value = int(self.raw.get("notebook_context_budget", default) or default)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def mapping_bucket(self, key: str) -> dict[str, Any]:
        bucket = self.raw.setdefault(key, {})
        return bucket if isinstance(bucket, dict) else {}

    def upstream_results_bucket(self) -> dict[str, Any]:
        return self.mapping_bucket("upstream_results")

    def notebook_task_map(self) -> dict[str, Any]:
        return self.mapping_bucket("notebook_task_map")

    def extend_collected_media(self, paths: Iterable[Any]) -> None:
        bucket = self.collected_media_bucket()
        bucket.extend(str(path) for path in paths if str(path).strip())

    def policy_hints(self) -> tuple[str, ...]:
        raw = self.raw.get("policy_hints") or ()
        return tuple(str(item).strip() for item in raw if str(item).strip())

    def set_policy_hints(self, hints: list[str] | tuple[str, ...]) -> None:
        self.raw["policy_hints"] = tuple(
            dict.fromkeys(str(item).strip() for item in hints if str(item).strip())
        )
