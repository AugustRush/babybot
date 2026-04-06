"""Shared child-task event/view primitives for orchestration runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .plan_notebook import PlanNotebook


@dataclass(frozen=True)
class ChildTaskEvent:
    """Lifecycle event emitted for one child task."""

    flow_id: str
    task_id: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildTaskView:
    resource_ids: tuple[str, ...]
    primary_resource_id: str
    execution_description: str
    public_description: str
    user_label: str
    deps: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()

    def metadata_payload(self, *, notebook_node_id: str = "") -> dict[str, Any]:
        return {
            "resource_id": self.primary_resource_id,
            "resource_ids": list(self.resource_ids),
            "skill_ids": list(self.skill_ids),
            "description": self.public_description,
            "execution_description": self.execution_description,
            "user_label": self.user_label,
            "notebook_node_id": notebook_node_id,
        }

    def state_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "resource_id": self.primary_resource_id,
            "resource_ids": list(self.resource_ids),
            "description": self.public_description,
            "execution_description": self.execution_description,
            "user_label": self.user_label,
            "deps": list(self.deps),
            **extra,
        }

    def event_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "resource_id": self.primary_resource_id,
            "resource_ids": list(self.resource_ids),
            "description": self.public_description,
            "user_label": self.user_label,
            "deps": list(self.deps),
            **extra,
        }


class InMemoryChildTaskBus:
    """Simple in-memory event sink for child-task lifecycle events."""

    def __init__(self) -> None:
        self._events: dict[str, list[ChildTaskEvent]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[ChildTaskEvent]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event: ChildTaskEvent) -> None:
        async with self._lock:
            self._events.setdefault(event.flow_id, []).append(event)
            for queue in list(self._subscribers.get(event.flow_id, ())):
                queue.put_nowait(event)

    def events_for(self, flow_id: str) -> list[ChildTaskEvent]:
        return list(self._events.get(flow_id, ()))

    async def subscribe(self, flow_id: str) -> AsyncIterator[ChildTaskEvent]:
        queue: asyncio.Queue[ChildTaskEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(flow_id, []).append(queue)
            for event in self._events.get(flow_id, ()):
                queue.put_nowait(event)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(flow_id, [])
                if queue in subscribers:
                    subscribers.remove(queue)
                if not subscribers:
                    self._subscribers.pop(flow_id, None)

    def clear_flow(self, flow_id: str) -> None:
        self._events.pop(flow_id, None)
        self._subscribers.pop(flow_id, None)


def notebook_feedback_payload(
    notebook: PlanNotebook | Any,
    notebook_node_id: str,
) -> dict[str, Any]:
    if not isinstance(notebook, PlanNotebook):
        return {}
    if not notebook_node_id or notebook_node_id not in notebook.nodes:
        return {}
    node = notebook.get_node(notebook_node_id)
    completed_steps = [
        candidate.title
        for candidate in notebook.nodes.values()
        if candidate.parent_id == node.parent_id and candidate.status == "completed"
    ][-3:]
    blockers = [
        checkpoint.message for checkpoint in node.checkpoints if checkpoint.status == "open"
    ]
    blockers.extend(issue.title for issue in node.issues if issue.status == "open")
    return {
        "notebook_phase": node.status,
        "notebook_owner": node.owner,
        "notebook_completed_steps": completed_steps,
        "notebook_blockers": blockers[:2],
        "notebook_next_action": (
            node.objective if node.status not in {"completed", "failed", "cancelled"} else ""
        ),
    }
