"""Context management primitives for the orchestration kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy

from .types import ExecutionContext


@dataclass
class ContextSnapshot:
    """Immutable snapshot for checkpoint/replay."""

    session_id: str
    state: dict[str, Any]
    events: list[dict[str, Any]]


class ContextManager:
    """Utility wrapper around ExecutionContext.

    Keeps context handling explicit and reusable without coupling business logic.
    """

    def __init__(self, context: ExecutionContext | None = None):
        self._context = context or ExecutionContext()

    @property
    def context(self) -> ExecutionContext:
        return self._context

    def set(self, key: str, value: Any) -> None:
        self._context.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.state.get(key, default)

    def merge(self, values: dict[str, Any]) -> None:
        self._context.state.update(values)

    def emit(self, event: str, **payload: Any) -> None:
        self._context.emit(event, **payload)

    def snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            session_id=self._context.session_id,
            state=copy.deepcopy(self._context.state),
            events=copy.deepcopy(self._context.events),
        )

    def restore(self, snapshot: ContextSnapshot) -> None:
        self._context.session_id = snapshot.session_id
        self._context.state = copy.deepcopy(snapshot.state)
        self._context.events = copy.deepcopy(snapshot.events)

    def fork(self, session_id: str | None = None) -> ExecutionContext:
        """Create a child context with copied state and independent event stream."""
        return ExecutionContext(
            session_id=session_id or self._context.session_id,
            state=copy.deepcopy(self._context.state),
            events=[],
        )
