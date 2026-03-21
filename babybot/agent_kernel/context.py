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

    # Keys whose values are shared singletons or contain unpicklable
    # resources (e.g. sqlite3.Connection inside TapeStore/HybridMemoryStore).
    # They are passed by reference to child contexts instead of deep-copied.
    _SHARED_STATE_KEYS = frozenset((
        "tape", "tape_store", "memory_store", "upstream_results",
    ))
    _OMITTED_STATE_KEYS = frozenset(("heartbeat",))

    def fork(self, session_id: str | None = None) -> ExecutionContext:
        """Create a child context with copied state and independent event stream.

        Shared objects listed in ``_SHARED_STATE_KEYS`` are passed by reference
        so child tasks can signal the same heartbeat / read the same tape.
        Everything else is deep-copied for isolation.
        """
        # Separate shared refs from copyable state to avoid deepcopy failures
        # on unpicklable objects (e.g. sqlite3.Connection in TapeStore).
        shared: dict[str, Any] = {}
        copyable: dict[str, Any] = {}
        for k, v in self._context.state.items():
            if k in self._SHARED_STATE_KEYS:
                shared[k] = v
            elif k in self._OMITTED_STATE_KEYS:
                continue
            else:
                copyable[k] = v
        new_state = copy.deepcopy(copyable)
        new_state.update(shared)
        return ExecutionContext(
            session_id=session_id or self._context.session_id,
            state=new_state,
            events=[],
        )
