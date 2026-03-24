"""Pluggable executor registry -- routes tasks to backend-specific executors."""

from __future__ import annotations

import logging

from ..protocols import ExecutorPort
from ..types import ExecutionContext, TaskContract, TaskResult

logger = logging.getLogger(__name__)

__all__ = ["ExecutorRegistry"]


class ExecutorRegistry:
    """Routes task execution to backend-specific ExecutorPort implementations.

    Each task's ``metadata["backend"]`` selects which executor handles it.
    Tasks without a backend (or with an unregistered backend) go to *default*.
    """

    def __init__(self, default: ExecutorPort) -> None:
        self._default = default
        self._backends: dict[str, ExecutorPort] = {}

    def register(self, backend: str, executor: ExecutorPort) -> None:
        self._backends[backend] = executor

    def list_backends(self) -> list[str]:
        return list(self._backends.keys())

    async def execute(
        self, task: TaskContract, context: ExecutionContext
    ) -> TaskResult:
        backend = task.metadata.get("backend", "")
        executor = self._backends.get(backend, self._default)
        if backend and backend not in self._backends:
            logger.debug(
                "Unknown backend %r for task %s, using default", backend, task.task_id
            )
        return await executor.execute(task, context)
