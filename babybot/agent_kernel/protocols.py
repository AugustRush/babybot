"""Ports for the orchestration kernel."""

from __future__ import annotations

from typing import Protocol

from .types import ExecutionContext, TaskContract, TaskResult


class ExecutorPort(Protocol):
    """Executes one task with a provided context and lease."""

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        ...
