"""Ports for the orchestration kernel."""

from __future__ import annotations

from typing import Protocol

from .types import ExecutionContext, ExecutionPlan, FinalResult, TaskContract, TaskResult


class PlannerPort(Protocol):
    """Builds a task DAG for a user goal."""

    async def plan(self, goal: str, context: ExecutionContext) -> ExecutionPlan:
        ...


class ExecutorPort(Protocol):
    """Executes one task with a provided context and lease."""

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        ...


class SynthesizerPort(Protocol):
    """Builds user-facing final output from full execution artifacts."""

    async def synthesize(
        self,
        goal: str,
        plan: ExecutionPlan,
        results: dict[str, TaskResult],
        context: ExecutionContext,
    ) -> FinalResult:
        ...
