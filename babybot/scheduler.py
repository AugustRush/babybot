"""Task scheduler with serial/parallel/hybrid execution modes.

.. deprecated::
    This module is superseded by :mod:`babybot.agent_kernel.dynamic_orchestrator`
    which provides dynamic orchestration with tool-calling loop.
    Kept for backward compatibility; prefer ``DynamicOrchestrator`` for new code.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from .heartbeat import Heartbeat


TaskStatus = Literal["pending", "running", "succeeded", "failed", "blocked", "skipped"]
ScheduleMode = Literal["serial", "parallel", "hybrid"]


@dataclass
class TaskSpec:
    """A scheduled subtask with dependencies and optional lease config."""

    task_id: str
    description: str
    deps: list[str] = field(default_factory=list)
    lease: dict[str, Any] = field(default_factory=dict)
    timeout: int | None = None
    retries: int = 0


@dataclass
class TaskResult:
    """Execution result for one task."""

    task_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""


class Scheduler:
    """A lightweight scheduler for DAG-like task execution."""

    def __init__(self, max_parallel: int = 4):
        self.max_parallel = max_parallel
        self.status: dict[str, TaskStatus] = {}
        self.results: dict[str, TaskResult] = {}
        self.events: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset runtime status and events."""
        self.status.clear()
        self.results.clear()
        self.events.clear()

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status snapshot."""
        return {
            "status": dict(self.status),
            "results": {
                task_id: {
                    "status": res.status,
                    "output": res.output,
                    "error": res.error,
                }
                for task_id, res in self.results.items()
            },
            "events": list(self.events),
        }

    async def run(
        self,
        tasks: list[TaskSpec],
        executor: Callable[[TaskSpec], Awaitable[str]],
        mode: ScheduleMode = "hybrid",
        heartbeat: Heartbeat | None = None,
    ) -> dict[str, TaskResult]:
        """Run tasks according to dependency constraints and scheduling mode."""
        self.reset()
        if not tasks:
            return {}

        tasks_map = {task.task_id: task for task in tasks}
        self._validate_tasks(tasks_map)
        for task in tasks:
            self.status[task.task_id] = "pending"

        if mode == "serial":
            order = self._topological_order(tasks_map)
            for task_id in order:
                await self._run_task(tasks_map[task_id], executor, heartbeat)
            return dict(self.results)

        if mode == "parallel" and any(task.deps for task in tasks):
            mode = "hybrid"

        if mode == "parallel":
            await asyncio.gather(*(self._run_task(task, executor, heartbeat) for task in tasks))
            return dict(self.results)

        await self._run_hybrid(tasks_map, executor, heartbeat)
        return dict(self.results)

    async def _run_hybrid(
        self,
        tasks_map: dict[str, TaskSpec],
        executor: Callable[[TaskSpec], Awaitable[str]],
        heartbeat: Heartbeat | None = None,
    ) -> None:
        semaphore = asyncio.Semaphore(max(1, self.max_parallel))
        pending = set(tasks_map.keys())

        while pending:
            ready = [
                task_id
                for task_id in pending
                if all(self.status.get(dep) == "succeeded" for dep in tasks_map[task_id].deps)
            ]

            blocked = [
                task_id
                for task_id in pending
                if any(self.status.get(dep) == "failed" for dep in tasks_map[task_id].deps)
            ]
            for task_id in blocked:
                self.status[task_id] = "blocked"
                self.results[task_id] = TaskResult(
                    task_id=task_id,
                    status="blocked",
                    error="Dependency failed.",
                )
                self.events.append({"task_id": task_id, "event": "blocked"})
                pending.remove(task_id)

            if not ready:
                # Remaining tasks are cyclic or waiting forever
                for task_id in list(pending):
                    self.status[task_id] = "skipped"
                    self.results[task_id] = TaskResult(
                        task_id=task_id,
                        status="skipped",
                        error="Task not executable due to dependency cycle or missing dependencies.",
                    )
                    self.events.append({"task_id": task_id, "event": "skipped"})
                    pending.remove(task_id)
                break

            async def run_with_limit(task: TaskSpec) -> None:
                async with semaphore:
                    await self._run_task(task, executor, heartbeat)

            await asyncio.gather(*(run_with_limit(tasks_map[task_id]) for task_id in ready))
            pending -= set(ready)

    async def _run_task(
        self,
        task: TaskSpec,
        executor: Callable[[TaskSpec], Awaitable[str]],
        heartbeat: Heartbeat | None = None,
    ) -> None:
        if heartbeat is not None:
            heartbeat.beat()
        self.status[task.task_id] = "running"
        self.events.append({"task_id": task.task_id, "event": "running"})
        attempts = max(0, int(task.retries)) + 1
        for attempt in range(1, attempts + 1):
            try:
                if task.timeout is not None and task.timeout > 0:
                    output = await asyncio.wait_for(
                        executor(task),
                        timeout=float(task.timeout),
                    )
                else:
                    output = await executor(task)
                self.status[task.task_id] = "succeeded"
                self.results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    status="succeeded",
                    output=output,
                )
                self.events.append({"task_id": task.task_id, "event": "succeeded"})
                if heartbeat is not None:
                    heartbeat.beat()
                return
            except Exception as e:
                if attempt < attempts:
                    self.events.append(
                        {
                            "task_id": task.task_id,
                            "event": "retrying",
                            "attempt": attempt,
                            "error": str(e),
                        }
                    )
                    continue
                self.status[task.task_id] = "failed"
                self.results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=str(e),
                )
                self.events.append(
                    {"task_id": task.task_id, "event": "failed", "error": str(e)}
                )

    def _validate_tasks(self, tasks_map: dict[str, TaskSpec]) -> None:
        for task_id, task in tasks_map.items():
            for dep in task.deps:
                if dep not in tasks_map:
                    raise ValueError(f"Task '{task_id}' depends on unknown task '{dep}'.")

    def _topological_order(self, tasks_map: dict[str, TaskSpec]) -> list[str]:
        indegree = {task_id: 0 for task_id in tasks_map}
        children: dict[str, list[str]] = {task_id: [] for task_id in tasks_map}

        for task_id, task in tasks_map.items():
            for dep in task.deps:
                indegree[task_id] += 1
                children[dep].append(task_id)

        queue = deque(task_id for task_id, deg in indegree.items() if deg == 0)
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(tasks_map):
            raise ValueError("Task graph contains a cycle.")
        return order
