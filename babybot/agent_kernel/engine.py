"""Planner-Executor-Synthesizer orchestration engine."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from .context import ContextManager
from .errors import classify_error, retry_delay_seconds
from .protocols import ExecutorPort, PlannerPort, SynthesizerPort
from .types import (
    ExecutionContext,
    ExecutionPlan,
    FinalResult,
    RunPolicy,
    TaskContract,
    TaskResult,
)


FAILED_STATES = {"failed", "blocked", "skipped"}


class PlanValidationError(ValueError):
    """Raised when a planner output is invalid."""


@dataclass
class WorkflowEngine:
    """Single built-in orchestration mode:

    1) planner builds a task graph
    2) executor pool runs tasks with dependency-aware concurrency
    3) synthesizer builds final output
    """

    planner: PlannerPort
    executor: ExecutorPort
    synthesizer: SynthesizerPort
    policy: RunPolicy = field(default_factory=RunPolicy)

    async def run(self, goal: str, context: ExecutionContext | None = None) -> FinalResult:
        ctx = context or ExecutionContext()
        plan = await self.planner.plan(goal, ctx)
        self._validate_plan(plan)
        results = await self._execute_plan(plan, ctx)
        return await self._safely_synthesize(goal, plan, results, ctx)

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext,
    ) -> dict[str, TaskResult]:
        task_map = {task.task_id: task for task in plan.tasks}
        pending = set(task_map)
        results: dict[str, TaskResult] = {}
        semaphore = asyncio.Semaphore(max(1, int(self.policy.max_parallel)))

        in_flight: dict[str, asyncio.Task] = {}
        done_event = asyncio.Event()

        async def _run_one(tid: str) -> tuple[str, TaskResult]:
            async with semaphore:
                return tid, await self._run_task(task_map[tid], context)

        def _on_done(_fut: asyncio.Task) -> None:
            done_event.set()

        while pending or in_flight:
            # Mark tasks whose deps failed as blocked
            blocked = [
                task_id
                for task_id in pending
                if any(results.get(dep, None) and results[dep].status in FAILED_STATES for dep in task_map[task_id].deps)
            ]
            for task_id in blocked:
                pending.discard(task_id)
                results[task_id] = TaskResult(
                    task_id=task_id,
                    status="blocked",
                    error="Dependency failed.",
                )
                context.emit("task.blocked", task_id=task_id)

            # Find newly ready tasks (deps satisfied, not already in flight)
            newly_ready = [
                task_id
                for task_id in pending
                if task_id not in in_flight
                and all(results.get(dep, None) and results[dep].status == "succeeded" for dep in task_map[task_id].deps)
            ]

            # Deadlock: nothing ready, nothing in flight, but still pending
            if not newly_ready and not in_flight and pending:
                for task_id in list(pending):
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        status="skipped",
                        error="Task not executable due to dependency cycle or unresolved deps.",
                    )
                    context.emit("task.skipped", task_id=task_id)
                    pending.discard(task_id)
                break

            # Launch newly ready tasks immediately
            for task_id in newly_ready:
                task = asyncio.create_task(_run_one(task_id))
                task.add_done_callback(_on_done)
                in_flight[task_id] = task

            # Wait for at least one task to complete
            if in_flight and not newly_ready:
                done_event.clear()
                await done_event.wait()

            # Harvest completed tasks
            for task_id in list(in_flight):
                if in_flight[task_id].done():
                    tid, task_result = in_flight[task_id].result()
                    results[tid] = task_result
                    pending.discard(tid)
                    del in_flight[task_id]

        return results

    async def _run_task(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        timeout_s = (
            task.timeout_s if task.timeout_s is not None else self.policy.default_timeout_s
        )
        retries = (
            int(task.retries)
            if task.retries is not None
            else int(self.policy.default_retries)
        )
        attempts = max(0, retries) + 1

        context.emit("task.running", task_id=task.task_id)
        last_error = ""
        last_error_type = "retryable"
        suggested_action = ""
        for attempt in range(1, attempts + 1):
            child_context = ContextManager(context).fork(
                session_id=f"{context.session_id}:{task.task_id}"
                if context.session_id
                else task.task_id
            )
            try:
                coroutine = self.executor.execute(task, child_context)
                result = (
                    await asyncio.wait_for(coroutine, timeout=float(timeout_s))
                    if timeout_s and timeout_s > 0
                    else await coroutine
                )
                self._merge_child_events(
                    parent=context,
                    child=child_context,
                    task_id=task.task_id,
                    attempt=attempt,
                )
                result.attempts = attempt
                if result.status == "succeeded":
                    # Merge upstream_results from child back to parent
                    child_upstream = child_context.state.get("upstream_results", {})
                    if child_upstream:
                        parent_upstream = context.state.setdefault("upstream_results", {})
                        parent_upstream.update(child_upstream)
                    # Merge collected media paths
                    child_media = child_context.state.get("media_paths_collected", [])
                    if child_media:
                        context.state.setdefault("media_paths_collected", []).extend(child_media)
                    context.emit("task.succeeded", task_id=task.task_id, attempt=attempt)
                    return result
                last_error = result.error or f"Task ended with status={result.status}"
                decision = classify_error(last_error)
                last_error_type = decision.error_type
                suggested_action = decision.suggested_action
                result.metadata.setdefault("error_type", decision.error_type)
                result.metadata.setdefault("suggested_action", decision.suggested_action)
                if not decision.retryable:
                    context.emit(
                        "task.failed",
                        task_id=task.task_id,
                        error=last_error,
                        error_type=decision.error_type,
                    )
                    return TaskResult(
                        task_id=task.task_id,
                        status="failed",
                        error=last_error,
                        attempts=attempt,
                        metadata=result.metadata,
                    )
            except asyncio.TimeoutError:
                self._merge_child_events(
                    parent=context,
                    child=child_context,
                    task_id=task.task_id,
                    attempt=attempt,
                )
                last_error = f"Task timed out after {timeout_s}s."
                decision = classify_error(last_error)
                last_error_type = decision.error_type
                suggested_action = decision.suggested_action
            except Exception as exc:  # pragma: no cover - defensive boundary
                self._merge_child_events(
                    parent=context,
                    child=child_context,
                    task_id=task.task_id,
                    attempt=attempt,
                )
                last_error = str(exc)
                decision = classify_error(last_error)
                last_error_type = decision.error_type
                suggested_action = decision.suggested_action
                if not decision.retryable:
                    break

            if attempt < attempts:
                decision = classify_error(last_error)
                if not decision.retryable:
                    break
                context.emit(
                    "task.retrying",
                    task_id=task.task_id,
                    attempt=attempt,
                    error=last_error,
                )
                await asyncio.sleep(retry_delay_seconds(attempt - 1))

        context.emit("task.failed", task_id=task.task_id, error=last_error)
        return TaskResult(
            task_id=task.task_id,
            status="failed",
            error=last_error,
            attempts=min(attempts, attempt),
            metadata={
                "error_type": last_error_type,
                "suggested_action": suggested_action,
            },
        )

    @staticmethod
    def _merge_child_events(
        parent: ExecutionContext,
        child: ExecutionContext,
        task_id: str,
        attempt: int,
    ) -> None:
        for event in child.events:
            parent.events.append(
                {
                    "event": "task.child_event",
                    "task_id": task_id,
                    "attempt": attempt,
                    "child": event,
                }
            )

    async def _safely_synthesize(
        self,
        goal: str,
        plan: ExecutionPlan,
        results: dict[str, TaskResult],
        context: ExecutionContext,
    ) -> FinalResult:
        try:
            final = await self.synthesizer.synthesize(goal, plan, results, context)
            if not final.task_results:
                final.task_results = dict(results)
            return final
        except Exception as exc:  # pragma: no cover - defensive fallback
            failed = [task_id for task_id, res in results.items() if res.status != "succeeded"]
            return FinalResult(
                conclusion=f"Synthesis failed: {exc}",
                failed_tasks=failed,
                task_results=dict(results),
            )

    def _validate_plan(self, plan: ExecutionPlan) -> None:
        if not plan.tasks:
            raise PlanValidationError("Execution plan is empty.")

        task_map: dict[str, TaskContract] = {}
        for task in plan.tasks:
            task_id = task.task_id.strip()
            if not task_id:
                raise PlanValidationError("Task ID cannot be empty.")
            if task_id in task_map:
                raise PlanValidationError(f"Duplicate task_id: {task_id}")
            task_map[task_id] = task

        for task in plan.tasks:
            for dep in task.deps:
                if dep not in task_map:
                    raise PlanValidationError(
                        f"Task '{task.task_id}' depends on unknown task '{dep}'."
                    )

        if self._contains_cycle(task_map):
            raise PlanValidationError("Execution plan has dependency cycle.")

    @staticmethod
    def _contains_cycle(task_map: dict[str, TaskContract]) -> bool:
        indegree = {task_id: 0 for task_id in task_map}
        children: dict[str, list[str]] = {task_id: [] for task_id in task_map}

        for task_id, task in task_map.items():
            for dep in task.deps:
                indegree[task_id] += 1
                children[dep].append(task_id)

        queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        return visited != len(task_map)
