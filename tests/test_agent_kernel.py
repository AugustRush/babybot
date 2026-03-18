from __future__ import annotations

import asyncio

import pytest

from babybot.agent_kernel import (
    ExecutionContext,
    ExecutionPlan,
    FinalResult,
    RunPolicy,
    TaskContract,
    TaskResult,
    WorkflowEngine,
)
from babybot.agent_kernel.engine import PlanValidationError


class DummyPlanner:
    def __init__(self, plan: ExecutionPlan):
        self._plan = plan

    async def plan(self, goal: str, context: ExecutionContext) -> ExecutionPlan:
        context.emit("planner.called", goal=goal)
        return self._plan


class DummyExecutor:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.order: list[str] = []

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        self.calls[task.task_id] = self.calls.get(task.task_id, 0) + 1
        self.order.append(task.task_id)
        if task.task_id == "t2" and self.calls["t2"] == 1:
            return TaskResult(task_id="t2", status="failed", error="transient")
        await asyncio.sleep(0.01)
        return TaskResult(task_id=task.task_id, status="succeeded", output=task.description)


class DummySynthesizer:
    async def synthesize(
        self,
        goal: str,
        plan: ExecutionPlan,
        results: dict[str, TaskResult],
        context: ExecutionContext,
    ) -> FinalResult:
        succeeded = [task_id for task_id, result in results.items() if result.status == "succeeded"]
        failed = [task_id for task_id, result in results.items() if result.status != "succeeded"]
        return FinalResult(
            conclusion=f"{goal}: {len(succeeded)} succeeded",
            evidence=succeeded,
            failed_tasks=failed,
            task_results=results,
        )


def test_workflow_engine_executes_dag_and_retries() -> None:
    plan = ExecutionPlan(
        tasks=(
            TaskContract(task_id="t1", description="step-1"),
            TaskContract(task_id="t2", description="step-2", retries=1),
            TaskContract(task_id="t3", description="step-3", deps=("t1", "t2")),
        )
    )
    engine = WorkflowEngine(
        planner=DummyPlanner(plan),
        executor=DummyExecutor(),
        synthesizer=DummySynthesizer(),
    )
    result = asyncio.run(engine.run("demo-goal"))

    assert result.conclusion == "demo-goal: 3 succeeded"
    assert result.failed_tasks == []
    assert result.task_results["t2"].attempts == 2
    assert result.task_results["t3"].status == "succeeded"


def test_workflow_engine_blocks_on_dependency_failure() -> None:
    class FailExecutor(DummyExecutor):
        async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
            if task.task_id == "t1":
                return TaskResult(task_id="t1", status="failed", error="boom")
            return await super().execute(task, context)

    plan = ExecutionPlan(
        tasks=(
            TaskContract(task_id="t1", description="root"),
            TaskContract(task_id="t2", description="child", deps=("t1",)),
        )
    )
    engine = WorkflowEngine(
        planner=DummyPlanner(plan),
        executor=FailExecutor(),
        synthesizer=DummySynthesizer(),
    )
    result = asyncio.run(engine.run("demo-goal"))

    assert result.task_results["t1"].status == "failed"
    assert result.task_results["t2"].status == "blocked"
    assert "t2" in result.failed_tasks


def test_plan_validation_rejects_cycles() -> None:
    cyclic_plan = ExecutionPlan(
        tasks=(
            TaskContract(task_id="t1", description="a", deps=("t2",)),
            TaskContract(task_id="t2", description="b", deps=("t1",)),
        )
    )
    engine = WorkflowEngine(
        planner=DummyPlanner(cyclic_plan),
        executor=DummyExecutor(),
        synthesizer=DummySynthesizer(),
    )
    with pytest.raises(PlanValidationError):
        asyncio.run(engine.run("cyclic-goal"))


def test_workflow_engine_uses_forked_context_per_task() -> None:
    class MutatingExecutor(DummyExecutor):
        async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
            context.state["internal"] = task.task_id
            context.emit("executor.mutated", task_id=task.task_id)
            return TaskResult(task_id=task.task_id, status="succeeded", output="ok")

    plan = ExecutionPlan(tasks=(TaskContract(task_id="t1", description="root"),))
    root_ctx = ExecutionContext(session_id="root", state={"keep": 1})
    engine = WorkflowEngine(
        planner=DummyPlanner(plan),
        executor=MutatingExecutor(),
        synthesizer=DummySynthesizer(),
    )

    result = asyncio.run(engine.run("goal", context=root_ctx))

    assert result.task_results["t1"].status == "succeeded"
    assert root_ctx.state == {"keep": 1}
    assert any(event["event"] == "task.child_event" for event in root_ctx.events)


def test_task_retry_zero_overrides_policy_default_retries() -> None:
    class AlwaysFailExecutor(DummyExecutor):
        async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
            self.calls[task.task_id] = self.calls.get(task.task_id, 0) + 1
            return TaskResult(task_id=task.task_id, status="failed", error="boom")

    plan = ExecutionPlan(
        tasks=(TaskContract(task_id="t1", description="no retry", retries=0),)
    )
    executor = AlwaysFailExecutor()
    engine = WorkflowEngine(
        planner=DummyPlanner(plan),
        executor=executor,
        synthesizer=DummySynthesizer(),
        policy=RunPolicy(default_retries=3),
    )

    result = asyncio.run(engine.run("goal"))
    assert result.task_results["t1"].status == "failed"
    assert result.task_results["t1"].attempts == 1
    assert executor.calls["t1"] == 1


def test_workflow_engine_does_not_retry_non_retryable_failures() -> None:
    class FatalExecutor(DummyExecutor):
        async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
            self.calls[task.task_id] = self.calls.get(task.task_id, 0) + 1
            return TaskResult(task_id=task.task_id, status="failed", error="forbidden: invalid api key")

    plan = ExecutionPlan(
        tasks=(TaskContract(task_id="t1", description="fatal", retries=3),)
    )
    executor = FatalExecutor()
    engine = WorkflowEngine(
        planner=DummyPlanner(plan),
        executor=executor,
        synthesizer=DummySynthesizer(),
    )

    result = asyncio.run(engine.run("goal"))
    assert result.task_results["t1"].status == "failed"
    assert result.task_results["t1"].attempts == 1
    assert executor.calls["t1"] == 1
