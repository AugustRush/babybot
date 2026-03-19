from __future__ import annotations

import asyncio
import json

from babybot.agent_kernel import ContextManager, ExecutionContext, TaskResult
from babybot.agent_kernel.dynamic_orchestrator import InMemoryChildTaskBus, InProcessChildTaskRuntime
from babybot.heartbeat import Heartbeat, TaskHeartbeatRegistry


def test_context_fork_does_not_share_parent_heartbeat() -> None:
    heartbeat = Heartbeat(idle_timeout=30)
    manager = ContextManager(
        ExecutionContext(session_id="root", state={"heartbeat": heartbeat})
    )

    child = manager.fork(session_id="child")

    assert "heartbeat" not in child.state


def test_task_heartbeat_registry_tracks_tasks_independently() -> None:
    parent = Heartbeat(idle_timeout=30)
    registry = TaskHeartbeatRegistry()

    task_a = registry.handle("flow-1", "task-a", parent=parent)
    task_b = registry.handle("flow-1", "task-b")
    task_a.beat(progress=0.25)
    task_b.beat(progress=0.75)

    snapshot = registry.snapshot("flow-1")

    assert set(snapshot) == {"task-a", "task-b"}
    assert snapshot["task-a"]["progress"] == 0.25
    assert snapshot["task-b"]["progress"] == 0.75


def test_task_heartbeat_registry_can_identify_stale_tasks_independently() -> None:
    registry = TaskHeartbeatRegistry()

    task_a = registry.handle("flow-1", "task-a")
    task_b = registry.handle("flow-1", "task-b")
    task_b.beat()
    asyncio.run(asyncio.sleep(0.03))
    task_b.beat()

    stale = registry.stale_tasks("flow-1", stale_after_s=0.02)

    assert set(stale) == {"task-a"}
    assert stale["task-a"]["status"] == "idle"


class _SelectiveHeartbeatBridge:
    async def execute(self, task, context):  # type: ignore[no-untyped-def]
        heartbeat = context.state["heartbeat"]
        if "healthy" in task.description:
            for _ in range(3):
                heartbeat.beat(status="running")
                await asyncio.sleep(0.01)
            return TaskResult(task_id=task.task_id, status="succeeded", output="ok")

        await asyncio.sleep(0.08)
        return TaskResult(task_id=task.task_id, status="succeeded", output="late")


class _DummyResourceManager:
    def resolve_resource_scope(self, resource_id: str, require_tools: bool = False):
        del require_tools
        if resource_id == "skill.weather":
            return {"include_groups": ["skill_weather"]}, ("weather",)
        return None


def test_wait_for_tasks_uses_per_task_heartbeat_to_surface_stalled_child() -> None:
    registry = TaskHeartbeatRegistry()
    runtime = InProcessChildTaskRuntime(
        flow_id="flow-1",
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        bridge=_SelectiveHeartbeatBridge(),  # type: ignore[arg-type]
        child_task_bus=InMemoryChildTaskBus(),
        task_heartbeat_registry=registry,
        max_parallel=2,
        max_tasks=5,
        stale_after_s=0.02,
    )

    async def _run() -> tuple[str, str]:
        stalled_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "stalled child"},
            task_counter=0,
            context=ExecutionContext(session_id="flow-1"),
        )
        healthy_id = await runtime.dispatch(
            {"resource_id": "skill.weather", "description": "healthy child"},
            task_counter=1,
            context=ExecutionContext(session_id="flow-1"),
        )
        return stalled_id, await runtime.wait_for_tasks([stalled_id, healthy_id])

    stalled_id, payload = asyncio.run(_run())
    results = json.loads(payload)

    assert results[stalled_id] == "recoverable: child task heartbeat stalled"
