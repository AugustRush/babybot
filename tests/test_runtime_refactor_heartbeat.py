from __future__ import annotations

from babybot.agent_kernel import ContextManager, ExecutionContext
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
