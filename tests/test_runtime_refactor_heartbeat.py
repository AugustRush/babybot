from __future__ import annotations

from babybot.agent_kernel import ContextManager, ExecutionContext
from babybot.heartbeat import Heartbeat


def test_context_fork_shares_heartbeat_object_today() -> None:
    heartbeat = Heartbeat(idle_timeout=30)
    manager = ContextManager(
        ExecutionContext(session_id="root", state={"heartbeat": heartbeat})
    )

    child = manager.fork(session_id="child")

    assert child.state["heartbeat"] is heartbeat
