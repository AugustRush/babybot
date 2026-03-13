from babybot.agent_kernel import ContextManager, ExecutionContext


def test_context_snapshot_and_restore() -> None:
    manager = ContextManager(ExecutionContext(session_id="s1"))
    manager.set("k", {"a": 1})
    manager.emit("started", step=1)

    snap = manager.snapshot()
    manager.set("k", {"a": 2})
    manager.emit("updated", step=2)

    manager.restore(snap)
    assert manager.context.session_id == "s1"
    assert manager.get("k") == {"a": 1}
    assert len(manager.context.events) == 1
    assert manager.context.events[0]["event"] == "started"


def test_context_fork_copies_state_and_isolates_events() -> None:
    manager = ContextManager(ExecutionContext(session_id="root"))
    manager.set("count", 1)
    manager.emit("root.event")

    child = manager.fork(session_id="child")
    child.state["count"] = 2
    child.emit("child.event")

    assert manager.context.session_id == "root"
    assert manager.get("count") == 1
    assert len(manager.context.events) == 1
    assert child.session_id == "child"
    assert child.state["count"] == 2
    assert len(child.events) == 1
