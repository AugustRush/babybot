from __future__ import annotations

import asyncio

import pytest

from babybot.interactive_sessions.types import InteractiveReply


class FakeBackend:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.send_calls = 0

    async def start(self, chat_key: str):
        self.start_calls += 1
        return {"chat_key": chat_key, "backend": "claude", "ordinal": self.start_calls}

    async def send(self, handle, message: str) -> InteractiveReply:
        self.send_calls += 1
        return InteractiveReply(text=f"{handle['chat_key']}:{message}")

    async def stop(self, handle, reason: str = "user_stop") -> None:
        del handle, reason
        self.stop_calls += 1

    def status(self, handle) -> dict[str, str]:
        return {"backend": handle["backend"]}


@pytest.mark.asyncio
async def test_manager_reuses_existing_chat_session():
    from babybot.interactive_sessions.manager import InteractiveSessionManager

    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=7200,
    )

    first = await manager.start(chat_key="feishu:c1", backend_name="claude")
    second = await manager.start(chat_key="feishu:c1", backend_name="claude")

    assert first.session_id == second.session_id
    assert backend.start_calls == 1


@pytest.mark.asyncio
async def test_manager_stops_expired_session_before_send():
    from babybot.interactive_sessions.manager import InteractiveSessionManager

    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=1,
        time_fn=lambda: 100.0,
    )
    await manager.start(chat_key="feishu:c1", backend_name="claude")
    manager._time_fn = lambda: 102.0

    reply = await manager.send("feishu:c1", "hello")

    assert "超时关闭" in reply.text
    assert backend.stop_calls == 1


def test_manager_cleans_up_lock_after_stop():
    from babybot.interactive_sessions.manager import InteractiveSessionManager

    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=7200,
    )

    async def _run() -> tuple[bool, bool, bool]:
        await manager.start(chat_key="feishu:c1", backend_name="claude")
        has_lock_before = "feishu:c1" in manager._locks
        stopped = await manager.stop("feishu:c1")
        has_lock_after = "feishu:c1" in manager._locks
        return has_lock_before, stopped, has_lock_after

    has_lock_before, stopped, has_lock_after = asyncio.run(_run())

    assert has_lock_before is True
    assert stopped is True
    assert has_lock_after is False


def test_manager_has_active_session_ignores_expired_sessions():
    from babybot.interactive_sessions.manager import InteractiveSessionManager

    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=1,
        time_fn=lambda: 100.0,
    )

    async def _run() -> bool:
        await manager.start(chat_key="feishu:c1", backend_name="claude")
        manager._time_fn = lambda: 102.0
        return manager.has_active_session("feishu:c1")

    active = asyncio.run(_run())

    assert active is False


def test_manager_status_prunes_expired_session():
    from babybot.interactive_sessions.manager import InteractiveSessionManager

    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=1,
        time_fn=lambda: 100.0,
    )

    async def _run():
        await manager.start(chat_key="feishu:c1", backend_name="claude")
        manager._time_fn = lambda: 102.0
        return manager.status("feishu:c1"), manager.summary()

    status, summary = asyncio.run(_run())

    assert status is None
    assert summary["active_count"] == 0


def test_manager_cleanup_prunes_all_expired_sessions():
    from babybot.interactive_sessions.manager import InteractiveSessionManager

    backend = FakeBackend()
    manager = InteractiveSessionManager(
        backends={"claude": backend},
        max_age_seconds=1,
        time_fn=lambda: 100.0,
    )

    async def _run() -> tuple[int, dict[str, object]]:
        await manager.start(chat_key="feishu:c1", backend_name="claude")
        await manager.start(chat_key="feishu:c2", backend_name="claude")
        manager._time_fn = lambda: 102.0
        pruned = manager.cleanup()
        return pruned, manager.summary()

    pruned, summary = asyncio.run(_run())

    assert pruned == 2
    assert summary["active_count"] == 0
