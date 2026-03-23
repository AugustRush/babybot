"""Tests for babybot.cron — ScheduledTaskDef & CronScheduler."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from babybot.cron import CronScheduler, ScheduledTaskDef, ScheduledTaskManager
# ── ScheduledTaskDef.from_dict ─────────────────────────────────────────

class TestScheduledTaskDefFromDict:
    def test_cron_string(self):
        d = {
            "name": "news",
            "prompt": "get news",
            "schedule": "0 9 * * *",
            "target": {"channel": "feishu", "chat_id": "oc_123"},
        }
        td = ScheduledTaskDef.from_dict(d)
        assert td.name == "news"
        assert td.prompt == "get news"
        assert td.cron == "0 9 * * *"
        assert td.interval is None
        assert td.channel == "feishu"
        assert td.chat_id == "oc_123"
        assert td.enabled is True

    def test_interval_dict(self):
        d = {
            "name": "reminder",
            "prompt": "drink water",
            "schedule": {"interval": 3600},
            "target": {"channel": "feishu", "chat_id": "oc_456"},
        }
        td = ScheduledTaskDef.from_dict(d)
        assert td.cron is None
        assert td.interval == 3600.0

    def test_run_at_dict(self):
        d = {
            "name": "once",
            "prompt": "run once",
            "schedule": {"run_at": "2030-01-01T17:10:00+08:00"},
            "target": {"channel": "feishu", "chat_id": "oc_789"},
        }
        td = ScheduledTaskDef.from_dict(d)
        assert td.run_at is not None
        assert td.interval is None
        assert td.cron is None

    def test_run_at_time_only_string(self):
        d = {
            "name": "once",
            "prompt": "run once",
            "schedule": {"run_at": "17:10"},
            "target": {"channel": "feishu", "chat_id": "oc_789"},
        }
        td = ScheduledTaskDef.from_dict(d)
        assert td.run_at is not None
        assert "T" in td.run_at

    def test_enabled_false(self):
        d = {
            "name": "off",
            "prompt": "noop",
            "schedule": "* * * * *",
            "target": {},
            "enabled": False,
        }
        td = ScheduledTaskDef.from_dict(d)
        assert td.enabled is False

    def test_default_enabled(self):
        d = {
            "name": "x",
            "prompt": "y",
            "schedule": "0 0 * * *",
            "target": {},
        }
        td = ScheduledTaskDef.from_dict(d)
        assert td.enabled is True

    def test_invalid_schedule_missing(self):
        with pytest.raises(ValueError, match="exactly one of cron, interval or run_at"):
            ScheduledTaskDef.from_dict(
                {
                    "name": "broken",
                    "prompt": "noop",
                    "target": {},
                }
            )

    def test_invalid_interval_non_positive(self):
        with pytest.raises(ValueError, match="interval must be > 0"):
            ScheduledTaskDef.from_dict(
                {
                    "name": "broken",
                    "prompt": "noop",
                    "schedule": {"interval": 0},
                    "target": {},
                }
            )


# ── Helpers ───────────────────────────────────────────────────────────

def _make_config():
    cfg = MagicMock()
    cfg.system.idle_timeout = 10
    cfg.system.timeout = 30
    cfg._scheduled_tasks = []
    cfg.get_scheduled_tasks = lambda: list(cfg._scheduled_tasks)
    cfg.save_scheduled_tasks = lambda tasks: setattr(cfg, "_scheduled_tasks", list(tasks))
    return cfg


def _make_message_bus():
    bus = MagicMock()
    bus.enqueue = AsyncMock()
    bus.enqueue_and_wait = AsyncMock(return_value=MagicMock(text="ok", media_paths=[]))
    return bus


# ── Dynamic management ───────────────────────────────────────────────

class TestDynamicManagement:
    def test_add_and_list(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        td = ScheduledTaskDef(name="t1", prompt="p1", interval=60)
        sched.add_task(td)
        tasks = sched.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "t1"
        assert tasks[0]["enabled"] is True

    def test_remove_task(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        td = ScheduledTaskDef(name="t1", prompt="p1", interval=60)
        sched.add_task(td)
        assert sched.remove_task("t1") is True
        assert sched.list_tasks() == []

    def test_remove_nonexistent(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        assert sched.remove_task("nope") is False

    def test_update_task(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        td = ScheduledTaskDef(name="t1", prompt="old", interval=60)
        sched.add_task(td)
        assert sched.update_task("t1", prompt="new") is True
        assert sched._tasks["t1"].prompt == "new"

    def test_update_nonexistent(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        assert sched.update_task("nope", prompt="x") is False

    def test_update_disable_removes_next_fire(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        td = ScheduledTaskDef(name="t1", prompt="p", interval=60)
        sched.add_task(td)
        assert "t1" in sched._next_fire
        sched.update_task("t1", enabled=False)
        assert "t1" not in sched._next_fire

    def test_update_reenable_restores_next_fire(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        td = ScheduledTaskDef(name="t1", prompt="p", interval=60, enabled=False)
        sched.add_task(td)
        assert "t1" not in sched._next_fire
        sched.update_task("t1", enabled=True)
        assert "t1" in sched._next_fire

    def test_add_duplicate_name_rejected(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        sched.add_task(ScheduledTaskDef(name="t1", prompt="p1", interval=60))
        with pytest.raises(ValueError, match="Duplicate scheduled task name"):
            sched.add_task(ScheduledTaskDef(name="t1", prompt="p2", interval=60))

    def test_update_rejects_invalid_schedule(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        sched.add_task(ScheduledTaskDef(name="t1", prompt="p1", interval=60))
        with pytest.raises(ValueError, match="exactly one of cron, interval or run_at"):
            sched.update_task("t1", interval=None)

    def test_execute_builds_scheduled_message(self):
        bus = _make_message_bus()
        sched = CronScheduler(_make_config(), bus)
        task = ScheduledTaskDef(
            name="daily",
            prompt="send update",
            interval=60,
            channel="feishu",
            chat_id="c1",
        )
        asyncio.run(sched._execute_one(task))
        bus.enqueue_and_wait.assert_awaited_once()
        msg = bus.enqueue_and_wait.await_args.args[0]
        assert msg.channel == "feishu"
        assert msg.chat_id == "c1"
        assert msg.content == "send update"
        assert msg.metadata["scheduled_task"] is True
        assert msg.metadata["scheduled_task_name"] == "daily"

    def test_run_at_task_marked_disabled_after_execution(self):
        cfg = _make_config()
        cfg.save_scheduled_tasks(
            [
                {
                    "name": "once",
                    "prompt": "run once",
                    "schedule": {"run_at": "2030-01-01T17:10:00+08:00"},
                    "target": {"channel": "feishu", "chat_id": "c1"},
                    "enabled": True,
                }
            ]
        )
        bus = _make_message_bus()
        td = ScheduledTaskDef.from_dict(cfg.get_scheduled_tasks()[0])
        sched = CronScheduler(cfg, bus, task_defs=[td])
        asyncio.run(sched._execute_one(td))
        assert sched._tasks["once"].enabled is False
        assert cfg.get_scheduled_tasks()[0]["enabled"] is False


class TestScheduledTaskManager:
    def test_create_update_delete_persists(self):
        cfg = _make_config()
        manager = ScheduledTaskManager(cfg)

        created = manager.create_task(
            name="daily",
            prompt="summarize",
            channel="feishu",
            chat_id="c1",
            cron="0 9 * * *",
            enabled=True,
            require_active_runtime=False,
        )
        assert created["name"] == "daily"
        assert len(cfg.get_scheduled_tasks()) == 1

        updated = manager.update_task(
            "daily",
            prompt="summarize tech news",
            interval_seconds=3600,
            cron=None,
            require_active_runtime=False,
        )
        assert updated["schedule"] == {"interval": 3600}
        assert cfg.get_scheduled_tasks()[0]["prompt"] == "summarize tech news"

        assert manager.delete_task("daily") is True
        assert cfg.get_scheduled_tasks() == []

    def test_list_tasks_without_scheduler_uses_workspace_file(self):
        cfg = _make_config()
        cfg.save_scheduled_tasks(
            [
                {
                    "name": "daily",
                    "prompt": "summarize",
                    "schedule": "0 9 * * *",
                    "target": {"channel": "feishu", "chat_id": "c1"},
                    "enabled": True,
                }
            ]
        )
        manager = ScheduledTaskManager(cfg)
        tasks = manager.list_tasks()
        assert tasks[0]["name"] == "daily"
        assert tasks[0]["running"] is False

    def test_save_task_generates_stable_name(self):
        cfg = _make_config()
        manager = ScheduledTaskManager(cfg)

        first = manager.save_task(
            prompt="提醒大家喝水和休息",
            channel="feishu",
            chat_id="c1",
            interval_seconds=7200,
            require_active_runtime=False,
        )
        second = manager.save_task(
            prompt="提醒大家喝水和休息",
            channel="feishu",
            chat_id="c1",
            interval_seconds=7200,
            require_active_runtime=False,
        )

        assert first["name"] == second["name"]
        assert second["_action"] == "updated_existing"
        assert len(cfg.get_scheduled_tasks()) == 1

    def test_save_task_updates_existing_by_name(self):
        cfg = _make_config()
        manager = ScheduledTaskManager(cfg)
        created = manager.save_task(
            name="daily-news",
            prompt="总结科技新闻",
            channel="feishu",
            chat_id="c1",
            cron="0 9 * * *",
            require_active_runtime=False,
        )

        updated = manager.save_task(
            name="daily-news",
            prompt="总结并发送科技新闻",
            channel="feishu",
            chat_id="c1",
            cron="0 10 * * *",
            require_active_runtime=False,
        )

        assert created["_action"] == "created"
        assert updated["_action"] == "updated_by_name"
        assert cfg.get_scheduled_tasks()[0]["prompt"] == "总结并发送科技新闻"

    def test_create_task_requires_active_runtime_when_requested(self):
        cfg = _make_config()
        manager = ScheduledTaskManager(cfg)
        with pytest.raises(RuntimeError, match="runtime is not active"):
            manager.create_task(
                name="x",
                prompt="p",
                channel="feishu",
                chat_id="c1",
                interval_seconds=60,
                require_active_runtime=True,
            )


# ── remove_task cancels running ──────────────────────────────────────

class TestRemoveCancelsRunning:
    @pytest.mark.asyncio
    async def test_remove_cancels_in_flight(self):
        sched = CronScheduler(_make_config(), _make_message_bus())
        td = ScheduledTaskDef(name="t1", prompt="p", interval=60)
        sched.add_task(td)

        # Simulate a running asyncio task
        async def hang():
            await asyncio.sleep(999)

        fake_task = asyncio.create_task(hang())
        sched._running_tasks["t1"] = fake_task

        sched.remove_task("t1")
        await asyncio.sleep(0)
        assert fake_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_clears_running_tasks_after_cancellation(self):
        cfg = _make_config()
        bus = MagicMock()

        async def stuck_enqueue(msg):
            await asyncio.sleep(999)

        bus.enqueue_and_wait = AsyncMock(side_effect=stuck_enqueue)
        sched = CronScheduler(
            cfg,
            bus,
            task_defs=[
                ScheduledTaskDef(
                    name="t1", prompt="p", interval=0.1, channel="feishu", chat_id="c1"
                )
            ],
        )

        await sched.start()
        await asyncio.sleep(0.2)
        assert "t1" in sched._running_tasks
        await sched.stop()
        assert "t1" not in sched._running_tasks


# ── Integration: interval task fires and pushes ──────────────────────

class TestIntegration:
    @pytest.mark.asyncio
    async def test_interval_task_fires(self):
        cfg = _make_config()
        bus = _make_message_bus()

        td = ScheduledTaskDef(
            name="quick", prompt="do it", interval=0.1,
            channel="feishu", chat_id="c1",
        )
        sched = CronScheduler(cfg, bus, task_defs=[td])
        await sched.start()

        # Wait for the task to fire
        await asyncio.sleep(0.5)
        await sched.stop()

        bus.enqueue_and_wait.assert_called()
        msg = bus.enqueue_and_wait.call_args.args[0]
        assert msg.content == "do it"
        assert msg.chat_id == "c1"


# ── Timeout produces error response ─────────────────────────────────

class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_pushes_error(self):
        cfg = _make_config()
        bus = _make_message_bus()

        td = ScheduledTaskDef(
            name="stuck", prompt="hang", interval=0.1,
            channel="feishu", chat_id="c1",
        )
        sched = CronScheduler(cfg, bus, task_defs=[td])
        await sched.start()

        await asyncio.sleep(0.3)
        await sched.stop()

        bus.enqueue_and_wait.assert_called()


# ── Missing channel doesn't crash ────────────────────────────────────

class TestMissingChannel:
    @pytest.mark.asyncio
    async def test_missing_channel_logs_warning(self):
        cfg = _make_config()
        bus = _make_message_bus()
        td = ScheduledTaskDef(
            name="orphan", prompt="hi", interval=0.1,
            channel="nonexistent", chat_id="c1",
        )
        sched = CronScheduler(cfg, bus, task_defs=[td])
        await sched.start()
        await asyncio.sleep(0.4)
        await sched.stop()
        # Should not raise — just logs warning


# ── No overlap: skip if still running ────────────────────────────────

class TestNoOverlap:
    @pytest.mark.asyncio
    async def test_no_duplicate_fire(self):
        cfg = _make_config()
        call_count = 0

        bus = MagicMock()

        async def slow_enqueue(msg):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.5)  # longer than interval

        bus.enqueue_and_wait = AsyncMock(side_effect=slow_enqueue)

        td = ScheduledTaskDef(
            name="slow", prompt="work", interval=0.1,
            channel="feishu", chat_id="c1",
        )
        sched = CronScheduler(cfg, bus, task_defs=[td])
        await sched.start()

        await asyncio.sleep(0.4)
        await sched.stop()

        # Should have only started once (second fire skipped because still running)
        assert call_count == 1
