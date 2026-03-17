"""CronScheduler — periodic task execution service."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from croniter import croniter

from .message_bus import MessageBus

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTaskDef:
    """Definition of a single scheduled task."""

    name: str
    prompt: str
    cron: str | None = None
    interval: float | None = None
    run_at: str | None = None
    channel: str = ""
    chat_id: str = ""
    enabled: bool = True

    def validate(self) -> None:
        """Validate task scheduling and routing fields."""
        if not self.name.strip():
            raise ValueError("Scheduled task name must not be empty")
        if not self.prompt.strip():
            raise ValueError(f"Scheduled task '{self.name}' prompt must not be empty")

        has_cron = self.cron is not None
        has_interval = self.interval is not None
        has_run_at = self.run_at is not None
        if sum((has_cron, has_interval, has_run_at)) != 1:
            raise ValueError(
                f"Scheduled task '{self.name}' must define exactly one of cron, interval or run_at"
            )

        if self.interval is not None:
            if self.interval <= 0:
                raise ValueError(
                    f"Scheduled task '{self.name}' interval must be > 0 seconds"
                )

        if self.cron is not None:
            try:
                croniter(self.cron, datetime.datetime.now())
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Scheduled task '{self.name}' has invalid cron expression: {self.cron}"
                ) from exc
        if self.run_at is not None:
            # Validate and normalize one-shot run time.
            self.run_at = self._parse_run_at(self.run_at).isoformat(timespec="seconds")

    @staticmethod
    def _parse_run_at(value: Any) -> datetime.datetime:
        if isinstance(value, (int, float)):
            return datetime.datetime.fromtimestamp(
                float(value), tz=datetime.timezone.utc
            ).astimezone()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("run_at must not be empty")
            if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", text):
                parts = [int(p) for p in text.split(":")]
                hour, minute = parts[0], parts[1]
                second = parts[2] if len(parts) > 2 else 0
                if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    raise ValueError(f"Invalid run_at clock time: {text}")
                now = datetime.datetime.now().astimezone()
                candidate = now.replace(
                    hour=hour,
                    minute=minute,
                    second=second,
                    microsecond=0,
                )
                if candidate <= now:
                    candidate += datetime.timedelta(days=1)
                return candidate
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
            return dt.astimezone()
        raise ValueError(f"Unsupported run_at value: {value!r}")

    @classmethod
    def from_dict(cls, d: dict) -> ScheduledTaskDef:
        schedule = d.get("schedule")
        cron: str | None = None
        interval: float | None = None
        run_at: str | None = None

        if isinstance(schedule, str):
            cron = schedule
        elif isinstance(schedule, dict):
            if "interval" in schedule:
                interval = float(schedule["interval"])
            elif "run_at" in schedule:
                run_at = str(schedule["run_at"])
            elif "once_at" in schedule:
                run_at = str(schedule["once_at"])
            else:
                raise ValueError(
                    "Schedule object must include 'interval' or 'run_at'"
                )

        target = d.get("target", {})
        task = cls(
            name=d["name"],
            prompt=d["prompt"],
            cron=cron,
            interval=interval,
            run_at=run_at,
            channel=target.get("channel", ""),
            chat_id=target.get("chat_id", ""),
            enabled=d.get("enabled", True),
        )
        task.validate()
        return task

    def to_dict(self) -> dict[str, Any]:
        """Serialize to workspace task-file format."""
        schedule: str | dict[str, Any]
        if self.cron is not None:
            schedule = self.cron
        elif self.interval is not None:
            schedule = {"interval": self.interval}
        elif self.run_at is not None:
            schedule = {"run_at": self.run_at}
        else:
            raise ValueError(f"Scheduled task '{self.name}' has no valid schedule")
        return {
            "name": self.name,
            "prompt": self.prompt,
            "schedule": schedule,
            "target": {
                "channel": self.channel,
                "chat_id": self.chat_id,
            },
            "enabled": self.enabled,
        }


class CronScheduler:
    """Run scheduled tasks alongside the MessageBus event loop."""

    def __init__(
        self,
        config,
        message_bus: MessageBus,
        task_defs: list[ScheduledTaskDef] | None = None,
    ) -> None:
        self._config = config
        self._message_bus = message_bus

        self._tasks: dict[str, ScheduledTaskDef] = {}
        self._next_fire: dict[str, float] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._one_shot_attempts: dict[str, int] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()

        if task_defs:
            for td in task_defs:
                td.validate()
                if td.name in self._tasks:
                    raise ValueError(f"Duplicate scheduled task name: {td.name}")
                self._tasks[td.name] = td

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        now = time.monotonic()
        for name, task in self._tasks.items():
            if task.enabled:
                self._next_fire[name] = self._compute_next_fire(task, now)
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("CronScheduler started with %d task(s)", len(self._tasks))

    def is_running(self) -> bool:
        return self._running

    async def stop(self) -> None:
        self._running = False
        self._wake_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
            self._loop_task = None
        # Cancel all running tasks
        for t in list(self._running_tasks.values()):
            if not t.done():
                t.cancel()
        self._running_tasks.clear()
        logger.info("CronScheduler stopped")

    # ── Dynamic management API ────────────────────────────────────────

    def add_task(self, task: ScheduledTaskDef) -> None:
        task.validate()
        if task.name in self._tasks:
            raise ValueError(f"Duplicate scheduled task name: {task.name}")
        self._tasks[task.name] = task
        if task.enabled:
            self._next_fire[task.name] = self._compute_next_fire(task)
        self._wake_event.set()

    def remove_task(self, name: str) -> bool:
        if name not in self._tasks:
            return False
        del self._tasks[name]
        self._next_fire.pop(name, None)
        running = self._running_tasks.pop(name, None)
        if running and not running.done():
            running.cancel()
        self._wake_event.set()
        return True

    def update_task(self, name: str, **fields) -> bool:
        task = self._tasks.get(name)
        if task is None:
            return False
        updated = ScheduledTaskDef(
            name=task.name,
            prompt=task.prompt,
            cron=task.cron,
            interval=task.interval,
            run_at=task.run_at,
            channel=task.channel,
            chat_id=task.chat_id,
            enabled=task.enabled,
        )
        for key, val in fields.items():
            if hasattr(updated, key):
                object.__setattr__(updated, key, val)
        updated.validate()
        self._tasks[name] = updated
        if not updated.enabled:
            self._next_fire.pop(name, None)
        else:
            self._next_fire[name] = self._compute_next_fire(updated)
        self._wake_event.set()
        return True

    def list_tasks(self) -> list[dict]:
        now = time.monotonic()
        result = []
        for name, task in self._tasks.items():
            nf = self._next_fire.get(name)
            result.append({
                "name": name,
                "prompt": task.prompt,
                "enabled": task.enabled,
                "channel": task.channel,
                "chat_id": task.chat_id,
                "schedule_type": (
                    "run_at"
                    if task.run_at is not None
                    else "interval"
                    if task.interval is not None
                    else "cron"
                ),
                "running": name in self._running_tasks,
                "next_fire_in": round(nf - now, 1) if nf is not None else None,
            })
        return result

    # ── Internal ──────────────────────────────────────────────────────

    def _compute_next_fire(
        self, task: ScheduledTaskDef, now: float | None = None
    ) -> float:
        if now is None:
            now = time.monotonic()
        if task.interval is not None:
            return now + task.interval
        if task.cron is not None:
            wall_now = datetime.datetime.now()
            nxt = croniter(task.cron, wall_now).get_next(float)
            delta = nxt - wall_now.timestamp()
            return now + max(0.0, delta)
        if task.run_at is not None:
            run_ts = ScheduledTaskDef._parse_run_at(task.run_at).timestamp()
            delta = run_ts - time.time()
            return now + max(0.0, delta)
        raise ValueError(f"Scheduled task '{task.name}' has no valid schedule")

    async def _run_loop(self) -> None:
        try:
            while self._running:
                self._wake_event.clear()

                now = time.monotonic()
                # Collect due tasks
                due: list[ScheduledTaskDef] = []
                for name, fire_at in list(self._next_fire.items()):
                    if fire_at <= now:
                        task = self._tasks.get(name)
                        if task and task.enabled and name not in self._running_tasks:
                            due.append(task)

                # Launch due tasks
                for task in due:
                    at = asyncio.create_task(self._execute_one(task))
                    self._running_tasks[task.name] = at
                    # One-shot tasks are rescheduled only on failure.
                    if task.run_at is None:
                        self._next_fire[task.name] = self._compute_next_fire(task)
                    else:
                        self._next_fire[task.name] = time.monotonic() + 60.0

                # Calculate sleep duration
                delay = 60.0
                if self._next_fire:
                    now = time.monotonic()
                    earliest = min(self._next_fire.values())
                    delay = min(delay, max(0.1, earliest - now))

                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    def _mark_one_shot_completed(self, task_name: str) -> None:
        task = self._tasks.get(task_name)
        if task is None or task.run_at is None:
            return
        task.enabled = False
        self._next_fire.pop(task_name, None)
        try:
            raw_tasks = self._config.get_scheduled_tasks()
            updated: list[dict[str, Any]] = []
            changed = False
            now_iso = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            for item in raw_tasks:
                current = dict(item)
                if current.get("name") == task_name:
                    current["enabled"] = False
                    schedule = current.get("schedule")
                    if isinstance(schedule, dict):
                        schedule = dict(schedule)
                        schedule["completed_at"] = now_iso
                        current["schedule"] = schedule
                    changed = True
                updated.append(current)
            if changed:
                self._config.save_scheduled_tasks(updated)
        except Exception:
            logger.exception("Failed persisting one-shot completion: %s", task_name)

    async def _execute_one(self, task: ScheduledTaskDef) -> None:
        from .channels.base import InboundMessage

        logger.info("Dispatching scheduled task via MessageBus: %s", task.name)
        try:
            message = InboundMessage(
                channel=task.channel,
                sender_id=f"scheduled:{task.name}",
                chat_id=task.chat_id,
                content=task.prompt,
                metadata={
                    "scheduled_task": True,
                    "scheduled_task_name": task.name,
                },
            )
            response = await self._message_bus.enqueue_and_wait(message)
            if task.run_at is not None:
                # Check if response indicates an error before marking complete
                response_text = getattr(response, "text", "") or ""
                if response_text.startswith("处理任务时出错"):
                    raise RuntimeError(f"One-shot task returned error: {response_text}")
                self._one_shot_attempts.pop(task.name, None)
                self._mark_one_shot_completed(task.name)
        except AttributeError:
            # Backward compatibility for tests/mocks without enqueue_and_wait.
            await self._message_bus.enqueue(
                InboundMessage(
                    channel=task.channel,
                    sender_id=f"scheduled:{task.name}",
                    chat_id=task.chat_id,
                    content=task.prompt,
                    metadata={
                        "scheduled_task": True,
                        "scheduled_task_name": task.name,
                    },
                )
            )
            if task.run_at is not None:
                self._one_shot_attempts.pop(task.name, None)
                self._mark_one_shot_completed(task.name)
        except asyncio.CancelledError:
            logger.info("Scheduled task cancelled: %s", task.name)
            raise
        except Exception:
            logger.exception("Failed scheduled task dispatch: %s", task.name)
            if task.run_at is not None:
                attempts = self._one_shot_attempts.get(task.name, 0) + 1
                self._one_shot_attempts[task.name] = attempts
                if attempts >= 3:
                    logger.warning(
                        "One-shot task '%s' exceeded max retries (%d), disabling.",
                        task.name, attempts,
                    )
                    self._one_shot_attempts.pop(task.name, None)
                    task.enabled = False
                    self._next_fire.pop(task.name, None)
                    self._mark_one_shot_completed(task.name)
                else:
                    # Retry after a short delay.
                    self._next_fire[task.name] = time.monotonic() + 60.0
        finally:
            self._running_tasks.pop(task.name, None)


class ScheduledTaskManager:
    """Persist and synchronize scheduled tasks with the live scheduler."""

    def __init__(self, config: Any, scheduler: CronScheduler | None = None) -> None:
        self._config = config
        self._scheduler = scheduler

    def bind_scheduler(self, scheduler: CronScheduler | None) -> None:
        self._scheduler = scheduler

    @staticmethod
    def _schedule_key(cron: str | None, interval_seconds: float | None) -> str:
        if cron is not None:
            return f"cron:{cron.strip()}"
        if interval_seconds is not None:
            value = int(interval_seconds) if float(interval_seconds).is_integer() else interval_seconds
            return f"interval:{value}"
        raise ValueError("Scheduled task must define cron, interval_seconds or run_at")

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.strip().lower())
        slug = slug.strip("-")
        return slug or "task"

    def suggest_task_name(
        self,
        *,
        prompt: str,
        channel: str,
        chat_id: str,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
    ) -> str:
        """Generate a stable task name from task intent and routing."""
        if delay_seconds is not None:
            target = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay_seconds)
            run_at = target.isoformat(timespec="seconds")
        if run_at is not None:
            schedule = f"run_at:{ScheduledTaskDef._parse_run_at(run_at).isoformat(timespec='minutes')}"
        else:
            schedule = self._schedule_key(cron, interval_seconds)
        prompt_slug = self._slugify(prompt)[:24]
        route_slug = self._slugify(f"{channel}-{chat_id}")[:24]
        schedule_slug = self._slugify(schedule)[:24]
        return f"{prompt_slug}-{route_slug}-{schedule_slug}".strip("-")

    def _equivalent_task(
        self,
        task: ScheduledTaskDef,
        *,
        prompt: str,
        channel: str,
        chat_id: str,
        cron: str | None,
        interval_seconds: float | None,
        run_at: str | None,
    ) -> bool:
        return (
            task.prompt.strip() == prompt.strip()
            and task.channel == channel
            and task.chat_id == chat_id
            and task.cron == cron
            and task.interval == interval_seconds
            and task.run_at == run_at
        )

    def _ensure_runtime_active(self, require_active_runtime: bool) -> None:
        if not require_active_runtime:
            return
        if self._scheduler is None or not self._scheduler.is_running():
            raise RuntimeError(
                "Scheduled task runtime is not active. Start gateway mode, "
                "or set require_active_runtime=false to persist only."
            )

    def _load_defs(self) -> list[ScheduledTaskDef]:
        raw_tasks = self._config.get_scheduled_tasks()
        seen: set[str] = set()
        defs: list[ScheduledTaskDef] = []
        for raw in raw_tasks:
            task = ScheduledTaskDef.from_dict(raw)
            if task.name in seen:
                raise ValueError(f"Duplicate scheduled task name: {task.name}")
            seen.add(task.name)
            defs.append(task)
        return defs

    def _save_defs(self, defs: list[ScheduledTaskDef]) -> None:
        self._config.save_scheduled_tasks([task.to_dict() for task in defs])

    def list_tasks(self) -> list[dict[str, Any]]:
        if self._scheduler is not None:
            return self._scheduler.list_tasks()
        return [
            {
                "name": task.name,
                "prompt": task.prompt,
                "enabled": task.enabled,
                "channel": task.channel,
                "chat_id": task.chat_id,
                "running": False,
                "next_fire_in": None,
            }
            for task in self._load_defs()
        ]

    def create_task(
        self,
        *,
        name: str | None,
        prompt: str,
        channel: str,
        chat_id: str,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
        enabled: bool = True,
        require_active_runtime: bool = False,
    ) -> dict[str, Any]:
        self._ensure_runtime_active(require_active_runtime)
        if delay_seconds is not None:
            if run_at is not None:
                raise ValueError("Cannot specify both delay_seconds and run_at")
            target = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay_seconds)
            run_at = target.isoformat(timespec="seconds")
        defs = self._load_defs()
        resolved_name = (name or "").strip() or self.suggest_task_name(
            prompt=prompt,
            channel=channel,
            chat_id=chat_id,
            cron=cron,
            interval_seconds=interval_seconds,
            run_at=run_at,
        )
        task = ScheduledTaskDef(
            name=resolved_name,
            prompt=prompt,
            cron=cron,
            interval=interval_seconds,
            run_at=run_at,
            channel=channel,
            chat_id=chat_id,
            enabled=enabled,
        )
        task.validate()
        for existing in defs:
            if self._equivalent_task(
                existing,
                prompt=prompt,
                channel=channel,
                chat_id=chat_id,
                cron=cron,
                interval_seconds=interval_seconds,
                run_at=run_at,
            ):
                updated = self.update_task(
                    existing.name,
                    prompt=prompt,
                    channel=channel,
                    chat_id=chat_id,
                    cron=cron,
                    interval_seconds=interval_seconds,
                    run_at=run_at,
                    enabled=enabled,
                    require_active_runtime=require_active_runtime,
                )
                updated["_action"] = "updated_existing"
                return updated
            if existing.name == task.name:
                raise ValueError(f"Duplicate scheduled task name: {task.name}")
        defs.append(task)
        self._save_defs(defs)
        if self._scheduler is not None:
            self._scheduler.add_task(task)
        payload = task.to_dict()
        payload["_action"] = "created"
        return payload

    def update_task(
        self,
        name: str,
        *,
        prompt: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
        enabled: bool | None = None,
        require_active_runtime: bool = False,
    ) -> dict[str, Any]:
        self._ensure_runtime_active(require_active_runtime)
        if delay_seconds is not None:
            if run_at is not None:
                raise ValueError("Cannot specify both delay_seconds and run_at")
            target = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay_seconds)
            run_at = target.isoformat(timespec="seconds")
        defs = self._load_defs()
        for index, task in enumerate(defs):
            if task.name != name:
                continue
            updated = ScheduledTaskDef(
                name=task.name,
                prompt=task.prompt if prompt is None else prompt,
                cron=task.cron,
                interval=task.interval,
                run_at=task.run_at,
                channel=task.channel if channel is None else channel,
                chat_id=task.chat_id if chat_id is None else chat_id,
                enabled=task.enabled if enabled is None else enabled,
            )
            if cron is not None or interval_seconds is not None or run_at is not None:
                updated.cron = cron
                updated.interval = interval_seconds
                updated.run_at = run_at
            updated.validate()
            defs[index] = updated
            self._save_defs(defs)
            if self._scheduler is not None:
                synced = self._scheduler.update_task(
                    name,
                    prompt=updated.prompt,
                    cron=updated.cron,
                    interval=updated.interval,
                    run_at=updated.run_at,
                    channel=updated.channel,
                    chat_id=updated.chat_id,
                    enabled=updated.enabled,
                )
                if not synced and updated.enabled:
                    self._scheduler.add_task(updated)
            return updated.to_dict()
        raise ValueError(f"Scheduled task not found: {name}")

    def delete_task(self, name: str) -> bool:
        defs = self._load_defs()
        remaining = [task for task in defs if task.name != name]
        if len(remaining) == len(defs):
            return False
        self._save_defs(remaining)
        if self._scheduler is not None:
            self._scheduler.remove_task(name)
        return True

    def render_tasks(self) -> str:
        """JSON-friendly string for LLM tool responses."""
        return json.dumps(
            {"tasks": self.list_tasks()},
            ensure_ascii=False,
            indent=2,
        )

    def save_task(
        self,
        *,
        name: str | None = None,
        prompt: str,
        channel: str,
        chat_id: str,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: str | None = None,
        delay_seconds: float | None = None,
        enabled: bool = True,
        require_active_runtime: bool = False,
    ) -> dict[str, Any]:
        """Create or update a task using explicit or inferred identity."""
        self._ensure_runtime_active(require_active_runtime)
        if delay_seconds is not None:
            if run_at is not None:
                raise ValueError("Cannot specify both delay_seconds and run_at")
            target = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay_seconds)
            run_at = target.isoformat(timespec="seconds")
        if name and any(task.name == name for task in self._load_defs()):
            payload = self.update_task(
                name,
                prompt=prompt,
                channel=channel,
                chat_id=chat_id,
                cron=cron,
                interval_seconds=interval_seconds,
                run_at=run_at,
                enabled=enabled,
                require_active_runtime=require_active_runtime,
            )
            payload["_action"] = "updated_by_name"
            return payload
        return self.create_task(
            name=name,
            prompt=prompt,
            channel=channel,
            chat_id=chat_id,
            cron=cron,
            interval_seconds=interval_seconds,
            run_at=run_at,
            enabled=enabled,
            require_active_runtime=require_active_runtime,
        )
