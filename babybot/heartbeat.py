"""Heartbeat-based idle timeout for long-running agent tasks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _TaskHeartbeatRecord:
    last_beat: float
    progress: float | None = None
    status: str = "idle"


class TaskHeartbeatHandle:
    """Per-task heartbeat handle that can also keep a parent heartbeat alive."""

    def __init__(
        self,
        registry: "TaskHeartbeatRegistry",
        flow_id: str,
        task_id: str,
        parent: "Heartbeat | None" = None,
    ) -> None:
        self._registry = registry
        self._flow_id = flow_id
        self._task_id = task_id
        self._parent = parent

    def beat(
        self,
        *,
        progress: float | None = None,
        status: str | None = None,
    ) -> None:
        self._registry.beat(
            self._flow_id,
            self._task_id,
            progress=progress,
            status=status,
        )
        if self._parent is not None:
            self._parent.beat()

    @contextlib.asynccontextmanager
    async def keep_alive(self, interval: float = 5.0) -> AsyncIterator[None]:
        async def _ticker() -> None:
            while True:
                await asyncio.sleep(interval)
                self.beat()

        task = asyncio.create_task(_ticker())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


class TaskHeartbeatRegistry:
    """Track child-task heartbeat state independently per flow."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, _TaskHeartbeatRecord]] = {}

    def handle(
        self,
        flow_id: str,
        task_id: str,
        parent: "Heartbeat | None" = None,
    ) -> TaskHeartbeatHandle:
        self._records.setdefault(flow_id, {}).setdefault(
            task_id,
            _TaskHeartbeatRecord(last_beat=time.monotonic()),
        )
        return TaskHeartbeatHandle(self, flow_id, task_id, parent=parent)

    def beat(
        self,
        flow_id: str,
        task_id: str,
        *,
        progress: float | None = None,
        status: str | None = None,
    ) -> None:
        record = self._records.setdefault(flow_id, {}).setdefault(
            task_id,
            _TaskHeartbeatRecord(last_beat=time.monotonic()),
        )
        record.last_beat = time.monotonic()
        if progress is not None:
            record.progress = progress
        if status is not None:
            record.status = status

    def snapshot(self, flow_id: str) -> dict[str, dict[str, float | str | None]]:
        records = self._records.get(flow_id, {})
        return {
            task_id: {
                "last_beat": record.last_beat,
                "progress": record.progress,
                "status": record.status,
            }
            for task_id, record in records.items()
        }

    def stale_tasks(
        self,
        flow_id: str,
        *,
        stale_after_s: float,
    ) -> dict[str, dict[str, float | str | None]]:
        now = time.monotonic()
        records = self._records.get(flow_id, {})
        stale: dict[str, dict[str, float | str | None]] = {}
        for task_id, record in records.items():
            age_s = now - record.last_beat
            if age_s >= stale_after_s:
                stale[task_id] = {
                    "last_beat": record.last_beat,
                    "progress": record.progress,
                    "status": record.status,
                    "age_s": age_s,
                }
        return stale


class Heartbeat:
    """Track liveness via periodic beats; cancel work after idle timeout.

    Usage::

        hb = Heartbeat(idle_timeout=60)

        # In worker coroutines, call hb.beat() on every meaningful progress.
        # The watcher cancels the wrapped coroutine if no beat arrives
        # within *idle_timeout* seconds.

        result = await hb.watch(some_coro(), hard_timeout=180)

        # For long LLM calls that can't beat internally, use keep_alive:
        async with hb.keep_alive():
            result = await slow_llm_call()
    """

    def __init__(self, idle_timeout: float = 60.0) -> None:
        self.idle_timeout = idle_timeout
        self._last_beat: float = time.monotonic()
        self._event = asyncio.Event()

    def beat(self) -> None:
        """Record a heartbeat (called by agent steps / tool invocations)."""
        self._last_beat = time.monotonic()
        self._event.set()

    @contextlib.asynccontextmanager
    async def keep_alive(self, interval: float | None = None) -> AsyncIterator[None]:
        """Background ticker that beats periodically.

        Use this around long-running awaits (e.g. LLM calls) that cannot
        beat on their own.  Beats every *interval* seconds (default:
        ``idle_timeout / 3``).
        """
        if interval is None:
            interval = max(5.0, self.idle_timeout / 3)

        async def _ticker() -> None:
            while True:
                await asyncio.sleep(interval)
                self.beat()

        task = asyncio.create_task(_ticker())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def watch(self, coro, hard_timeout: float | None = None) -> T:  # type: ignore[type-var]
        """Run *coro* as a task while monitoring heartbeat liveness.

        Raises ``asyncio.TimeoutError`` if:
        - No heartbeat arrives within *idle_timeout* seconds, **or**
        - Total wall-clock time exceeds *hard_timeout* (if provided).
        """
        loop = asyncio.get_running_loop()
        task: asyncio.Task = asyncio.ensure_future(coro)
        deadline = (loop.time() + hard_timeout) if hard_timeout else None

        # Reset state so the first idle window starts now.
        self._last_beat = time.monotonic()
        self._event.clear()

        try:
            while not task.done():
                self._event.clear()
                remaining_idle = self.idle_timeout - (time.monotonic() - self._last_beat)

                if deadline is not None:
                    remaining_hard = deadline - loop.time()
                    if remaining_hard <= 0:
                        task.cancel()
                        raise asyncio.TimeoutError(
                            f"Hard timeout ({hard_timeout}s) exceeded"
                        )
                    wait_time = min(remaining_idle, remaining_hard)
                else:
                    wait_time = remaining_idle

                if wait_time <= 0:
                    task.cancel()
                    raise asyncio.TimeoutError(
                        f"Idle timeout ({self.idle_timeout}s) — no heartbeat"
                    )

                # Wait for EITHER the task to complete OR a heartbeat event,
                # whichever comes first.  This ensures we notice task completion
                # immediately instead of waiting for the event timeout.
                event_future = asyncio.ensure_future(self._event.wait())
                try:
                    done_set, pending_set = await asyncio.wait(
                        {task, event_future},
                        timeout=max(0.1, wait_time),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    # Always clean up the event future to avoid leaks.
                    if not event_future.done():
                        event_future.cancel()
                        try:
                            await event_future
                        except (asyncio.CancelledError, Exception):
                            pass

                if task in done_set:
                    # Task finished — break out and return result.
                    break

                if not done_set:
                    # Neither completed — timeout expired, check idle / hard.
                    idle_elapsed = time.monotonic() - self._last_beat
                    if idle_elapsed >= self.idle_timeout:
                        task.cancel()
                        raise asyncio.TimeoutError(
                            f"Idle timeout ({self.idle_timeout}s) — no heartbeat"
                        )
                    if deadline is not None and loop.time() >= deadline:
                        task.cancel()
                        raise asyncio.TimeoutError(
                            f"Hard timeout ({hard_timeout}s) exceeded"
                        )
                # else: heartbeat event fired — loop again with refreshed timer.

            return await task
        except asyncio.CancelledError:
            # If the task was cancelled by us, convert to TimeoutError.
            if not task.done():
                task.cancel()
            raise asyncio.TimeoutError("Task cancelled during heartbeat watch")
