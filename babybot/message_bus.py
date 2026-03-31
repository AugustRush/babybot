"""MessageBus — async queue + semaphore based message processing."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import dataclasses
import datetime
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .feedback_events import (
    feedback_dedupe_key,
    normalize_runtime_feedback_event,
    render_runtime_feedback_event,
)
from .heartbeat import Heartbeat

if TYPE_CHECKING:
    from .channels.base import BaseChannel, InboundMessage
    from .config import Config
    from .orchestrator import OrchestratorAgent, TaskResponse

logger = logging.getLogger(__name__)


def _runtime_event_job_id(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        job_id = str(event.get("job_id", "") or "").strip()
        if job_id:
            return job_id
        payload = event.get("payload") or {}
        if isinstance(payload, dict):
            job_id = str(payload.get("job_id", "") or "").strip()
            if job_id:
                return job_id
    return ""


@dataclass
class MessageEnvelope:
    """Wrapper around an inbound message with tracking metadata."""

    message: InboundMessage
    heartbeat: Heartbeat
    completion: asyncio.Future["TaskResponse"] | None = None
    created_at: float = field(default_factory=time.monotonic)


class MessageBus:
    """Queue-based message dispatcher with concurrency control.

    Replaces per-chat Lock with:
    - An asyncio.Queue for decoupling ingestion from processing
    - A global Semaphore for overall concurrency
    - Per-chat Semaphores for per-conversation concurrency
    """

    def __init__(
        self,
        config: Config,
        orchestrator: OrchestratorAgent,
        channels: dict[str, BaseChannel],
    ) -> None:
        self._config = config
        self._orchestrator = orchestrator
        self._channels = channels
        queue_size = int(getattr(config.system, "message_queue_maxsize", 1000) or 1000)
        self._user_queue: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue(
            maxsize=max(1, queue_size)
        )
        self._scheduled_queue: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue(
            maxsize=max(1, queue_size)
        )
        self._global_sem = asyncio.Semaphore(config.system.max_concurrency)
        self._chat_sems: OrderedDict[str, asyncio.Semaphore] = OrderedDict()
        max_concurrency = max(1, int(config.system.max_concurrency))
        requested_scheduled = int(
            getattr(
                config.system,
                "scheduled_max_concurrency",
                max(1, max_concurrency // 4),
            )
            or 0
        )
        if max_concurrency <= 1:
            self._scheduled_workers = 0
            self._user_workers = 1
        else:
            self._scheduled_workers = min(
                max(1, requested_scheduled), max_concurrency - 1
            )
            self._user_workers = max_concurrency - self._scheduled_workers
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._running = False
        self._accepting = False
        # Cache inspect.signature results to avoid per-message overhead.
        self._process_task_parameters: set[str] | None = None
        self._supports_stream_callback: bool | None = None
        self._supports_runtime_event_callback: bool | None = None
        self._supports_interactive_output_callback: bool | None = None
        self._supports_intermediate_message: bool | None = None

    def _get_chat_sem(self, key: str) -> asyncio.Semaphore:
        if not isinstance(self._chat_sems, OrderedDict):
            self._chat_sems = OrderedDict(self._chat_sems)
        sem = self._chat_sems.get(key)
        if sem is not None:
            self._chat_sems.move_to_end(key)
            return sem

        while len(self._chat_sems) >= 2000:
            # Only evict semaphores that are not currently acquired.
            evicted = False
            for evict_key in list(self._chat_sems.keys()):
                candidate = self._chat_sems[evict_key]
                # A semaphore at its initial value is idle (not acquired).
                if candidate._value >= self._config.system.max_per_chat:  # noqa: SLF001
                    del self._chat_sems[evict_key]
                    evicted = True
                    break
            if not evicted:
                # All semaphores are in use; exceed the soft limit rather
                # than evicting an active semaphore.
                break
        sem = asyncio.Semaphore(self._config.system.max_per_chat)
        self._chat_sems[key] = sem
        return sem

    async def enqueue(self, msg: InboundMessage) -> None:
        """Non-blocking enqueue of an inbound message."""
        if not self._accepting:
            raise RuntimeError("MessageBus is not accepting new messages")
        hb = Heartbeat(idle_timeout=float(self._config.system.idle_timeout))
        envelope = MessageEnvelope(message=msg, heartbeat=hb)
        queue = (
            self._scheduled_queue
            if self._is_scheduled(msg) and self._scheduled_workers > 0
            else self._user_queue
        )
        await queue.put(envelope)

    async def enqueue_and_wait(
        self,
        msg: InboundMessage,
        timeout: float | None = None,
    ) -> "TaskResponse":
        """Enqueue and wait until processing finishes."""
        if not self._accepting:
            raise RuntimeError("MessageBus is not accepting new messages")
        loop = asyncio.get_running_loop()
        hb = Heartbeat(idle_timeout=float(self._config.system.idle_timeout))
        completion: asyncio.Future["TaskResponse"] = loop.create_future()
        envelope = MessageEnvelope(message=msg, heartbeat=hb, completion=completion)
        queue = (
            self._scheduled_queue
            if self._is_scheduled(msg) and self._scheduled_workers > 0
            else self._user_queue
        )
        await queue.put(envelope)
        if timeout is not None:
            return await asyncio.wait_for(completion, timeout=timeout)
        return await completion

    @staticmethod
    def _is_scheduled(msg: InboundMessage) -> bool:
        return bool((msg.metadata or {}).get("scheduled_task"))

    @staticmethod
    def _format_runtime_progress_text(event_payload: dict[str, Any]) -> str:
        return render_runtime_feedback_event(
            normalize_runtime_feedback_event(event_payload)
        )

    def _ensure_process_task_signature_cached(self) -> None:
        if self._process_task_parameters is not None:
            return
        try:
            self._process_task_parameters = set(
                inspect.signature(self._orchestrator.process_task).parameters
            )
        except (TypeError, ValueError):
            self._process_task_parameters = set()

    async def start(self) -> None:
        """Start the consumer loop."""
        if self._running:
            return
        self._running = True
        self._accepting = True
        self._worker_tasks = []
        for idx in range(self._user_workers):
            self._worker_tasks.append(
                asyncio.create_task(
                    self._worker_loop(self._user_queue, f"user-{idx + 1}")
                )
            )
        for idx in range(self._scheduled_workers):
            self._worker_tasks.append(
                asyncio.create_task(
                    self._worker_loop(self._scheduled_queue, f"scheduled-{idx + 1}")
                )
            )
        logger.info(
            "MessageBus started user_workers=%d scheduled_workers=%d",
            self._user_workers,
            self._scheduled_workers,
        )

    async def stop(self, *, drain: bool = True) -> None:
        """Graceful shutdown: signal consumer and wait."""
        if not self._running:
            return
        self._accepting = False
        if drain:
            try:
                await asyncio.wait_for(self._user_queue.join(), timeout=30.0)
                await asyncio.wait_for(self._scheduled_queue.join(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("MessageBus drain timed out; forcing shutdown")

        # Stop all workers with sentinels.
        for _ in range(self._user_workers):
            try:
                self._user_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning("User queue full during stop; cancelling workers")
                break
        for _ in range(self._scheduled_workers):
            try:
                self._scheduled_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning("Scheduled queue full during stop; cancelling workers")
                break

        if self._worker_tasks:
            done, pending = await asyncio.wait(self._worker_tasks, timeout=15.0)
            for task in pending:
                task.cancel()
            self._worker_tasks = []

        self._running = False
        logger.info("MessageBus stopped")

    async def _worker_loop(
        self,
        queue: asyncio.Queue[MessageEnvelope | None],
        label: str,
    ) -> None:
        while True:
            envelope = await queue.get()
            try:
                if envelope is None:
                    return
                await self._dispatch(envelope)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "MessageBus worker %s failed to dispatch envelope", label
                )
            finally:
                queue.task_done()

    async def _dispatch(self, envelope: MessageEnvelope) -> None:
        """Acquire semaphores and process."""
        msg = envelope.message
        chat_key = f"{msg.channel}:{msg.chat_id}"

        try:
            async with self._global_sem:
                chat_sem = self._get_chat_sem(chat_key)
                async with chat_sem:
                    await self._process(envelope)
        except Exception as exc:
            if envelope.completion is not None and not envelope.completion.done():
                envelope.completion.set_exception(exc)
            raise

    async def _process(self, envelope: MessageEnvelope) -> None:
        """Process a single message: ack → execute → respond."""
        from .channels.tools import ChannelToolContext
        from .orchestrator import TaskResponse

        msg = envelope.message
        heartbeat = envelope.heartbeat
        start = time.perf_counter()

        logger.info(
            "Bus processing channel=%s chat_id=%s sender_id=%s content=%s",
            msg.channel,
            msg.chat_id,
            msg.sender_id,
            (msg.content or "")[:120],
        )

        channel = self._channels.get(msg.channel)
        is_scheduled = bool((msg.metadata or {}).get("scheduled_task"))

        # Send instant acknowledgement if configured.
        if channel and self._config.system.send_ack and not is_scheduled:
            try:
                ack_response = TaskResponse(text="收到，正在处理...")
                await channel.send_response(
                    msg.chat_id,
                    ack_response,
                    sender_id=msg.sender_id,
                    metadata=msg.metadata,
                )
            except Exception:
                logger.warning(
                    "Failed to send ack for chat_id=%s", msg.chat_id, exc_info=True
                )

        # Set channel context for this message (contextvars — concurrency safe).
        ctx = ChannelToolContext(
            channel_name=msg.channel,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            metadata={
                **(msg.metadata or {}),
                "request_received_at": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            },
        )
        ChannelToolContext.set_current(ctx)

        # hard_timeout is a safety net; idle_timeout handles "stuck" detection.
        # Ensure hard_timeout is at least 3x idle_timeout to avoid killing active tasks.
        hard_timeout = max(
            float(self._config.system.timeout),
            float(self._config.system.idle_timeout) * 3,
        )
        stream_message_id: str | None = None
        stream_last_patched = ""
        interactive_last_sent_text = ""
        interactive_sent_partial = False
        stream_lock = asyncio.Lock()
        stream_detached = False
        stream_cfg = getattr(channel, "config", None)
        stream_enabled = bool(
            channel
            and msg.channel == "feishu"
            and not is_scheduled
            and getattr(stream_cfg, "stream_reply", False)
            and callable(getattr(channel, "create_stream_message", None))
            and callable(getattr(channel, "patch_stream_message", None))
        )

        async def _stream_callback(accumulated_text: str) -> None:
            nonlocal stream_message_id, stream_last_patched, stream_detached
            text = (accumulated_text or "").strip()
            if not text:
                return
            async with stream_lock:
                if not stream_enabled or channel is None or stream_detached:
                    return
                if stream_message_id is None:
                    stream_message_id = await channel.create_stream_message(  # type: ignore[attr-defined]
                        msg.chat_id,
                        text,
                        sender_id=msg.sender_id,
                        metadata=msg.metadata,
                    )
                    if stream_message_id:
                        stream_last_patched = text
                    return
                if text == stream_last_patched:
                    return
                ok = await channel.patch_stream_message(stream_message_id, text)  # type: ignore[attr-defined]
                if ok:
                    stream_last_patched = text

        async def _reset_stream() -> None:
            """Detach the current stream card so the next stream_callback creates a new one."""
            nonlocal stream_message_id, stream_last_patched
            async with stream_lock:
                stream_message_id = None
                stream_last_patched = ""

        async def _send_intermediate_message(text: str) -> None:
            """Send an independent message card (not patching the stream card)."""
            if not text or not text.strip() or channel is None:
                return
            try:
                await channel.send_response(
                    msg.chat_id,
                    TaskResponse(text=text),
                    sender_id=msg.sender_id,
                    metadata=msg.metadata,
                )
            except Exception:
                logger.warning(
                    "Failed to send intermediate message channel=%s chat_id=%s",
                    msg.channel,
                    msg.chat_id,
                    exc_info=True,
                )

        process_kwargs: dict[str, Any] = {
            "chat_key": f"{msg.channel}:{msg.chat_id}",
            "heartbeat": heartbeat,
            "media_paths": msg.media_paths,
        }
        runtime_events: list[dict[str, Any]] = []
        last_runtime_progress_key: tuple[str, str, str, str, str, str, str] | None = None
        last_runtime_progress_text_by_scope: dict[tuple[str, str, str, str], str] = {}

        async def _runtime_event_callback(event: Any) -> None:
            nonlocal stream_detached, last_runtime_progress_key
            if dataclasses.is_dataclass(event):
                payload = dataclasses.asdict(event)
            elif isinstance(event, dict):
                payload = dict(event)
            else:
                payload = {
                    "event": getattr(event, "event", ""),
                    "task_id": getattr(event, "task_id", ""),
                    "flow_id": getattr(event, "flow_id", ""),
                    "payload": dict(getattr(event, "payload", {}) or {}),
                }
            runtime_events.append(payload)
            if isinstance(msg.metadata, dict):
                msg.metadata["_runtime_events"] = list(runtime_events)
            logger.info(
                "Runtime event channel=%s chat_id=%s flow_id=%s task_id=%s event=%s",
                msg.channel,
                msg.chat_id,
                payload.get("flow_id", ""),
                payload.get("task_id", ""),
                payload.get("event", ""),
            )
            normalized_event = normalize_runtime_feedback_event(payload)
            progress_text = self._format_runtime_progress_text(payload)
            if not progress_text or channel is None:
                return
            progress_key = feedback_dedupe_key(normalized_event)
            if progress_key == last_runtime_progress_key:
                return
            progress_scope = (
                normalized_event.job_id,
                normalized_event.flow_id,
                normalized_event.task_id,
                normalized_event.stage,
            )
            if last_runtime_progress_text_by_scope.get(progress_scope) == progress_text:
                return
            last_runtime_progress_key = progress_key
            last_runtime_progress_text_by_scope[progress_scope] = progress_text
            if (
                normalized_event.stage != "interactive_session"
                and normalized_event.state in {"running", "completed", "failed"}
            ):
                stream_detached = True
            if (
                normalized_event.stage == "interactive_session"
                and msg.channel != "feishu"
            ):
                return
            try:
                await channel.send_response(
                    msg.chat_id,
                    TaskResponse(text=progress_text),
                    sender_id=msg.sender_id,
                    metadata=msg.metadata,
                    message_format="post",
                )
            except Exception:
                logger.warning(
                    "Failed to send runtime progress channel=%s chat_id=%s event=%s",
                    msg.channel,
                    msg.chat_id,
                    payload.get("event", ""),
                    exc_info=True,
                )

        async def _interactive_output_callback(event: Any) -> None:
            nonlocal interactive_last_sent_text, interactive_sent_partial
            if dataclasses.is_dataclass(event):
                payload = dataclasses.asdict(event)
            elif isinstance(event, dict):
                payload = dict(event)
            else:
                payload = {
                    "event": getattr(event, "event", ""),
                    "text": getattr(event, "text", ""),
                    "delta": getattr(event, "delta", ""),
                }
            event_name = str(payload.get("event", "") or "").strip().lower()
            text = str(payload.get("text", "") or "").strip()
            if not text or channel is None:
                return
            if msg.channel == "feishu" and stream_enabled:
                await _stream_callback(text)
                return
            should_send = False
            if event_name == "message_delta" and not interactive_sent_partial:
                should_send = True
                interactive_sent_partial = True
            elif event_name == "message_complete":
                should_send = True
            if not should_send or text == interactive_last_sent_text:
                return
            try:
                await channel.send_response(
                    msg.chat_id,
                    TaskResponse(text=text),
                    sender_id=msg.sender_id,
                    metadata=msg.metadata,
                )
                interactive_last_sent_text = text
            except Exception:
                logger.warning(
                    "Failed to send interactive output channel=%s chat_id=%s event=%s",
                    msg.channel,
                    msg.chat_id,
                    event_name,
                    exc_info=True,
                )

        if stream_enabled:
            if self._supports_stream_callback is None:
                self._ensure_process_task_signature_cached()
                self._supports_stream_callback = (
                    "stream_callback" in (self._process_task_parameters or set())
                )
            if self._supports_stream_callback:
                _stream_callback.reset = _reset_stream  # type: ignore[attr-defined]
                process_kwargs["stream_callback"] = _stream_callback
        if self._supports_runtime_event_callback is None:
            self._ensure_process_task_signature_cached()
            self._supports_runtime_event_callback = (
                "runtime_event_callback" in (self._process_task_parameters or set())
            )
        if self._supports_runtime_event_callback:
            process_kwargs["runtime_event_callback"] = _runtime_event_callback
        if self._supports_interactive_output_callback is None:
            self._ensure_process_task_signature_cached()
            self._supports_interactive_output_callback = (
                "interactive_output_callback"
                in (self._process_task_parameters or set())
            )
        if self._supports_interactive_output_callback:
            process_kwargs["interactive_output_callback"] = _interactive_output_callback
        if channel is not None:
            if self._supports_intermediate_message is None:
                self._ensure_process_task_signature_cached()
                self._supports_intermediate_message = (
                    "send_intermediate_message"
                    in (self._process_task_parameters or set())
                )
            if self._supports_intermediate_message:
                process_kwargs["send_intermediate_message"] = _send_intermediate_message
        try:
            response = await heartbeat.watch(
                self._orchestrator.process_task(
                    msg.content,
                    **process_kwargs,
                ),
                hard_timeout=hard_timeout,
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "Task timeout channel=%s chat_id=%s reason=%s",
                msg.channel,
                msg.chat_id,
                str(exc),
            )
            job_id = _runtime_event_job_id(runtime_events)
            job_hint = f" job_id={job_id}。" if job_id else ""
            response = TaskResponse(
                text=(
                    f"任务执行超时（{self._config.system.idle_timeout}s 空闲 / "
                    f"{hard_timeout:.0f}s 上限）。"
                    f"{job_hint}请尝试更具体的指令，或要求分步骤执行。"
                )
            )
        except Exception as e:
            logger.exception(
                "Error processing task channel=%s chat_id=%s", msg.channel, msg.chat_id
            )
            response = TaskResponse(text=f"处理失败：{e}")
        finally:
            ChannelToolContext.set_current(None)

        # Send actual response.
        if channel:
            try:
                streamed = bool(stream_message_id)
                if streamed and not stream_detached and response.text.strip():
                    final_text = response.text.strip()
                    if stream_last_patched != final_text:
                        patched = await channel.patch_stream_message(  # type: ignore[attr-defined]
                            stream_message_id,
                            final_text,
                        )
                        if patched:
                            stream_last_patched = final_text
                if streamed and not stream_detached and response.media_paths:
                    await channel.send_response(
                        msg.chat_id,
                        TaskResponse(text="", media_paths=list(response.media_paths)),
                        sender_id=msg.sender_id,
                        metadata=msg.metadata,
                    )
                elif not streamed or stream_detached:
                    final_text = response.text.strip()
                    outbound_response = response
                    if final_text and final_text == interactive_last_sent_text:
                        outbound_response = TaskResponse(
                            text="",
                            media_paths=list(response.media_paths or []),
                        )
                    if outbound_response.text or outbound_response.media_paths:
                        await channel.send_response(
                            msg.chat_id,
                            outbound_response,
                            sender_id=msg.sender_id,
                            metadata=msg.metadata,
                        )
            except Exception:
                logger.exception("Error sending response on channel '%s'", msg.channel)

        elapsed = time.perf_counter() - start
        if runtime_events and isinstance(msg.metadata, dict):
            msg.metadata["_runtime_events"] = list(runtime_events)
        logger.info(
            "Bus handled message channel=%s chat_id=%s elapsed=%.2fs text_len=%d media=%d",
            msg.channel,
            msg.chat_id,
            elapsed,
            len(response.text or ""),
            len(response.media_paths or []),
        )
        if envelope.completion is not None and not envelope.completion.done():
            envelope.completion.set_result(response)
