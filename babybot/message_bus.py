"""MessageBus — async queue + semaphore based message processing."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .heartbeat import Heartbeat

if TYPE_CHECKING:
    from .channels.base import BaseChannel, InboundMessage
    from .channels.tools import ChannelToolContext
    from .config import Config
    from .orchestrator import OrchestratorAgent, TaskResponse

logger = logging.getLogger(__name__)


@dataclass
class MessageEnvelope:
    """Wrapper around an inbound message with tracking metadata."""

    message: InboundMessage
    heartbeat: Heartbeat
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
        self._queue: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()
        self._global_sem = asyncio.Semaphore(config.system.max_concurrency)
        self._chat_sems: dict[str, asyncio.Semaphore] = {}
        self._consumer_task: asyncio.Task | None = None
        self._running = False

    def _get_chat_sem(self, key: str) -> asyncio.Semaphore:
        sem = self._chat_sems.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self._config.system.max_per_chat)
            self._chat_sems[key] = sem
        return sem

    async def enqueue(self, msg: InboundMessage) -> None:
        """Non-blocking enqueue of an inbound message."""
        hb = Heartbeat(idle_timeout=float(self._config.system.idle_timeout))
        envelope = MessageEnvelope(message=msg, heartbeat=hb)
        await self._queue.put(envelope)

    async def start(self) -> None:
        """Start the consumer loop."""
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Graceful shutdown: signal consumer and wait."""
        self._running = False
        await self._queue.put(None)  # sentinel
        if self._consumer_task is not None:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._consumer_task.cancel()

    async def _consume_loop(self) -> None:
        """Main consumer: pull envelopes and dispatch."""
        while self._running:
            envelope = await self._queue.get()
            if envelope is None:
                break
            asyncio.create_task(self._dispatch(envelope))

    async def _dispatch(self, envelope: MessageEnvelope) -> None:
        """Acquire semaphores and process."""
        msg = envelope.message
        chat_key = f"{msg.channel}:{msg.chat_id}"

        async with self._global_sem:
            chat_sem = self._get_chat_sem(chat_key)
            async with chat_sem:
                await self._process(envelope)

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

        # Send instant acknowledgement if configured.
        if channel and self._config.system.send_ack:
            try:
                ack_response = TaskResponse(text="收到，正在处理...")
                await channel.send_response(
                    msg.chat_id,
                    ack_response,
                    sender_id=msg.sender_id,
                    metadata=msg.metadata,
                )
            except Exception:
                logger.warning("Failed to send ack for chat_id=%s", msg.chat_id, exc_info=True)

        # Set channel context for this message (contextvars — concurrency safe).
        ctx = ChannelToolContext(
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            metadata=msg.metadata,
        )
        ChannelToolContext.set_current(ctx)

        # hard_timeout is a safety net; idle_timeout handles "stuck" detection.
        # Ensure hard_timeout is at least 3x idle_timeout to avoid killing active tasks.
        hard_timeout = max(
            float(self._config.system.timeout),
            float(self._config.system.idle_timeout) * 3,
        )
        try:
            chat_key = f"{msg.channel}:{msg.chat_id}"
            response = await heartbeat.watch(
                self._orchestrator.process_task(msg.content, chat_key=chat_key, heartbeat=heartbeat),
                hard_timeout=hard_timeout,
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "Task timeout channel=%s chat_id=%s reason=%s",
                msg.channel,
                msg.chat_id,
                str(exc),
            )
            response = TaskResponse(
                text=(
                    f"任务执行超时（{self._config.system.idle_timeout}s 空闲 / "
                    f"{hard_timeout:.0f}s 上限）。"
                    "请尝试更具体的指令，或要求分步骤执行。"
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
                await channel.send_response(
                    msg.chat_id,
                    response,
                    sender_id=msg.sender_id,
                    metadata=msg.metadata,
                )
            except Exception:
                logger.exception("Error sending response on channel '%s'", msg.channel)

        elapsed = time.perf_counter() - start
        logger.info(
            "Bus handled message channel=%s chat_id=%s elapsed=%.2fs text_len=%d media=%d",
            msg.channel,
            msg.chat_id,
            elapsed,
            len(response.text or ""),
            len(response.media_paths or []),
        )
