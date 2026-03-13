"""ChannelManager — central hub between channels and orchestrator."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import Config
from ..orchestrator import OrchestratorAgent, TaskResponse
from .base import BaseChannel, InboundMessage
from .registry import discover_channels
from .tools import ChannelToolContext

logger = logging.getLogger(__name__)


class ChannelManager:
    """Discover, instantiate and manage channel lifecycle."""

    def __init__(self, config: Config, orchestrator: OrchestratorAgent) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.channels: dict[str, BaseChannel] = {}
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._init_channels()
        self._register_channel_tools()

    # ── Bootstrap ────────────────────────────────────────────────────

    def _init_channels(self) -> None:
        """Auto-discover channel classes and instantiate enabled ones."""
        channels_conf = self.config.raw_config.get("channels", {})
        available = discover_channels()

        for name, channel_cls in available.items():
            section = channels_conf.get(name, {})
            if not section.get("enabled", False):
                continue
            ch_config = self.config.get_channel_config(name)
            if ch_config is None:
                continue
            self.channels[name] = channel_cls(config=ch_config, manager=self)

    def _register_channel_tools(self) -> None:
        """Register channel tools with the resource manager."""
        for name, channel in self.channels.items():
            channel_tools = getattr(channel, "get_channel_tools", None)
            if callable(channel_tools):
                tools = channel_tools()
                if tools:
                    self.orchestrator.resource_manager.register_channel_tools(tools)
                    logger.info("Registered channel tools for '%s'", name)

    # ── Message handling ─────────────────────────────────────────────

    async def handle_message(self, msg: InboundMessage) -> None:
        """Route an inbound message through orchestrator → channel reply."""
        lock_key = f"{msg.channel}:{msg.chat_id}"
        lock = self._chat_locks.setdefault(lock_key, asyncio.Lock())

        ctx = ChannelToolContext(
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            metadata=msg.metadata,
        )
        self.orchestrator.resource_manager.set_channel_context(ctx)

        async with lock:
            try:
                response = await self.orchestrator.process_task(msg.content)
            except Exception as e:
                logger.error("Error processing task: %s", e)
                response = TaskResponse(text=f"处理失败：{e}")
            finally:
                self.orchestrator.resource_manager.set_channel_context(None)

        channel = self.channels.get(msg.channel)
        if channel:
            await channel.send_response(
                msg.chat_id,
                response,
                sender_id=msg.sender_id,
                metadata=msg.metadata,
            )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start every enabled channel concurrently."""
        if not self.channels:
            logger.info("No enabled channels found")
            return
        tasks = [ch.start() for ch in self.channels.values()]
        await asyncio.gather(*tasks)

    async def stop_all(self) -> None:
        """Gracefully stop every running channel."""
        for ch in self.channels.values():
            try:
                await ch.stop()
            except Exception as e:
                logger.warning("Error stopping channel %s: %s", ch.name, e)
