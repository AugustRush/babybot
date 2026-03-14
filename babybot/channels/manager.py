"""ChannelManager — central hub between channels and orchestrator."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import Config
from ..message_bus import MessageBus
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
        self._init_channels()
        self._register_channel_tools()
        self._bus = MessageBus(
            config=config,
            orchestrator=orchestrator,
            channels=self.channels,
        )

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
        """Enqueue an inbound message for async processing."""
        logger.info(
            "Inbound message channel=%s chat_id=%s sender_id=%s message_id=%s content=%s",
            msg.channel,
            msg.chat_id,
            msg.sender_id,
            (msg.metadata.get("message_id") if isinstance(msg.metadata, dict) else ""),
            (msg.content or "")[:120],
        )
        await self._bus.enqueue(msg)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start every enabled channel and the message bus."""
        await self._bus.start()
        if not self.channels:
            logger.info("No enabled channels found")
            return
        tasks = [ch.start() for ch in self.channels.values()]
        await asyncio.gather(*tasks)

    async def stop_all(self) -> None:
        """Gracefully stop the message bus and every running channel."""
        await self._bus.stop()
        for ch in self.channels.values():
            try:
                await ch.stop()
            except Exception as e:
                logger.warning("Error stopping channel %s: %s", ch.name, e)
