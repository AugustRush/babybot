"""ChannelManager — central hub between channels and orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any

from ..config import Config
from ..orchestrator import OrchestratorAgent, TaskResponse
from .base import BaseChannel, InboundMessage
from .registry import discover_channels


class ChannelManager:
    """Discover, instantiate and manage channel lifecycle."""

    def __init__(self, config: Config, orchestrator: OrchestratorAgent) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.channels: dict[str, BaseChannel] = {}
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._init_channels()

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

    # ── Message handling ─────────────────────────────────────────────

    async def handle_message(self, msg: InboundMessage) -> None:
        """Route an inbound message through orchestrator → channel reply."""
        lock_key = f"{msg.channel}:{msg.chat_id}"
        lock = self._chat_locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            try:
                response = await self.orchestrator.process_task(msg.content)
            except Exception as e:
                response = TaskResponse(text=f"处理失败：{e}")

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
            print("No enabled channels found.")
            return
        tasks = [ch.start() for ch in self.channels.values()]
        await asyncio.gather(*tasks)

    async def stop_all(self) -> None:
        """Gracefully stop every running channel."""
        for ch in self.channels.values():
            try:
                await ch.stop()
            except Exception as e:
                print(f"Error stopping channel {ch.name}: {e}")
