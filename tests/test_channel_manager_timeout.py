from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from babybot.channels.base import InboundMessage
from babybot.channels.manager import ChannelManager
from babybot.orchestrator import TaskResponse


@dataclass
class _System:
    timeout: int = 1
    idle_timeout: int = 1
    max_concurrency: int = 2
    max_per_chat: int = 1
    send_ack: bool = False


class _Config:
    def __init__(self) -> None:
        self.system = _System()
        self.raw_config = {"channels": {}}

    def get_channel_config(self, name: str) -> Any:
        return None


class _Resource:
    def set_channel_context(self, ctx: Any) -> None:
        pass


class _Orchestrator:
    def __init__(self) -> None:
        self.resource_manager = _Resource()

    async def process_task(self, user_input: str, chat_key: str = "", heartbeat: Any = None, media_paths: list[str] | None = None) -> TaskResponse:
        # Simulate a stuck task that never beats the heartbeat.
        await asyncio.sleep(10)
        return TaskResponse(text="late")


class _Channel:
    def __init__(self) -> None:
        self.responses: list[TaskResponse] = []

    async def send_response(self, chat_id: str, response: TaskResponse, **kwargs: Any) -> None:
        self.responses.append(response)


def test_channel_manager_returns_timeout_response() -> None:
    config = _Config()
    manager = ChannelManager(config=config, orchestrator=_Orchestrator())
    fake_channel = _Channel()
    manager.channels["feishu"] = fake_channel

    msg = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="draw a parrot",
    )

    async def _run() -> None:
        await manager._bus.start()
        await manager.handle_message(msg)
        # Give the bus time to dispatch and process (with timeout).
        await asyncio.sleep(3)
        await manager._bus.stop()

    asyncio.run(_run())

    assert len(fake_channel.responses) >= 1
    # Should contain a timeout response.
    assert any("超时" in r.text for r in fake_channel.responses)
