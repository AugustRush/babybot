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


class _Config:
    def __init__(self) -> None:
        self.system = _System(timeout=1)
        self.raw_config = {"channels": {}}

    def get_channel_config(self, name: str) -> Any:
        return None


class _Resource:
    def set_channel_context(self, ctx: Any) -> None:
        pass


class _Orchestrator:
    def __init__(self) -> None:
        self.resource_manager = _Resource()

    async def process_task(self, user_input: str) -> TaskResponse:
        await asyncio.sleep(2)
        return TaskResponse(text="late")


class _Channel:
    def __init__(self) -> None:
        self.responses: list[TaskResponse] = []

    async def send_response(self, chat_id: str, response: TaskResponse, **kwargs: Any) -> None:
        self.responses.append(response)


def test_channel_manager_returns_timeout_response() -> None:
    manager = ChannelManager(config=_Config(), orchestrator=_Orchestrator())
    fake_channel = _Channel()
    manager.channels["feishu"] = fake_channel

    msg = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="draw a parrot",
    )
    asyncio.run(manager.handle_message(msg))

    assert len(fake_channel.responses) == 1
    assert "超时" in fake_channel.responses[0].text
