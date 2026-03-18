from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from babybot.channels.base import InboundMessage
from babybot.message_bus import MessageBus
from babybot.orchestrator import TaskResponse


@dataclass
class _System:
    timeout: int = 30
    idle_timeout: int = 30
    max_concurrency: int = 1
    max_per_chat: int = 1
    send_ack: bool = False
    message_queue_maxsize: int = 100
    scheduled_max_concurrency: int = 0


class _Config:
    def __init__(self) -> None:
        self.system = _System()


class _StreamingOrchestrator:
    async def process_task(
        self,
        user_input: str,
        chat_key: str = "",
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        stream_callback: Any = None,
    ) -> TaskResponse:
        del user_input, chat_key, heartbeat, media_paths
        if stream_callback is not None:
            await stream_callback("你")
            await stream_callback("你好")
        return TaskResponse(text="你好")


class _FakeFeishuChannel:
    def __init__(self, stream_reply: bool) -> None:
        self.config = SimpleNamespace(stream_reply=stream_reply)
        self.created: list[str] = []
        self.patched: list[str] = []
        self.sent: list[TaskResponse] = []

    async def create_stream_message(
        self,
        chat_id: str,
        text: str,
        *,
        sender_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        del chat_id, sender_id, metadata
        self.created.append(text)
        return "om_stream_1"

    async def patch_stream_message(self, message_id: str, text: str) -> bool:
        assert message_id == "om_stream_1"
        self.patched.append(text)
        return True

    async def send_response(
        self, chat_id: str, response: TaskResponse, **kwargs: Any
    ) -> None:
        del chat_id, kwargs
        self.sent.append(response)


def test_message_bus_uses_stream_callback_and_skips_duplicate_text_send() -> None:
    channel = _FakeFeishuChannel(stream_reply=True)
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=_StreamingOrchestrator(),  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
    )

    async def _run() -> TaskResponse:
        await bus.start()
        try:
            return await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    response = asyncio.run(_run())

    assert response.text == "你好"
    assert channel.created == ["你"]
    assert channel.patched[-1] == "你好"
    assert channel.sent == []


def test_message_bus_falls_back_to_normal_send_when_stream_disabled() -> None:
    channel = _FakeFeishuChannel(stream_reply=False)
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=_StreamingOrchestrator(),  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
    )

    async def _run() -> None:
        await bus.start()
        try:
            await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    asyncio.run(_run())

    assert channel.created == []
    assert channel.patched == []
    assert len(channel.sent) == 1
    assert channel.sent[0].text == "你好"
