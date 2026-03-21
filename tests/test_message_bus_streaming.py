from __future__ import annotations

import asyncio
from dataclasses import dataclass
import datetime as dt
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from babybot.channels.base import InboundMessage
from babybot.agent_kernel import ChildTaskEvent
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


class _RuntimeEventOrchestrator:
    def __init__(self) -> None:
        self.seen_runtime_callback = False

    async def process_task(
        self,
        user_input: str,
        chat_key: str = "",
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        stream_callback: Any = None,
        runtime_event_callback: Any = None,
    ) -> TaskResponse:
        del user_input, chat_key, heartbeat, media_paths, stream_callback
        if runtime_event_callback is not None:
            self.seen_runtime_callback = True
            await runtime_event_callback(
                ChildTaskEvent(
                    flow_id="flow-1",
                    task_id="task-1",
                    event="started",
                    payload={"resource_id": "skill.weather"},
                )
            )
        return TaskResponse(text="done")


class _ProgressEventOrchestrator:
    async def process_task(
        self,
        user_input: str,
        chat_key: str = "",
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        stream_callback: Any = None,
        runtime_event_callback: Any = None,
    ) -> TaskResponse:
        del user_input, chat_key, heartbeat, media_paths, stream_callback
        if runtime_event_callback is not None:
            await runtime_event_callback({
                "flow_id": "flow-1",
                "task_id": "task-1",
                "event": "progress",
                "payload": {
                    "resource_id": "skill.weather",
                    "description": "先查询杭州天气",
                    "status": "查询天气",
                    "progress": 0.5,
                },
            })
            await runtime_event_callback({
                "flow_id": "flow-1",
                "task_id": "task-1",
                "event": "succeeded",
                "payload": {
                    "resource_id": "skill.weather",
                    "description": "先查询杭州天气",
                    "output": "杭州多云 26℃",
                },
            })
        return TaskResponse(text="最终结果")


class _StreamingProgressEventOrchestrator:
    async def process_task(
        self,
        user_input: str,
        chat_key: str = "",
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        stream_callback: Any = None,
        runtime_event_callback: Any = None,
    ) -> TaskResponse:
        del user_input, chat_key, heartbeat, media_paths
        if stream_callback is not None:
            await stream_callback("处理中")
            await stream_callback("处理中，请稍候")
        if runtime_event_callback is not None:
            await runtime_event_callback({
                "flow_id": "flow-1",
                "task_id": "task-1",
                "event": "progress",
                "payload": {
                    "resource_id": "skill.weather",
                    "description": "先查询杭州天气",
                    "status": "查询天气",
                    "progress": 0.5,
                },
            })
            await runtime_event_callback({
                "flow_id": "flow-1",
                "task_id": "task-1",
                "event": "succeeded",
                "payload": {
                    "resource_id": "skill.weather",
                    "description": "先查询杭州天气",
                    "output": "杭州多云 26℃",
                },
            })
        return TaskResponse(text="最终结果")


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


def test_message_bus_captures_runtime_events_from_orchestrator_callback() -> None:
    channel = _FakeFeishuChannel(stream_reply=False)
    orchestrator = _RuntimeEventOrchestrator()
    metadata: dict[str, Any] = {}
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=orchestrator,  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
        metadata=metadata,
    )

    async def _run() -> TaskResponse:
        await bus.start()
        try:
            return await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    response = asyncio.run(_run())

    assert response.text == "done"
    assert orchestrator.seen_runtime_callback is True
    assert metadata["_runtime_events"] == [{
        "flow_id": "flow-1",
        "task_id": "task-1",
        "event": "started",
        "payload": {"resource_id": "skill.weather"},
    }]


def test_message_bus_sends_progress_reply_when_runtime_stage_completes() -> None:
    channel = _FakeFeishuChannel(stream_reply=False)
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=_ProgressEventOrchestrator(),  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
        metadata={},
    )

    async def _run() -> TaskResponse:
        await bus.start()
        try:
            return await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    response = asyncio.run(_run())

    assert response.text == "最终结果"
    assert len(channel.sent) == 3
    assert "处理中" in channel.sent[0].text
    assert "50%" in channel.sent[0].text
    assert channel.sent[1].text == "阶段完成：先查询杭州天气"
    assert channel.sent[2].text == "最终结果"


def test_message_bus_adds_request_received_at_metadata_for_downstream_tools() -> None:
    channel = _FakeFeishuChannel(stream_reply=False)
    observed: dict[str, Any] = {}

    class _MetadataOrchestrator:
        async def process_task(
            self,
            user_input: str,
            chat_key: str = "",
            heartbeat: Any = None,
            media_paths: list[str] | None = None,
            stream_callback: Any = None,
            runtime_event_callback: Any = None,
        ) -> TaskResponse:
            del user_input, chat_key, heartbeat, media_paths, stream_callback, runtime_event_callback
            from babybot.channels.tools import ChannelToolContext

            ctx = ChannelToolContext.get_current()
            observed.update((ctx.metadata or {}) if ctx is not None else {})
            return TaskResponse(text="ok")

    metadata: dict[str, Any] = {}
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=_MetadataOrchestrator(),  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
        metadata=metadata,
    )

    frozen_now = dt.datetime(2026, 3, 19, 11, 13, 10, tzinfo=dt.timezone(dt.timedelta(hours=8)))

    async def _run() -> TaskResponse:
        await bus.start()
        try:
            return await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    with patch("babybot.message_bus.datetime") as mock_datetime:
        mock_datetime.datetime.now.return_value = frozen_now
        mock_datetime.timezone = dt.timezone
        mock_datetime.timedelta = dt.timedelta
        asyncio.run(_run())

    assert observed["request_received_at"] == "2026-03-19T11:13:10+08:00"


def test_message_bus_sends_final_reply_as_new_message_after_runtime_progress() -> None:
    channel = _FakeFeishuChannel(stream_reply=True)
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=_StreamingProgressEventOrchestrator(),  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
        metadata={},
    )

    async def _run() -> TaskResponse:
        await bus.start()
        try:
            return await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    response = asyncio.run(_run())

    assert response.text == "最终结果"
    assert channel.created == ["处理中"]
    assert channel.patched == ["处理中，请稍候"]
    assert len(channel.sent) == 3
    assert "处理中" in channel.sent[0].text
    assert "50%" in channel.sent[0].text
    assert channel.sent[1].text == "阶段完成：先查询杭州天气"
    assert channel.sent[2].text == "最终结果"


def test_message_bus_dedupes_repeated_runtime_progress_messages() -> None:
    class _DuplicateProgressOrchestrator:
        async def process_task(
            self,
            user_input: str,
            chat_key: str = "",
            heartbeat: Any = None,
            media_paths: list[str] | None = None,
            stream_callback: Any = None,
            runtime_event_callback: Any = None,
        ) -> TaskResponse:
            del user_input, chat_key, heartbeat, media_paths, stream_callback
            if runtime_event_callback is not None:
                payload = {
                    "flow_id": "flow-1",
                    "task_id": "task-1",
                    "event": "progress",
                    "payload": {
                        "resource_id": "skill.weather",
                        "description": "下载模型",
                        "status": "下载模型",
                        "progress": 0.3,
                    },
                }
                await runtime_event_callback(payload)
                await runtime_event_callback(payload)
            return TaskResponse(text="done")

    channel = _FakeFeishuChannel(stream_reply=False)
    bus = MessageBus(
        config=_Config(),  # type: ignore[arg-type]
        orchestrator=_DuplicateProgressOrchestrator(),  # type: ignore[arg-type]
        channels={"feishu": channel},  # type: ignore[arg-type]
    )
    msg = InboundMessage(
        channel="feishu",
        sender_id="ou_user_1",
        chat_id="oc_chat_1",
        content="hello",
        metadata={},
    )

    async def _run() -> TaskResponse:
        await bus.start()
        try:
            return await bus.enqueue_and_wait(msg, timeout=3)
        finally:
            await bus.stop()

    asyncio.run(_run())

    assert len(channel.sent) == 2
    assert "30%" in channel.sent[0].text
    assert channel.sent[1].text == "done"
