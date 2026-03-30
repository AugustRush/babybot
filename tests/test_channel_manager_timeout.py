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

    def get_scheduled_tasks(self) -> list[dict[str, Any]]:
        return []


class _Resource:
    def set_channel_context(self, ctx: Any) -> None:
        pass


class _Orchestrator:
    def __init__(self) -> None:
        self.resource_manager = _Resource()

    async def process_task(self, user_input: str, chat_key: str = "", heartbeat: Any = None, media_paths: list[str] | None = None, runtime_event_callback: Any = None) -> TaskResponse:
        # Simulate a stuck task that never beats the heartbeat.
        del user_input, chat_key, media_paths
        if runtime_event_callback is not None:
            await runtime_event_callback(
                {
                    "job_id": "job-timeout-1",
                    "flow_id": "flow-timeout-1",
                    "task_id": "",
                    "event": "running",
                    "payload": {
                        "job_id": "job-timeout-1",
                        "state": "running",
                        "message": "still working",
                    },
                }
            )
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
    assert any("job-timeout-1" in r.text for r in fake_channel.responses)


def test_scheduled_task_message_skips_ack() -> None:
    config = _Config()
    config.system.send_ack = True

    class _FastOrchestrator(_Orchestrator):
        async def process_task(self, user_input: str, chat_key: str = "", heartbeat: Any = None, media_paths: list[str] | None = None) -> TaskResponse:
            if heartbeat:
                heartbeat.beat()
            return TaskResponse(text="done")

    manager = ChannelManager(config=config, orchestrator=_FastOrchestrator())
    fake_channel = _Channel()
    manager.channels["feishu"] = fake_channel

    msg = InboundMessage(
        channel="feishu",
        sender_id="scheduled:daily",
        chat_id="c1",
        content="run report",
        metadata={"scheduled_task": True, "scheduled_task_name": "daily"},
    )

    async def _run() -> None:
        await manager._bus.start()
        await manager.handle_message(msg)
        await asyncio.sleep(0.2)
        await manager._bus.stop()

    asyncio.run(_run())

    assert len(fake_channel.responses) == 1
    assert fake_channel.responses[0].text == "done"
