from __future__ import annotations

import asyncio

import pytest

from babybot.orchestrator import OrchestratorAgent
from babybot.interactive_sessions.types import (
    InteractiveRequest,
    InteractiveReply,
    InteractiveSession,
    InteractiveSessionStatus,
)


class FakeSessionManager:
    def __init__(self, *, active_session: bool = False) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.send_calls = 0
        self.stop_all_calls = 0
        self.started_backend_name = ""
        self._active_session = active_session
        self.last_request: InteractiveRequest | None = None
        self.expired_on_send = False

    async def start(self, *, chat_key: str, backend_name: str) -> InteractiveSession:
        self.start_calls += 1
        self.started_backend_name = backend_name
        self._active_session = True
        return InteractiveSession(
            session_id="sess_1",
            chat_key=chat_key,
            backend_name=backend_name,
            started_at=1.0,
            last_active_at=1.0,
            handle={},
        )

    async def send(self, chat_key: str, message: str | InteractiveRequest) -> InteractiveReply:
        del chat_key
        self.send_calls += 1
        if isinstance(message, InteractiveRequest):
            self.last_request = message
            if self.expired_on_send:
                self._active_session = False
                return InteractiveReply(
                    text="当前交互会话已超时关闭，请重新使用 @session start <backend> 启动。",
                    expired=True,
                )
            return InteractiveReply(text="backend reply" if message.text else "")
        return InteractiveReply(text="backend reply" if message else "")

    async def stop(self, chat_key: str, reason: str = "user_stop") -> bool:
        del chat_key, reason
        self.stop_calls += 1
        self._active_session = False
        return True

    async def stop_all(self, reason: str = "reset") -> int:
        del reason
        self.stop_all_calls += 1
        self._active_session = False
        return 1

    def has_active_session(self, chat_key: str) -> bool:
        del chat_key
        return self._active_session

    def status(self, chat_key: str) -> InteractiveSessionStatus | None:
        del chat_key
        if not self._active_session:
            return None
        return InteractiveSessionStatus(
            session_id="sess_1",
            chat_key="feishu:c1",
            backend_name="claude",
            started_at=1.0,
            last_active_at=1.0,
            backend_status={"backend": "claude", "mode": "resident", "alive": True, "pid": 4321},
        )

    def summary(self) -> dict[str, object]:
        return {
            "active_count": 1 if self._active_session else 0,
            "sessions": (
                [{
                    "chat_key": "feishu:c1",
                    "backend_name": "claude",
                    "backend_status": {"mode": "resident", "alive": True, "pid": 4321},
                }]
                if self._active_session
                else []
            ),
        }


def make_agent_with_session_manager(*, active_session: bool = False) -> OrchestratorAgent:
    agent = object.__new__(OrchestratorAgent)
    agent._interactive_sessions = FakeSessionManager(active_session=active_session)
    agent._initialized = True
    agent._init_lock = asyncio.Lock()
    agent._handoff_locks = {}
    agent._recent_flow_ids_by_chat = {}
    agent._background_tasks = set()
    agent.resource_manager = None
    agent.gateway = None
    agent.tape_store = None
    agent.memory_store = None
    agent.gateway_calls = 0

    class _Config:
        system = type(
            "S",
            (),
            {
                "context_history_tokens": 2000,
                "context_compact_threshold": 3000,
                "context_max_chats": 100,
                "idle_timeout": 30,
                "interactive_session_max_age_seconds": 7200,
            },
        )()

    agent.config = _Config()

    async def _answer_with_dag(*args, **kwargs):
        del args, kwargs
        agent.gateway_calls += 1
        return "dag reply", []

    agent._answer_with_dag = _answer_with_dag
    return agent


@pytest.mark.asyncio
async def test_process_task_starts_interactive_session_on_command():
    agent = make_agent_with_session_manager()

    response = await agent.process_task("@session start claude", chat_key="feishu:c1")

    assert "Claude 会话已启动" in response.text
    assert agent._interactive_sessions.start_calls == 1


@pytest.mark.asyncio
async def test_process_task_routes_active_chat_messages_to_session_backend():
    agent = make_agent_with_session_manager(active_session=True)

    response = await agent.process_task("帮我看看 /models", chat_key="feishu:c1")

    assert response.text == "backend reply"
    assert agent.gateway_calls == 0


@pytest.mark.asyncio
async def test_session_stop_command_closes_active_session():
    agent = make_agent_with_session_manager(active_session=True)

    response = await agent.process_task("@session stop", chat_key="feishu:c1")

    assert "已关闭" in response.text
    assert agent._interactive_sessions.stop_calls == 1


@pytest.mark.asyncio
async def test_session_status_command_reports_active_session():
    agent = make_agent_with_session_manager(active_session=True)

    response = await agent.process_task("@session status", chat_key="feishu:c1")

    assert "当前交互会话：claude" in response.text
    assert "resident" in response.text
    assert "4321" in response.text


def test_get_status_includes_interactive_session_summary():
    agent = make_agent_with_session_manager(active_session=True)

    status = agent.get_status()

    assert "interactive_sessions" in status
    assert status["interactive_sessions"]["active_count"] == 1
    assert status["interactive_sessions"]["sessions"][0]["backend_status"]["mode"] == "resident"


def test_process_task_interactive_session_message_keeps_media_paths() -> None:
    agent = make_agent_with_session_manager(active_session=True)

    response = asyncio.run(
        agent.process_task(
            "看这张图",
            chat_key="feishu:c1",
            media_paths=["/tmp/demo.png"],
        )
    )

    assert response.text == "backend reply"
    assert agent._interactive_sessions.last_request is not None
    assert agent._interactive_sessions.last_request.media_paths == ("/tmp/demo.png",)


def test_process_task_falls_back_to_dag_when_session_expires_on_send() -> None:
    agent = make_agent_with_session_manager(active_session=True)
    agent._interactive_sessions.expired_on_send = True

    response = asyncio.run(
        agent.process_task("继续", chat_key="feishu:c1")
    )

    assert response.text == "dag reply"
    assert agent.gateway_calls == 1


def test_reset_stops_active_sessions():
    agent = make_agent_with_session_manager(active_session=True)

    agent.reset()

    assert agent._interactive_sessions.stop_all_calls == 1
