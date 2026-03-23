"""Tests for OrchestratorAgent routing via DynamicOrchestrator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from babybot.agent_kernel import (
    ExecutionContext,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from babybot.agent_kernel.dynamic_orchestrator import InMemoryChildTaskBus
from babybot.context import TapeStore
from babybot.memory_store import HybridMemoryStore
from babybot.orchestrator import OrchestratorAgent


class _FakeGateway:
    """Returns scripted ModelResponse for generate() calls."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_idx = 0

    async def generate(
        self, request: ModelRequest, context: ExecutionContext,
    ) -> ModelResponse:
        if self._call_idx >= len(self._responses):
            return ModelResponse(text="(no more responses)")
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


class _FakeResourceManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "skill.weather-query",
                "type": "skill",
                "name": "weather-query",
                "purpose": "天气查询",
                "group": "skill_weather_query",
                "tool_count": 1,
                "active": True,
            },
            {
                "id": "group.browser",
                "type": "group",
                "name": "browser",
                "purpose": "网页浏览",
                "group": "browser",
                "tool_count": 1,
                "active": True,
            },
        ]

    def resolve_resource_scope(
        self, resource_id: str, require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        del require_tools
        if resource_id == "skill.weather-query":
            return {"include_groups": ["skill_weather_query"]}, ("weather-query",)
        if resource_id == "group.browser":
            return {"include_groups": ["browser"]}, ()
        return None

    async def run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: Any = None,
        tape_store: Any = None,
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        self.calls.append({
            "task_description": task_description,
            "lease": lease,
            "agent_name": agent_name,
            "skill_ids": skill_ids,
        })
        return "上海多云 16℃", []


def _make_agent(gateway: _FakeGateway, rm: _FakeResourceManager) -> OrchestratorAgent:
    agent = object.__new__(OrchestratorAgent)
    agent.gateway = gateway
    agent.resource_manager = rm
    agent.tape_store = None
    agent._child_task_bus = None
    agent._task_heartbeat_registry = None
    agent.config = type("C", (), {
        "system": type("S", (), {"context_history_tokens": 2000, "idle_timeout": 30})(),
    })()
    return agent


def test_direct_reply_without_tools() -> None:
    """Model calls reply_to_user directly for simple questions."""
    gateway = _FakeGateway([
        ModelResponse(
            text="",
            tool_calls=(
                ModelToolCall(
                    call_id="c1",
                    name="reply_to_user",
                    arguments={"text": "你好！有什么可以帮你的？"},
                ),
            ),
            finish_reason="tool_calls",
        ),
    ])
    rm = _FakeResourceManager()
    agent = _make_agent(gateway, rm)
    text, media = asyncio.run(agent._answer_with_dag("你好"))
    assert text == "你好！有什么可以帮你的？"
    assert media == []
    assert len(rm.calls) == 0


def test_dispatch_subagent_with_resource_scope() -> None:
    """Model dispatches a task, waits, and replies with result."""

    class SmartGateway(_FakeGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_id = ""

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="c1",
                            name="dispatch_task",
                            arguments={
                                "resource_id": "skill.weather-query",
                                "description": "查询上海今日天气",
                            },
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="c2",
                            name="wait_for_tasks",
                            arguments={"task_ids": [self._task_id]},
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c3",
                        name="reply_to_user",
                        arguments={"text": "上海多云 16℃"},
                    ),
                ),
                finish_reason="tool_calls",
            )

    rm = _FakeResourceManager()
    agent = _make_agent(SmartGateway(), rm)
    text, media = asyncio.run(agent._answer_with_dag("上海今天天气怎么样"))
    assert text == "上海多云 16℃"
    assert media == []
    assert len(rm.calls) == 1
    call = rm.calls[0]
    assert call["lease"]["include_groups"] == ["skill_weather_query"]
    assert call["skill_ids"] == ["weather-query"]


def test_dispatch_subagent_with_multiple_resource_scopes() -> None:
    class SmartGateway(_FakeGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_id = ""

        async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
            if self._call_idx == 0:
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="c1",
                            name="dispatch_task",
                            arguments={
                                "resource_ids": ["skill.weather-query", "group.browser"],
                                "description": "查看网页并查询上海今日天气",
                            },
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if self._call_idx == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_id = msg.content
                self._call_idx += 1
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="c2",
                            name="wait_for_tasks",
                            arguments={"task_ids": [self._task_id]},
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            self._call_idx += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c3",
                        name="reply_to_user",
                        arguments={"text": "完成"},
                    ),
                ),
                finish_reason="tool_calls",
            )

    rm = _FakeResourceManager()
    agent = _make_agent(SmartGateway(), rm)
    text, media = asyncio.run(agent._answer_with_dag("查看网页并查询天气"))
    assert text == "完成"
    assert media == []
    assert len(rm.calls) == 1
    call = rm.calls[0]
    assert set(call["lease"]["include_groups"]) == {"skill_weather_query", "browser"}
    assert call["skill_ids"] == ["weather-query"]


def test_plain_text_response() -> None:
    """Model responds with plain text (no tool calls)."""
    gateway = _FakeGateway([ModelResponse(text="直接文本回复")])
    rm = _FakeResourceManager()
    agent = _make_agent(gateway, rm)
    text, media = asyncio.run(agent._answer_with_dag("你好"))
    assert text == "直接文本回复"
    assert media == []


def test_answer_with_dag_passes_stream_callback_into_context() -> None:
    rm = _FakeResourceManager()
    agent = _make_agent(_FakeGateway([]), rm)
    seen: dict[str, Any] = {}

    class _FakeDynamicOrchestrator:
        def __init__(self, resource_manager: Any, gateway: Any) -> None:
            del resource_manager, gateway

        async def run(self, goal: str, context: ExecutionContext):
            del goal
            seen.update(context.state)
            return type("R", (), {"conclusion": "ok"})()

    stream_callback = lambda text: text

    with patch("babybot.orchestrator.DynamicOrchestrator", _FakeDynamicOrchestrator):
        text, media = asyncio.run(agent._answer_with_dag("你好", stream_callback=stream_callback))

    assert text == "ok"
    assert media == []
    assert seen["stream_callback"] is stream_callback


def test_answer_with_dag_passes_runtime_event_callback_and_shared_runtime_adapters() -> None:
    rm = _FakeResourceManager()
    agent = _make_agent(_FakeGateway([]), rm)
    agent._child_task_bus = object()
    agent._task_heartbeat_registry = object()
    agent._child_task_state_store = object()
    seen: dict[str, Any] = {}

    class _FakeDynamicOrchestrator:
        def __init__(
            self,
            resource_manager: Any,
            gateway: Any,
            child_task_bus: Any = None,
            task_heartbeat_registry: Any = None,
            state_store: Any = None,
            task_stale_after_s: Any = None,
        ) -> None:
            del resource_manager, gateway
            seen["child_task_bus"] = child_task_bus
            seen["task_heartbeat_registry"] = task_heartbeat_registry
            seen["state_store"] = state_store
            seen["task_stale_after_s"] = task_stale_after_s

        async def run(self, goal: str, context: ExecutionContext):
            del goal
            seen.update(context.state)
            return type("R", (), {"conclusion": "ok"})()

    runtime_event_callback = lambda event: event

    with patch("babybot.orchestrator.DynamicOrchestrator", _FakeDynamicOrchestrator):
        text, media = asyncio.run(
            agent._answer_with_dag("你好", runtime_event_callback=runtime_event_callback)
        )

    assert text == "ok"
    assert media == []
    assert seen["runtime_event_callback"] is runtime_event_callback
    assert seen["child_task_bus"] is agent._child_task_bus
    assert seen["task_heartbeat_registry"] is agent._task_heartbeat_registry
    assert seen["state_store"] is None
    assert seen["task_stale_after_s"] == 30


def test_answer_with_dag_passes_orchestrator_max_steps() -> None:
    rm = _FakeResourceManager()
    agent = _make_agent(_FakeGateway([]), rm)
    agent.config.system.orchestrator_max_steps = 12
    seen: dict[str, Any] = {}

    class _FakeDynamicOrchestrator:
        def __init__(self, resource_manager: Any, gateway: Any, max_steps: int | None = None) -> None:
            del resource_manager, gateway
            seen["max_steps"] = max_steps

        async def run(self, goal: str, context: ExecutionContext):
            del goal, context
            return type("R", (), {"conclusion": "ok"})()

    with patch("babybot.orchestrator.DynamicOrchestrator", _FakeDynamicOrchestrator):
        text, media = asyncio.run(agent._answer_with_dag("你好"))

    assert text == "ok"
    assert media == []
    assert seen["max_steps"] == 12


def test_consecutive_answer_with_dag_calls_do_not_replay_prior_runtime_events() -> None:
    class _RepeatableDispatchGateway(_FakeGateway):
        def __init__(self) -> None:
            super().__init__([])
            self._task_id = ""

        async def generate(
            self, request: ModelRequest, context: ExecutionContext,
        ) -> ModelResponse:
            del context
            step = self._call_idx % 3
            self._call_idx += 1
            if step == 0:
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="c1",
                            name="dispatch_task",
                            arguments={
                                "resource_id": "skill.weather-query",
                                "description": "查询杭州天气",
                            },
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            if step == 1:
                for msg in request.messages:
                    if msg.role == "tool" and msg.tool_call_id == "c1":
                        self._task_id = msg.content
                return ModelResponse(
                    text="",
                    tool_calls=(
                        ModelToolCall(
                            call_id="c2",
                            name="wait_for_tasks",
                            arguments={"task_ids": [self._task_id]},
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c3",
                        name="reply_to_user",
                        arguments={"text": "杭州多云 16℃"},
                    ),
                ),
                finish_reason="tool_calls",
            )

    async def _capture(events: list[dict[str, Any]], event: dict[str, Any]) -> None:
        events.append(dict(event))

    rm = _FakeResourceManager()
    agent = _make_agent(_RepeatableDispatchGateway(), rm)
    agent._child_task_bus = InMemoryChildTaskBus()

    first_events: list[dict[str, Any]] = []
    second_events: list[dict[str, Any]] = []

    asyncio.run(agent._answer_with_dag(
        "先告诉我杭州天气",
        runtime_event_callback=lambda event: _capture(first_events, event),
    ))
    asyncio.run(agent._answer_with_dag(
        "再告诉我杭州天气",
        runtime_event_callback=lambda event: _capture(second_events, event),
    ))

    assert [event["event"] for event in first_events] == ["queued", "started", "succeeded"]
    assert [event["event"] for event in second_events] == ["queued", "started", "succeeded"]
    assert first_events[0]["flow_id"] != second_events[0]["flow_id"]


def test_process_task_persists_runtime_events_to_tape(tmp_path: Path) -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._initialized = True
    agent._init_lock = asyncio.Lock()
    agent.resource_manager = object()
    agent.gateway = object()
    agent.tape_store = TapeStore(db_path=tmp_path / "context.db")
    agent._handoff_locks = {}

    class _Config:
        system = type(
            "S",
            (),
            {
                "context_history_tokens": 2000,
                "context_compact_threshold": 999999,
                "context_max_chats": 100,
                "idle_timeout": 30,
            },
        )()

    agent.config = _Config()

    async def _answer_with_dag(
        user_input: str,
        tape=None,
        heartbeat=None,
        media_paths=None,
        stream_callback=None,
        runtime_event_callback=None,
    ):
        del user_input, tape, heartbeat, media_paths, stream_callback
        if runtime_event_callback is not None:
            await runtime_event_callback(
                {
                    "flow_id": "flow-1",
                    "task_id": "task-1",
                    "event": "started",
                    "payload": {"resource_id": "skill.weather", "description": "查询天气"},
                }
            )
        return "done", []

    agent._answer_with_dag = _answer_with_dag

    response = asyncio.run(agent.process_task("查询天气", chat_key="feishu:chat-1"))

    assert response.text == "done"
    tape = agent.tape_store.get_or_create("feishu:chat-1")
    event_entries = [entry for entry in tape.entries if entry.kind == "event"]
    assert len(event_entries) == 1
    assert event_entries[0].payload["event"] == "started"
    assert event_entries[0].payload["payload"]["resource_id"] == "skill.weather"


def test_process_task_updates_memory_from_assistant_reply_and_success_events(tmp_path: Path) -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._initialized = True
    agent._init_lock = asyncio.Lock()
    agent.resource_manager = object()
    agent.gateway = object()
    agent.tape_store = TapeStore(db_path=tmp_path / "context.db")
    agent.memory_store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    agent.memory_store.ensure_bootstrap()
    agent._handoff_locks = {}

    class _Config:
        system = type(
            "S",
            (),
            {
                "context_history_tokens": 2000,
                "context_compact_threshold": 999999,
                "context_max_chats": 100,
                "idle_timeout": 30,
            },
        )()

    agent.config = _Config()

    async def _answer_with_dag(
        user_input: str,
        tape=None,
        heartbeat=None,
        media_paths=None,
        stream_callback=None,
        runtime_event_callback=None,
    ):
        del user_input, tape, heartbeat, media_paths, stream_callback
        if runtime_event_callback is not None:
            await runtime_event_callback(
                {
                    "flow_id": "flow-1",
                    "task_id": "task-1",
                    "event": "succeeded",
                    "payload": {
                        "resource_id": "skill.audio",
                        "description": "生成语音",
                        "output": "已生成 speech.wav",
                    },
                }
            )
        return "好的，后续默认中文并保持简洁，我会继续作为你的代码架构助手协助你。", []

    agent._answer_with_dag = _answer_with_dag

    response = asyncio.run(agent.process_task("继续修复语音", chat_key="feishu:chat-1"))

    assert response.text.startswith("好的，后续默认中文")
    records = agent.memory_store.list_memories(chat_id="feishu:chat-1")
    summaries = "\n".join(record.summary for record in records)
    keys = {(record.memory_type, record.key, str(record.value)) for record in records}

    assert ("relationship_policy", "default_language", "zh-CN") in keys
    assert ("relationship_policy", "response_style", "concise") in keys
    assert ("relationship_policy", "assistant_role", "代码架构助手") in keys
    assert "生成语音" in summaries
    assert "speech.wav" in summaries


def test_maybe_handoff_writes_extended_anchor_state(tmp_path: Path) -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._handoff_locks = {}
    agent.tape_store = TapeStore(db_path=tmp_path / "context.db")

    class _Gateway:
        async def complete(self, prompt: str, history_text: str) -> str:
            del prompt, history_text
            return json.dumps(
                {
                    "summary": "用户正在修改小猪图片",
                    "entities": ["小猪", "背景"],
                    "user_intent": "继续编辑图片",
                    "pending": "等待确认颜色",
                    "next_steps": ["把小猪改成白色"],
                    "artifacts": ["pig.png"],
                    "open_questions": ["是否保留蓝色背景"],
                    "decisions": ["继续沿用之前生成的图片"],
                },
                ensure_ascii=False,
            )

    class _Config:
        system = type("S", (), {"context_compact_threshold": 1})()

    agent.gateway = _Gateway()
    agent.config = _Config()

    tape = agent.tape_store.get_or_create("feishu:chat-1")
    start = tape.append("anchor", {"name": "session/start", "state": {}})
    user = tape.append("message", {"role": "user", "content": "画一只小猪"})
    assistant = tape.append("message", {"role": "assistant", "content": "已生成黑色小猪"})
    agent.tape_store.save_entries("feishu:chat-1", [start, user, assistant])

    asyncio.run(agent._maybe_handoff(tape, "feishu:chat-1"))

    anchor = tape.last_anchor()
    assert anchor is not None
    state = anchor.payload["state"]
    assert state["summary"] == "用户正在修改小猪图片"
    assert state["next_steps"] == ["把小猪改成白色"]
    assert state["artifacts"] == ["pig.png"]
    assert state["open_questions"] == ["是否保留蓝色背景"]
    assert state["decisions"] == ["继续沿用之前生成的图片"]


def test_inspect_runtime_flow_uses_stable_sectioned_format() -> None:
    from babybot.agent_kernel.dynamic_orchestrator import ChildTaskEvent, InMemoryChildTaskBus
    from babybot.heartbeat import TaskHeartbeatRegistry

    agent = object.__new__(OrchestratorAgent)
    agent._recent_flow_ids_by_chat = {"feishu:chat-1": "orchestrator:flow-1"}
    agent._task_heartbeat_registry = TaskHeartbeatRegistry()
    agent._child_task_bus = InMemoryChildTaskBus()

    agent._task_heartbeat_registry.beat(
        "orchestrator:flow-1",
        "task_1",
        status="下载模型",
        progress=0.5,
    )
    asyncio.run(
        agent._child_task_bus.publish(
            ChildTaskEvent(
                flow_id="orchestrator:flow-1",
                task_id="task_1",
                event="progress",
                payload={"description": "下载模型", "status": "下载模型", "progress": 0.5},
            )
        )
    )

    text = agent.inspect_runtime_flow(chat_key="feishu:chat-1")

    assert text.startswith("[Runtime Flow]")
    assert "flow_id=orchestrator:flow-1" in text
    assert "chat_key=feishu:chat-1" in text
    assert "[Tasks]" in text
    assert "[Recent Events]" in text
    assert "task_id=task_1" in text
    assert "event=progress" in text


def test_remember_flow_id_uses_lru_eviction() -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._recent_flow_ids_by_chat = {f"chat-{idx}": f"flow-{idx}" for idx in range(256)}

    agent._remember_flow_id("chat-0", "flow-0-new")
    agent._remember_flow_id("chat-new", "flow-new")

    assert agent._recent_flow_ids_by_chat["chat-0"] == "flow-0-new"
    assert "chat-1" not in agent._recent_flow_ids_by_chat
    assert agent._recent_flow_ids_by_chat["chat-new"] == "flow-new"


def test_get_handoff_lock_uses_lru_eviction() -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._handoff_locks = {f"chat-{idx}": asyncio.Lock() for idx in range(256)}

    recent = agent._get_handoff_lock("chat-0")
    added = agent._get_handoff_lock("chat-new")

    assert recent is agent._handoff_locks["chat-0"]
    assert added is agent._handoff_locks["chat-new"]
    assert "chat-1" not in agent._handoff_locks


def test_inspect_chat_context_uses_stable_sectioned_format(tmp_path: Path) -> None:
    from babybot.memory_store import HybridMemoryStore

    agent = object.__new__(OrchestratorAgent)
    agent.memory_store = HybridMemoryStore(
        db_path=tmp_path / "context.db",
        memory_dir=tmp_path / "memory",
    )
    agent.memory_store.ensure_bootstrap()
    agent.tape_store = TapeStore(db_path=tmp_path / "context.db")

    tape = agent.tape_store.get_or_create("feishu:chat-1")
    tape.append("anchor", {"name": "compact/1", "state": {"summary": "用户在处理语音问题"}})
    agent.memory_store.observe_user_message("feishu:chat-1", "以后默认中文，回答简洁")
    agent.memory_store.observe_anchor_state(
        "feishu:chat-1",
        {"pending": "继续处理语音失败"},
        source_ids=[1],
    )

    text = agent.inspect_chat_context("feishu:chat-1", query="继续语音任务")

    assert text.startswith("[Chat Context]")
    assert "chat_key=feishu:chat-1" in text
    assert "query=继续语音任务" in text
    assert "[Hot Context]" in text
    assert "[Warm Context]" in text
    assert "[Memory Records]" in text
    assert "[Tape Summary]" in text



def test_spawn_background_task_logs_exception_and_releases_reference(caplog: pytest.LogCaptureFixture) -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._background_tasks = set()

    async def _boom() -> None:
        raise RuntimeError("handoff boom")

    async def _run() -> None:
        agent._spawn_background_task(_boom(), label="handoff")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with caplog.at_level("ERROR"):
        asyncio.run(_run())

    assert agent._background_tasks == set()
    assert "Background task failed" in caplog.text
