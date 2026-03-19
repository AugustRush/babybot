"""Tests for OrchestratorAgent routing via DynamicOrchestrator."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from babybot.agent_kernel import ExecutionContext, ModelRequest, ModelResponse, ModelToolCall
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
            }
        ]

    def resolve_resource_scope(
        self, resource_id: str, require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        if resource_id != "skill.weather-query":
            return None
        return {"include_groups": ["skill_weather_query"]}, ("weather-query",)

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
    agent._child_task_state_store = None
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
    assert call["lease"] == {"include_groups": ["skill_weather_query"]}
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
    assert seen["state_store"] is agent._child_task_state_store
    assert seen["task_stale_after_s"] == 30
