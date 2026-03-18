"""Tests for internal child-task event emission in DynamicOrchestrator."""

from __future__ import annotations

import asyncio
from typing import Any

from babybot.agent_kernel import ExecutionContext, ModelRequest, ModelResponse, ModelToolCall
from babybot.agent_kernel.dynamic_orchestrator import DynamicOrchestrator, InMemoryChildTaskBus


class _DummyGateway:
    def __init__(self) -> None:
        self._call_idx = 0
        self._task_id = ""

    async def generate(
        self, request: ModelRequest, context: ExecutionContext,
    ) -> ModelResponse:
        del context
        if self._call_idx == 0:
            self._call_idx += 1
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c1",
                        name="dispatch_task",
                        arguments={
                            "resource_id": "skill.weather",
                            "description": "查询天气",
                        },
                    ),
                ),
                finish_reason="tool_calls",
            )
        if self._call_idx == 1:
            for message in request.messages:
                if message.role == "tool" and message.tool_call_id == "c1":
                    self._task_id = message.content
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


class _DummyResourceManager:
    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "skill.weather",
                "type": "skill",
                "name": "weather",
                "purpose": "天气查询",
                "group": "skill_weather",
                "tool_count": 1,
                "active": True,
            },
        ]

    def resolve_resource_scope(
        self, resource_id: str, require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        del require_tools
        if resource_id == "skill.weather":
            return {"include_groups": ["skill_weather"]}, ("weather",)
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
        del lease, agent_name, tape, tape_store, heartbeat, media_paths, skill_ids
        return f"done: {task_description}", []


def test_dynamic_orchestrator_emits_child_task_lifecycle_events() -> None:
    bus = InMemoryChildTaskBus()
    orchestrator = DynamicOrchestrator(
        resource_manager=_DummyResourceManager(),  # type: ignore[arg-type]
        gateway=_DummyGateway(),  # type: ignore[arg-type]
        child_task_bus=bus,
    )

    asyncio.run(orchestrator.run("查天气", ExecutionContext(session_id="flow-1")))

    events = bus.events_for("flow-1")
    assert [event.event for event in events] == ["queued", "started", "succeeded"]
    assert len({event.task_id for event in events}) == 1
