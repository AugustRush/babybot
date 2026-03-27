from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from babybot.agent_kernel import ExecutionContext, ModelRequest, ModelResponse, ModelToolCall
from babybot.agent_kernel.dynamic_orchestrator import DynamicOrchestrator
from babybot.orchestrator import OrchestratorAgent


class _RecordingPolicyStore:
    def __init__(self) -> None:
        self.recorded_decisions: list[dict[str, Any]] = []
        self.recorded_outcomes: list[dict[str, Any]] = []

    def record_decision(self, **payload: Any) -> None:
        self.recorded_decisions.append(dict(payload))

    def record_outcome(self, **payload: Any) -> None:
        self.recorded_outcomes.append(dict(payload))


class _PolicyGateway:
    def __init__(self) -> None:
        self._task_id = ""
        self._call_idx = 0

    async def generate(
        self, request: ModelRequest, context: ExecutionContext
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


class _PolicyResourceManager:
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
            }
        ]

    def resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
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
        memory_store: Any = None,
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        del lease, agent_name, tape, tape_store, memory_store, heartbeat, media_paths, skill_ids
        return f"done: {task_description}", []


def test_orchestrator_records_flow_level_outcome_on_success() -> None:
    agent = object.__new__(OrchestratorAgent)
    agent._policy_store = _RecordingPolicyStore()
    agent.config = SimpleNamespace(
        system=SimpleNamespace(
            policy_learning_enabled=True,
            context_history_tokens=2000,
            idle_timeout=30,
        )
    )
    agent.gateway = object()
    agent.resource_manager = object()
    agent._child_task_bus = None
    agent._task_heartbeat_registry = None
    agent.tape_store = None
    agent.memory_store = None

    class _Tape:
        chat_id = "feishu:c1"

    class _FakeDynamicOrchestrator:
        def __init__(self, resource_manager: Any, gateway: Any) -> None:
            del resource_manager, gateway

        async def run(self, goal: str, context: ExecutionContext):
            del goal
            context.emit("retrying", attempt=1)
            context.emit("dead_lettered", task_id="task-1")
            context.emit("stalled", task_id="task-2")
            context.emit(
                "policy_decision",
                decision_kind="scheduling",
                action_name="serial_dispatch",
                state_features={"deps_count": 0},
            )
            return SimpleNamespace(conclusion="完成", task_results={})

    from unittest.mock import patch

    with patch("babybot.orchestrator.DynamicOrchestrator", _FakeDynamicOrchestrator):
        text, media = asyncio.run(agent._answer_with_dag("查询天气", tape=_Tape()))

    assert text == "完成"
    assert media == []
    assert agent._policy_store.recorded_decisions[0]["decision_kind"] == "decomposition"
    assert agent._policy_store.recorded_outcomes[0]["final_status"] == "succeeded"
    assert agent._policy_store.recorded_outcomes[0]["outcome"]["retry_count"] == 1
    assert agent._policy_store.recorded_outcomes[0]["outcome"]["dead_letter_count"] == 1
    assert agent._policy_store.recorded_outcomes[0]["outcome"]["stalled_count"] == 1


def test_dynamic_orchestrator_records_dispatch_and_wait_events() -> None:
    context = ExecutionContext(session_id="flow-1", state={})
    orch = DynamicOrchestrator(
        resource_manager=_PolicyResourceManager(),  # type: ignore[arg-type]
        gateway=_PolicyGateway(),  # type: ignore[arg-type]
    )

    result = asyncio.run(orch.run("查询天气", context))

    assert result.conclusion == "完成"
    policy_events = [event for event in context.events if event["event"] == "policy_decision"]
    action_names = [event["action_name"] for event in policy_events]
    assert "serial_dispatch" in action_names
    assert "wait_barrier" in action_names
