"""Tests for DAG-driven orchestrator routing."""

from __future__ import annotations

import asyncio
from typing import Any

from babybot.agent_kernel.dag_ports import LLMPlanner, PlanOutput, PlannedTask
from babybot.orchestrator import OrchestratorAgent


class _FakeGateway:
    def __init__(self, plan: PlanOutput, merge_text: str = "merged") -> None:
        self._plan = plan
        self._merge_text = merge_text
        self.complete_calls: list[tuple[str, str]] = []

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: Any,
        heartbeat: Any = None,
    ) -> PlanOutput:
        return self._plan

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        heartbeat: Any = None,
        on_stream_text: Any = None,
    ) -> str:
        del on_stream_text
        self.complete_calls.append((system_prompt, user_prompt))
        return self._merge_text

    async def complete_messages(
        self,
        messages: Any,
        heartbeat: Any = None,
        on_stream_text: Any = None,
    ) -> str:
        del messages, heartbeat, on_stream_text
        return self._merge_text


class _FakeResourceManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_once = False

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
        self,
        resource_id: str,
        require_tools: bool = False,
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
        self.calls.append(
            {
                "task_description": task_description,
                "lease": lease,
                "agent_name": agent_name,
                "skill_ids": skill_ids,
            }
        )
        if self.fail_once and len(self.calls) == 1:
            raise RuntimeError("tool timeout")
        return "上海多云 16℃", []


class _ImageResourceManager(_FakeResourceManager):
    def get_resource_briefs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "skill.text-to-image",
                "type": "skill",
                "name": "text-to-image",
                "purpose": "文生图",
                "group": "skill_text_to_image",
                "tool_count": 1,
                "active": True,
            }
        ]

    def resolve_resource_scope(
        self,
        resource_id: str,
        require_tools: bool = False,
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        if resource_id != "skill.text-to-image":
            return None
        return {"include_groups": ["skill_text_to_image"]}, ("text-to-image",)


def test_planner_returns_direct_answer_without_tools() -> None:
    """When need_tools=false, the DAG produces a direct-answer task."""
    agent = object.__new__(OrchestratorAgent)
    gateway = _FakeGateway(
        PlanOutput(
            need_tools=False,
            direct_answer="这是直接回复",
            tasks=[],
        ),
        merge_text="这是直接回复",
    )
    agent.gateway = gateway
    agent.resource_manager = _FakeResourceManager()
    agent.tape_store = None
    agent.config = type("C", (), {"system": type("S", (), {"context_history_tokens": 2000})()})()

    text, media = asyncio.run(agent._answer_with_dag("你好"))
    assert text == "这是直接回复"
    assert media == []


def test_planner_dispatches_subagent_with_resource_scope() -> None:
    """When need_tools=true, tasks are dispatched via run_subagent_task."""
    resource = _FakeResourceManager()
    gateway = _FakeGateway(
        PlanOutput(
            need_tools=True,
            direct_answer="",
            tasks=[
                PlannedTask(
                    task_id="check_weather",
                    resource_id="skill.weather-query",
                    description="查询上海今日天气并给出建议",
                )
            ],
        ),
        merge_text="主Agent合并后的结果",
    )

    agent = object.__new__(OrchestratorAgent)
    agent.gateway = gateway
    agent.resource_manager = resource
    agent.tape_store = None
    agent.config = type("C", (), {"system": type("S", (), {"context_history_tokens": 2000})()})()

    text, media = asyncio.run(agent._answer_with_dag("上海今天天气怎么样"))

    # Single task → direct output, no merge
    assert text == "上海多云 16℃"
    assert media == []
    assert len(resource.calls) == 1
    call = resource.calls[0]
    assert call["lease"] == {"include_groups": ["skill_weather_query"]}
    assert call["skill_ids"] == ["weather-query"]


def test_dag_retries_failed_task() -> None:
    """WorkflowEngine retries failed tasks (default_retries=1)."""
    resource = _FakeResourceManager()
    resource.fail_once = True
    gateway = _FakeGateway(
        PlanOutput(
            need_tools=True,
            direct_answer="",
            tasks=[
                PlannedTask(
                    task_id="check_weather",
                    resource_id="skill.weather-query",
                    description="第一次尝试",
                )
            ],
        ),
        merge_text="重试后结果",
    )

    agent = object.__new__(OrchestratorAgent)
    agent.gateway = gateway
    agent.resource_manager = resource
    agent.tape_store = None
    agent.config = type("C", (), {"system": type("S", (), {"context_history_tokens": 2000})()})()

    text, media = asyncio.run(agent._answer_with_dag("查询天气"))

    # With default_retries=1, the engine retries once → second call succeeds
    assert text == "上海多云 16℃"
    assert len(resource.calls) == 2


def test_planner_heuristic_forces_image_skill_when_route_skips_tools() -> None:
    """Image generation heuristic overrides need_tools=false."""
    resource = _ImageResourceManager()
    gateway = _FakeGateway(
        PlanOutput(
            need_tools=False,
            direct_answer="",
            tasks=[],
        ),
        merge_text="已生成图片",
    )

    agent = object.__new__(OrchestratorAgent)
    agent.gateway = gateway
    agent.resource_manager = resource
    agent.tape_store = None
    agent.config = type("C", (), {"system": type("S", (), {"context_history_tokens": 2000})()})()

    text, media = asyncio.run(agent._answer_with_dag("画一只雄鹰"))
    # Image gen heuristic: the planner detects "画" and forces skill.text-to-image
    # But need_tools was false, so image heuristic kicks in at planner level
    # The result should use the image resource
    assert len(resource.calls) == 1
    assert resource.calls[0]["skill_ids"] == ["text-to-image"]
