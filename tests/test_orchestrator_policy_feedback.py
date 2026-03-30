from __future__ import annotations

import asyncio

import pytest

from babybot.orchestrator import OrchestratorAgent


class _FakePolicyStore:
    def __init__(self) -> None:
        self.feedback_rows: list[dict[str, str]] = []

    def record_feedback(
        self,
        *,
        flow_id: str,
        chat_key: str,
        rating: str,
        reason: str,
    ) -> None:
        self.feedback_rows.append(
            {
                "flow_id": flow_id,
                "chat_key": chat_key,
                "rating": rating,
                "reason": reason,
            }
        )

    def summarize_action_stats(
        self,
        *,
        decision_kind: str | None = None,
        state_bucket: str | None = None,
        now=None,
    ) -> dict[str, dict[str, float | int]]:
        del state_bucket, now
        if decision_kind == "scheduling":
            return {
                "serial_dispatch": {
                    "samples": 4,
                    "effective_samples": 3.2,
                    "mean_reward": 0.82,
                    "failure_rate": 0.0,
                }
            }
        return {}


def _make_agent() -> OrchestratorAgent:
    agent = object.__new__(OrchestratorAgent)
    agent._initialized = True
    agent._init_lock = asyncio.Lock()
    agent._interactive_sessions = None
    agent._recent_flow_ids_by_chat = {}
    agent._recent_flows_by_chat = {}
    agent._handoff_locks = {}
    agent._background_tasks = set()
    agent._policy_store = _FakePolicyStore()
    agent.resource_manager = None
    agent.gateway = None
    agent.tape_store = None
    agent.memory_store = None
    agent.config = type("Config", (), {"system": type("System", (), {})()})()
    return agent


@pytest.mark.asyncio
async def test_policy_feedback_command_records_explicit_user_rating() -> None:
    agent = _make_agent()
    agent._recent_flow_ids_by_chat["feishu:c1"] = "flow-1"

    response = await agent.process_task(
        "@policy feedback good 拆分合理",
        chat_key="feishu:c1",
    )

    assert "已记录" in response.text
    assert agent._policy_store.feedback_rows[0]["flow_id"] == "flow-1"
    assert agent._policy_store.feedback_rows[0]["rating"] == "good"
    assert agent._policy_store.feedback_rows[0]["reason"] == "拆分合理"


@pytest.mark.asyncio
async def test_policy_feedback_command_returns_clear_error_without_recent_flow() -> None:
    agent = _make_agent()

    response = await agent.process_task(
        "@policy feedback bad 没有最近任务",
        chat_key="feishu:c1",
    )

    assert "没有可反馈的最近任务" in response.text
    assert agent._policy_store.feedback_rows == []


@pytest.mark.asyncio
async def test_policy_inspect_command_reports_policy_summary() -> None:
    agent = _make_agent()

    response = await agent.process_task(
        "@policy inspect scheduling",
        chat_key="feishu:c1",
    )

    assert response.text.startswith("[Policy]")
    assert "decision_kind=scheduling" in response.text
    assert "action=serial_dispatch" in response.text


@pytest.mark.asyncio
async def test_policy_feedback_can_target_specific_flow() -> None:
    agent = _make_agent()
    agent._recent_flow_ids_by_chat["feishu:c1"] = "flow-new"
    agent._recent_flows_by_chat["feishu:c1"] = ["flow-new", "flow-old"]

    response = await agent.process_task(
        "@policy feedback flow-old bad 轮数失控",
        chat_key="feishu:c1",
    )

    assert "已记录" in response.text
    assert agent._policy_store.feedback_rows[0]["flow_id"] == "flow-old"


@pytest.mark.asyncio
async def test_policy_feedback_latest_requires_disambiguation_when_multiple_recent_flows() -> None:
    agent = _make_agent()
    agent._recent_flow_ids_by_chat["feishu:c1"] = "flow-new"
    agent._recent_flows_by_chat["feishu:c1"] = ["flow-new", "flow-old"]

    response = await agent.process_task(
        "@policy feedback latest good 很稳",
        chat_key="feishu:c1",
    )

    assert "请指定 flow_id" in response.text
    assert agent._policy_store.feedback_rows == []


def test_policy_choice_payload_includes_explain() -> None:
    agent = _make_agent()

    payload = agent.choose_scheduling_policy(
        features={
            "task_shape": "multi_step",
            "has_media": False,
            "independent_subtasks": 2,
        }
    )

    assert payload["action_name"] == "serial"
    assert payload["explain"]


def test_policy_feedback_can_target_specific_flow_sync() -> None:
    agent = _make_agent()
    agent._recent_flow_ids_by_chat["feishu:c1"] = "flow-new"
    agent._recent_flows_by_chat["feishu:c1"] = ["flow-new", "flow-old"]

    response = asyncio.run(
        agent.process_task(
            "@policy feedback flow-old bad 轮数失控",
            chat_key="feishu:c1",
        )
    )

    assert "已记录" in response.text
    assert agent._policy_store.feedback_rows[0]["flow_id"] == "flow-old"


def test_policy_feedback_latest_requires_disambiguation_sync() -> None:
    agent = _make_agent()
    agent._recent_flow_ids_by_chat["feishu:c1"] = "flow-new"
    agent._recent_flows_by_chat["feishu:c1"] = ["flow-new", "flow-old"]

    response = asyncio.run(
        agent.process_task(
            "@policy feedback latest good 很稳",
            chat_key="feishu:c1",
        )
    )

    assert "请指定 flow_id" in response.text
    assert agent._policy_store.feedback_rows == []
