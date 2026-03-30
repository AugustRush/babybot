from __future__ import annotations

import pytest

from babybot.task_contract import (
    TaskContract,
    assert_runtime_matches_contract,
    build_task_contract,
)


def test_build_task_contract_keeps_single_round_budget() -> None:
    contract = build_task_contract(
        user_input="两个专家讨论，一轮定胜负，直接给我最终答案。",
        chat_key="feishu:c1",
    )

    assert contract == TaskContract(
        chat_key="feishu:c1",
        goal="两个专家讨论，一轮定胜负，直接给我最终答案。",
        mode="debate",
        deliverable="final_answer",
        round_budget=1,
        termination_rule="single_round",
        allow_clarification=False,
        allowed_tools=("dispatch_team", "reply_to_user"),
        allowed_agents=(),
        metadata={},
    )


def test_build_task_contract_uses_normalized_execution_constraints() -> None:
    contract = build_task_contract(
        user_input="请比较两种方案。",
        chat_key="feishu:c1",
        execution_constraints={
            "mode": "interactive",
            "hard_limits": {
                "max_rounds": 2,
                "max_total_seconds": 600,
            },
            "soft_preferences": {"resolution_style": "balanced"},
            "degradation": {"on_budget_exhausted": "summarize_partial"},
        },
    )

    assert contract.mode == "debate"
    assert contract.round_budget == 2
    assert contract.metadata["execution_constraints"]["hard_limits"]["max_rounds"] == 2


def test_assert_runtime_matches_contract_rejects_round_budget_drift() -> None:
    contract = build_task_contract(
        user_input="两个专家讨论，一轮定胜负。",
        chat_key="feishu:c1",
    )

    with pytest.raises(ValueError, match="round budget"):
        assert_runtime_matches_contract(contract, max_rounds=5)


def test_build_task_contract_defaults_to_tool_workflow_tool_allowlist() -> None:
    contract = build_task_contract(
        user_input="请帮我查一下杭州天气，然后总结一下。",
        chat_key="feishu:c1",
    )

    assert contract.mode == "answer"
    assert contract.allowed_tools == (
        "dispatch_task",
        "wait_for_tasks",
        "get_task_result",
        "reply_to_user",
    )
