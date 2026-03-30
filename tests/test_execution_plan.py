from __future__ import annotations

from babybot.execution_plan import ExecutionPlan, PlanStep, build_execution_plan
from babybot.task_contract import TaskContract


def test_build_execution_plan_for_debate_carries_round_budget() -> None:
    contract = TaskContract(
        chat_key="feishu:c1",
        goal="比较两版诗，给出胜者",
        mode="debate",
        deliverable="winner",
        round_budget=1,
        termination_rule="single_round",
        allow_clarification=False,
        allowed_tools=(),
        allowed_agents=("judge_master",),
        metadata={},
    )

    plan = build_execution_plan(contract)

    assert plan == ExecutionPlan(
        plan_id=plan.plan_id,
        contract=contract,
        steps=(
            PlanStep(
                step_id="step_debate",
                kind="debate",
                title="Structured debate",
                payload={
                    "participants": ["judge_master"],
                    "round_budget": 1,
                    "stopping_condition": "single_round",
                },
            ),
        ),
        round_budget=1,
        stopping_condition="single_round",
    )


def test_build_execution_plan_for_single_answer_bypasses_debate() -> None:
    contract = TaskContract(
        chat_key="feishu:c1",
        goal="今天天气怎么样",
        mode="answer",
        deliverable="final_answer",
        round_budget=None,
        termination_rule="final_answer",
        allow_clarification=True,
        allowed_tools=(),
        allowed_agents=(),
        metadata={},
    )

    plan = build_execution_plan(contract)

    assert plan.steps[0].kind == "direct_answer"
    assert plan.round_budget is None
