from __future__ import annotations

from babybot.execution_plan import (
    ExecutionPlan,
    PlanStep,
    build_execution_plan,
    compile_execution_plan_to_notebook,
)
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
        allowed_tools=("dispatch_team", "reply_to_user"),
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
                kind="team_debate",
                title="Structured debate",
                payload={
                    "participants": ["judge_master"],
                    "round_budget": 1,
                    "stopping_condition": "single_round",
                    "allowed_tools": ["dispatch_team", "reply_to_user"],
                    "allowed_agents": ["judge_master"],
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
        allowed_tools=(
            "dispatch_task",
            "wait_for_tasks",
            "get_task_result",
            "reply_to_user",
        ),
        allowed_agents=(),
        metadata={},
    )

    plan = build_execution_plan(contract)

    assert plan.steps[0].kind == "tool_workflow"
    assert plan.round_budget is None
    assert plan.steps[0].payload["allowed_tools"] == [
        "dispatch_task",
        "wait_for_tasks",
        "get_task_result",
        "reply_to_user",
    ]


def test_compile_execution_plan_to_notebook_creates_root_and_step_nodes() -> None:
    contract = TaskContract(
        chat_key="feishu:c1",
        goal="今天天气怎么样",
        mode="answer",
        deliverable="final_answer",
        round_budget=None,
        termination_rule="final_answer",
        allow_clarification=True,
        allowed_tools=(
            "dispatch_task",
            "wait_for_tasks",
            "get_task_result",
            "reply_to_user",
        ),
        allowed_agents=(),
        metadata={},
    )
    plan = build_execution_plan(contract)

    notebook = compile_execution_plan_to_notebook(
        plan,
        flow_id="flow-plan",
        metadata={"chat_key": contract.chat_key},
    )

    assert notebook.goal == contract.goal
    assert notebook.plan_id == plan.plan_id
    assert notebook.root_node_id in notebook.nodes
    step_nodes = [
        node for node in notebook.nodes.values() if node.parent_id == notebook.root_node_id
    ]
    assert len(step_nodes) == 1
    assert step_nodes[0].kind == "tool_workflow"
    assert step_nodes[0].title == "Tool-guided answer"


def test_compile_execution_plan_to_notebook_promotes_debate_to_team_node() -> None:
    contract = TaskContract(
        chat_key="feishu:c1",
        goal="两个专家讨论并给出结论",
        mode="debate",
        deliverable="final_answer",
        round_budget=2,
        termination_rule="round_budget",
        allow_clarification=False,
        allowed_tools=("dispatch_team", "reply_to_user"),
        allowed_agents=("architect", "reviewer"),
        metadata={},
    )
    plan = build_execution_plan(contract)

    notebook = compile_execution_plan_to_notebook(
        plan,
        flow_id="flow-team-plan",
        metadata={"chat_key": contract.chat_key},
    )

    step_nodes = [
        node for node in notebook.nodes.values() if node.parent_id == notebook.root_node_id
    ]

    assert len(step_nodes) == 1
    assert step_nodes[0].kind == "team_debate"
    assert step_nodes[0].metadata["step_id"] == "step_debate"
    assert step_nodes[0].metadata["payload"]["participants"] == ["architect", "reviewer"]
