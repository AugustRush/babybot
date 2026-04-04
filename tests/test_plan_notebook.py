from __future__ import annotations

import pytest

from babybot.agent_kernel.types import ExecutionContext
from babybot.execution_plan import build_execution_plan
from babybot.task_contract import TaskContract


def _build_contract() -> TaskContract:
    return TaskContract(
        chat_key="feishu:c1",
        goal="参考仓库补齐本地 pdf 技能",
        mode="answer",
        deliverable="final_answer",
        round_budget=None,
        termination_rule="final_answer",
        allow_clarification=False,
        allowed_tools=(
            "dispatch_task",
            "wait_for_tasks",
            "get_task_result",
            "reply_to_user",
        ),
        allowed_agents=(),
        metadata={"flow_id": "flow-1"},
    )


def test_create_root_notebook_preserves_goal_and_plan_metadata() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    contract = _build_contract()
    plan = build_execution_plan(contract)

    notebook = create_root_notebook(
        goal=contract.goal,
        flow_id="flow-1",
        plan_id=plan.plan_id,
        metadata={"chat_key": contract.chat_key},
    )

    assert notebook.goal == contract.goal
    assert notebook.flow_id == "flow-1"
    assert notebook.plan_id == plan.plan_id
    assert notebook.root_node_id in notebook.nodes
    assert notebook.nodes[notebook.root_node_id].objective == contract.goal
    assert notebook.nodes[notebook.root_node_id].status == "running"


def test_notebook_child_node_tracks_parent_owner_resources_and_events() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    notebook = create_root_notebook(goal="修复复杂任务", flow_id="flow-2")
    child = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="task",
        title="Research repository",
        objective="对照远端仓库检查差异",
        owner="orchestrator",
        resource_ids=("web", "code"),
        deps=(),
    )

    notebook.record_event(
        node_id=child.node_id,
        kind="observation",
        summary="发现本地少了 design 文档",
        detail="远端有 design/design.md，本地没有。",
    )
    notebook.record_event(
        node_id=child.node_id,
        kind="decision",
        summary="需要补齐 design 文档",
        detail="先创建缺失文档，再做验证。",
    )

    child_after = notebook.nodes[child.node_id]
    assert child_after.parent_id == notebook.root_node_id
    assert child_after.owner == "orchestrator"
    assert child_after.resource_ids == ("web", "code")
    assert [event.kind for event in child_after.events] == ["observation", "decision"]
    assert child_after.latest_summary == "需要补齐 design 文档"


def test_notebook_rejects_invalid_terminal_status_regression() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    notebook = create_root_notebook(goal="修复复杂任务", flow_id="flow-3")
    child = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="task",
        title="Implement fix",
        objective="写入修复",
        owner="worker",
        resource_ids=("code",),
        deps=(),
    )

    notebook.transition_node(child.node_id, "completed")

    with pytest.raises(ValueError):
        notebook.transition_node(child.node_id, "running")


def test_execution_context_exposes_notebook_state_view() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    notebook = create_root_notebook(goal="修复复杂任务", flow_id="flow-4")
    context = ExecutionContext(
        state={
            "plan_notebook": notebook,
            "plan_notebook_id": notebook.notebook_id,
            "current_notebook_node_id": notebook.root_node_id,
            "notebook_context_budget": 2400,
        }
    )

    assert context.notebook_state["plan_notebook"] is notebook
    assert context.notebook_state["plan_notebook_id"] == notebook.notebook_id
    assert context.notebook_state["current_notebook_node_id"] == notebook.root_node_id
    assert context.notebook_state["notebook_context_budget"] == 2400


def test_notebook_creates_repair_branch_and_checkpoint_for_failed_node() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    notebook = create_root_notebook(goal="修复失败任务", flow_id="flow-5")
    failed = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Validate skill",
        objective="验证本地技能",
        owner="worker",
    )
    notebook.transition_node(
        failed.node_id,
        "failed",
        summary="校验失败",
        detail="缺少 design 文档",
    )

    repair = notebook.promote_failure_to_repair(
        failed.node_id,
        owner="repair-worker",
        message="补齐 design 文档并重新验证",
    )

    assert repair.parent_id == failed.node_id
    assert repair.metadata["repair_for"] == failed.node_id
    assert repair.status == "running"
    assert any(
        checkpoint.kind == "needs_repair" and checkpoint.status == "open"
        for checkpoint in notebook.get_node(failed.node_id).checkpoints
    )


def test_notebook_waits_for_human_input_until_checkpoint_is_resolved() -> None:
    from babybot.agent_kernel.plan_notebook import create_root_notebook

    notebook = create_root_notebook(goal="等待用户输入", flow_id="flow-6")
    child = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Ask for file",
        objective="请求示例 PDF",
        owner="worker",
    )

    checkpoint = notebook.mark_needs_human_input(
        child.node_id,
        message="需要用户提供示例 PDF",
    )

    assert notebook.get_node(child.node_id).status == "waiting"
    assert notebook.ready_to_finalize() is False

    notebook.resolve_checkpoints(child.node_id, kind=checkpoint.kind)
    notebook.transition_node(child.node_id, "completed", summary="用户已提供示例 PDF")

    assert notebook.ready_to_finalize() is True
