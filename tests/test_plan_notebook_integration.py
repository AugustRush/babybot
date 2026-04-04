from __future__ import annotations

from babybot.agent_kernel.plan_notebook import create_root_notebook
from babybot.agent_kernel.plan_notebook_context import build_worker_context_view
from babybot.agent_kernel.plan_notebook_store import PlanNotebookStore
from babybot.orchestrator import OrchestratorAgent


def test_plan_notebook_integration_round_trips_failure_repair_and_completion(
    tmp_path,
) -> None:
    notebook = create_root_notebook(goal="修复本地 pdf 技能", flow_id="flow-int")
    inspect = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Inspect reference",
        objective="检查参考仓库",
        owner="worker",
    )
    notebook.transition_node(
        inspect.node_id,
        "completed",
        summary="已检查参考仓库",
        detail="远端包含 design/design.md",
        metadata={"progress": True},
    )

    failed = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Validate local skill",
        objective="校验本地 skill",
        owner="worker",
        deps=(inspect.node_id,),
    )
    notebook.transition_node(
        failed.node_id,
        "failed",
        summary="本地校验失败",
        detail="缺少 design 文档",
    )
    repair = notebook.promote_failure_to_repair(
        failed.node_id,
        owner="repair-worker",
        message="补齐 design 文档并重新校验",
    )
    notebook.transition_node(
        repair.node_id,
        "completed",
        summary="repair applied",
        detail="已创建 design/design.md 并同步说明。",
        metadata={"progress": True},
    )
    notebook.resolve_checkpoints(failed.node_id, kind="needs_repair")
    notebook.set_completion_summary(
        OrchestratorAgent._build_notebook_completion_summary(
            notebook,
            final_text="本地 pdf 技能已补齐并完成修复",
        )
    )

    store = PlanNotebookStore(tmp_path / "notebooks.db")
    store.save_notebook(notebook, chat_key="feishu:chat-1")
    loaded = store.load_notebook(notebook.notebook_id)

    assert loaded is not None
    assert loaded.ready_to_finalize() is True
    assert loaded.completion_summary["final_summary"] == "本地 pdf 技能已补齐并完成修复"
    assert "repair applied" in " ".join(loaded.completion_summary["node_summaries"])
    matches = store.search_raw_text("补齐 design 文档", notebook_id=notebook.notebook_id)
    assert matches


def test_plan_notebook_integration_keeps_worker_context_bounded_for_large_history() -> None:
    notebook = create_root_notebook(goal="处理超长多步骤任务", flow_id="flow-budget")
    current = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="child_task",
        title="Patch local skill",
        objective="补齐本地 skill 缺失项",
        owner="worker",
    )
    notebook.add_decision(
        node_id=current.node_id,
        summary="先补齐 design 文档，再更新 README",
        rationale="保持变更顺序清晰。",
        metadata={"progress": True},
    )
    for idx in range(40):
        notebook.record_event(
            node_id=current.node_id,
            kind="tool_result",
            summary=f"noise-{idx}",
            detail=f"低价值中间输出 {idx}",
        )

    view = build_worker_context_view(notebook, current.node_id, token_budget=220)
    raw_text = "\n".join(
        f"{event.summary}: {event.detail}" for event in notebook.get_node(current.node_id).events
    )

    assert view.metadata["used_tokens"] <= 220
    assert len(view.text) < len(raw_text)
    assert "先补齐 design 文档，再更新 README" in view.text
    assert "noise-0" not in view.text
