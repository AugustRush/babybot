from __future__ import annotations

from babybot.agent_kernel.plan_notebook import create_root_notebook
from babybot.context_views import summarize_context_view
from babybot.memory_models import MemoryRecord


def _build_notebook():
    notebook = create_root_notebook(goal="对照远端仓库补齐本地 pdf 技能", flow_id="flow-x")
    dep = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="task",
        title="Inspect remote repo",
        objective="检查远端仓库结构",
        owner="worker",
        resource_ids=("web",),
    )
    notebook.transition_node(dep.node_id, "completed", summary="已检查远端仓库", detail="缺少 design/design.md")
    notebook.add_decision(
        node_id=dep.node_id,
        summary="需要补齐 design 文档",
        rationale="远端仓库存在 design/design.md，本地没有。",
    )
    notebook.add_artifact(
        node_id=dep.node_id,
        path="/tmp/design.md",
        label="design 文档草稿",
    )

    current = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="task",
        title="Patch local skill",
        objective="补齐本地技能缺失文件",
        owner="worker",
        resource_ids=("code",),
        deps=(dep.node_id,),
    )
    notebook.add_issue(
        node_id=current.node_id,
        title="本地未找到 design 文档",
        detail="需要创建新文件并同步关键说明。",
    )
    for idx in range(6):
        notebook.record_event(
            node_id=current.node_id,
            kind="tool_result",
            summary=f"noise-{idx}",
            detail=f"不应优先进入 prompt 的低价值输出片段 {idx}",
        )
    notebook.record_event(
        node_id=current.node_id,
        kind="observation",
        summary="当前需要落盘 design 文档",
        detail="优先写入设计文档，再做校验。",
    )
    return notebook, dep.node_id, current.node_id


def test_worker_context_view_prioritizes_objective_deps_decisions_and_blockers() -> None:
    from babybot.agent_kernel.plan_notebook_context import build_worker_context_view

    notebook, dep_id, current_id = _build_notebook()

    view = build_worker_context_view(notebook, current_id, token_budget=260)

    assert "补齐本地技能缺失文件" in view.text
    assert dep_id in view.text
    assert "缺少 design/design.md" in view.text
    assert "需要补齐 design 文档" in view.text
    assert "本地未找到 design 文档" in view.text
    assert "noise-0" not in view.text


def test_search_notebook_text_can_find_omitted_raw_event_details() -> None:
    from babybot.agent_kernel.plan_notebook_context import (
        build_worker_context_view,
        search_notebook_text,
    )

    notebook, _, current_id = _build_notebook()
    compact = build_worker_context_view(notebook, current_id, token_budget=180)

    assert "低价值输出片段 0" not in compact.text
    matches = search_notebook_text(notebook, "低价值输出片段 0")

    assert matches
    assert "低价值输出片段 0" in matches[0]["detail"]


def test_completion_context_view_uses_summary_and_skips_low_signal_noise() -> None:
    from babybot.agent_kernel.plan_notebook_context import build_completion_context_view

    notebook, _, current_id = _build_notebook()
    notebook.transition_node(
        current_id,
        "completed",
        summary="已补齐本地技能",
        detail="已创建 design 文档并完成本地同步。",
    )
    notebook.set_completion_summary(
        {
            "final_summary": "本地 minimax-pdf 技能已补齐 design 文档并完成同步。",
            "decision_register": ["先补齐文档，再校验技能结构"],
            "artifact_manifest": ["/tmp/design.md"],
        }
    )

    view = build_completion_context_view(notebook, token_budget=220)

    assert "本地 minimax-pdf 技能已补齐 design 文档并完成同步。" in view.text
    assert "/tmp/design.md" in view.text
    assert "先补齐文档，再校验技能结构" in view.text
    assert "noise-1" not in view.text


def test_context_views_prefer_notebook_summary_records_over_ephemeral_task_state() -> None:
    class _MemoryStore:
        def __init__(self, records):
            self._records = records

        def list_memories(self, chat_id: str):
            assert chat_id == "chat-1"
            return list(self._records)

    notebook_record = MemoryRecord(
        memory_type="notebook_summary",
        key="completion",
        value={"final_summary": "Notebook summary"},
        summary="Notebook summary: 本地技能已补齐 design 文档",
        tier="ephemeral",
        scope="chat",
        scope_id="chat-1",
    )
    task_state_record = MemoryRecord(
        memory_type="task_state",
        key="last_success",
        value={"description": "old", "output": "old output"},
        summary="最近完成：旧任务",
        tier="ephemeral",
        scope="chat",
        scope_id="chat-1",
    )

    text = summarize_context_view(
        memory_store=_MemoryStore([task_state_record, notebook_record]),
        chat_id="chat-1",
        query="补齐本地 pdf 技能",
    )

    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert lines
    assert lines[0] == "- Notebook summary: 本地技能已补齐 design 文档"


def test_context_views_stay_concise_and_surface_notebook_index_for_related_follow_up() -> None:
    class _MemoryStore:
        def __init__(self, records):
            self._records = records

        def list_memories(self, chat_id: str):
            assert chat_id == "chat-1"
            return list(self._records)

    notebook_summary = MemoryRecord(
        memory_type="notebook_summary",
        key="completion:nb-1",
        value={"final_summary": "已补齐 minimax-pdf design 文档"},
        summary="Notebook summary: 已补齐 minimax-pdf design 文档",
        tier="soft",
        scope="chat",
        scope_id="chat-1",
    )
    notebook_index = MemoryRecord(
        memory_type="notebook_index",
        key="index:nb-1",
        value={"search_terms": ["minimax-pdf", "design"], "node_summaries": ["Inspect reference"]},
        summary="Notebook index: minimax-pdf, design, Inspect reference",
        tier="soft",
        scope="chat",
        scope_id="chat-1",
    )
    task_records = [
        MemoryRecord(
            memory_type="task_state",
            key=f"task-{idx}",
            value={"description": f"old-{idx}", "output": f"noise-{idx}"},
            summary=f"最近完成：旧任务 {idx}",
            tier="ephemeral",
            scope="chat",
            scope_id="chat-1",
        )
        for idx in range(12)
    ]

    text = summarize_context_view(
        memory_store=_MemoryStore(task_records + [notebook_index, notebook_summary]),
        chat_id="chat-1",
        query="继续完善 minimax-pdf 的 design 说明",
    )

    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(lines) <= 18
    assert lines[0] == "- Notebook summary: 已补齐 minimax-pdf design 文档"
    assert any("Notebook index: minimax-pdf, design, Inspect reference" == line[2:] for line in lines)
