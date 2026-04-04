from __future__ import annotations

from pathlib import Path

from babybot.agent_kernel.plan_notebook import create_root_notebook
from babybot.memory_store import HybridMemoryStore


def _build_notebook():
    notebook = create_root_notebook(
        goal="对照远端仓库补齐本地 pdf 技能",
        flow_id="flow-10",
        plan_id="plan-10",
        metadata={"chat_key": "feishu:c1"},
    )
    child = notebook.add_child_node(
        parent_id=notebook.root_node_id,
        kind="task",
        title="Inspect remote repo",
        objective="检查远端仓库结构",
        owner="worker",
        resource_ids=("web",),
    )
    notebook.record_event(
        node_id=child.node_id,
        kind="observation",
        summary="发现远端存在 design 文档",
        detail="远端 skills/minimax-pdf/design/design.md 存在，本地没有。",
    )
    notebook.transition_node(
        child.node_id,
        "completed",
        summary="已检查远端仓库",
        detail="确认 design/design.md 缺失。",
    )
    notebook.set_completion_summary(
        {
            "final_summary": "本地 minimax-pdf 技能已补齐 design 文档。",
            "decision_register": ["先补齐 design 文档，再做技能结构校验"],
            "artifact_manifest": ["/tmp/design.md"],
            "node_summaries": ["远端有 design/design.md，本地缺失"],
            "search_terms": ["minimax-pdf", "design", "技能"],
        }
    )
    return notebook


def test_plan_notebook_store_round_trips_snapshot_and_completion_summary(tmp_path: Path) -> None:
    from babybot.agent_kernel.plan_notebook_store import PlanNotebookStore

    store = PlanNotebookStore(tmp_path / "plan_notebook.db")
    notebook = _build_notebook()

    store.save_notebook(notebook, chat_key="feishu:c1")
    loaded = store.load_notebook(notebook.notebook_id)

    assert loaded is not None
    assert loaded.notebook_id == notebook.notebook_id
    assert loaded.completion_summary["final_summary"] == "本地 minimax-pdf 技能已补齐 design 文档。"
    assert len(loaded.raw_events) >= 2


def test_plan_notebook_store_searches_raw_event_text_with_like_fallback(tmp_path: Path) -> None:
    from babybot.agent_kernel.plan_notebook_store import PlanNotebookStore

    store = PlanNotebookStore(tmp_path / "plan_notebook.db", prefer_fts=False)
    notebook = _build_notebook()
    store.save_notebook(notebook, chat_key="feishu:c1")

    matches = store.search_raw_text(
        "本地没有",
        notebook_id=notebook.notebook_id,
        chat_key="feishu:c1",
    )

    assert matches
    assert "本地没有" in matches[0]["detail"]


def test_memory_store_observe_notebook_completion_persists_summary_and_index(tmp_path: Path) -> None:
    memory_store = HybridMemoryStore(
        db_path=tmp_path / "memory.db",
        memory_dir=tmp_path / "memory",
    )
    notebook = _build_notebook()

    memory_store.observe_notebook_completion(
        chat_id="feishu:c1",
        notebook_id=notebook.notebook_id,
        completion_summary=notebook.completion_summary,
    )

    records = memory_store.list_memories(chat_id="feishu:c1")
    summaries = [record.summary for record in records if record.memory_type == "notebook_summary"]
    indexes = [record.summary for record in records if record.memory_type == "notebook_index"]

    assert any("本地 minimax-pdf 技能已补齐 design 文档" in item for item in summaries)
    assert any("minimax-pdf" in item for item in indexes)
