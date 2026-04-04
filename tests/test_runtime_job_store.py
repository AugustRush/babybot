from __future__ import annotations

from pathlib import Path

from babybot.runtime_job_store import RuntimeJobStore


def test_runtime_job_store_persists_notebook_id_across_create_and_transition(
    tmp_path: Path,
) -> None:
    store = RuntimeJobStore(tmp_path / "runtime_jobs.db")

    created = store.create(
        chat_key="feishu:c1",
        goal="复杂任务",
        plan_id="plan-1",
        notebook_id="notebook-1",
        metadata={"flow_id": "flow-1"},
    )

    assert created.notebook_id == "notebook-1"

    updated = store.transition(
        created.job_id,
        "running",
        progress_message="执行中",
        notebook_id="notebook-1",
    )

    fetched = store.get(created.job_id)

    assert updated.notebook_id == "notebook-1"
    assert fetched is not None
    assert fetched.notebook_id == "notebook-1"
    assert fetched.progress_message == "执行中"
