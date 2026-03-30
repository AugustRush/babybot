from __future__ import annotations

from datetime import datetime, timedelta, timezone

from babybot.runtime_job_store import RuntimeJobStore
from babybot.orchestrator import OrchestratorAgent


def test_runtime_job_store_persists_state_transitions(tmp_path) -> None:
    store = RuntimeJobStore(tmp_path / "jobs.db")
    job = store.create(chat_key="feishu:c1", goal="long task")

    store.transition(job.job_id, "running", progress_message="开始执行")
    store.transition(job.job_id, "waiting_tool", progress_message="等待外部工具")

    loaded = store.get(job.job_id)

    assert loaded is not None
    assert loaded.state == "waiting_tool"
    assert loaded.progress_message == "等待外部工具"


def test_runtime_job_store_returns_latest_job_for_chat(tmp_path) -> None:
    store = RuntimeJobStore(tmp_path / "jobs.db")
    first = store.create(chat_key="feishu:c1", goal="first")
    second = store.create(chat_key="feishu:c1", goal="second")

    latest = store.latest_for_chat("feishu:c1")

    assert latest is not None
    assert latest.job_id != first.job_id
    assert latest.job_id == second.job_id


def test_orchestrator_job_status_command_reads_persisted_job(tmp_path) -> None:
    store = RuntimeJobStore(tmp_path / "jobs.db")
    job = store.create(chat_key="feishu:c1", goal="long task")
    store.transition(job.job_id, "waiting_tool", progress_message="等待外部工具")

    agent = object.__new__(OrchestratorAgent)
    agent._initialized = True
    agent._init_lock = None
    agent._interactive_sessions = None
    agent._recent_flow_ids_by_chat = {}
    agent._recent_flows_by_chat = {}
    agent._handoff_locks = {}
    agent._background_tasks = set()
    agent._runtime_job_store = store
    agent.resource_manager = None
    agent.gateway = None
    agent.tape_store = None
    agent.memory_store = None
    agent.config = type("Config", (), {"system": type("System", (), {})()})()

    import asyncio

    response = asyncio.run(
        agent.process_task(f"@job status {job.job_id}", chat_key="feishu:c1")
    )

    assert job.job_id in response.text
    assert "waiting_tool" in response.text


def test_orchestrator_job_resume_command_replays_persisted_goal(tmp_path) -> None:
    store = RuntimeJobStore(tmp_path / "jobs.db")
    job = store.create(
        chat_key="feishu:c1",
        goal="继续处理图片",
        metadata={"media_paths": ["/tmp/demo.png"], "flow_id": "flow-old"},
    )
    store.transition(job.job_id, "failed", progress_message="执行失败", error="boom")

    agent = object.__new__(OrchestratorAgent)
    agent._initialized = True
    agent._init_lock = None
    agent._interactive_sessions = None
    agent._recent_flow_ids_by_chat = {}
    agent._recent_flows_by_chat = {}
    agent._handoff_locks = {}
    agent._background_tasks = set()
    agent._runtime_job_store = store
    agent.resource_manager = None
    agent.gateway = None
    agent.tape_store = None
    agent.memory_store = None
    agent.config = type("Config", (), {"system": type("System", (), {})()})()

    async def _answer_with_dag(*args, **kwargs):
        del args, kwargs
        return "已恢复执行", []

    agent._answer_with_dag = _answer_with_dag

    import asyncio

    response = asyncio.run(
        agent.process_task(f"@job resume {job.job_id}", chat_key="feishu:c1")
    )

    latest = store.latest_for_chat("feishu:c1")

    assert response.text == "已恢复执行"
    assert latest is not None
    assert latest.job_id != job.job_id
    assert latest.goal == "继续处理图片"
    assert latest.metadata["resumed_from"] == job.job_id
    assert latest.metadata["media_paths"] == ["/tmp/demo.png"]


def test_runtime_job_store_cleanup_prunes_stale_orphaned_jobs(tmp_path) -> None:
    store = RuntimeJobStore(tmp_path / "jobs.db")
    stale = store.create(chat_key="feishu:c1", goal="stale task")
    fresh = store.create(
        chat_key="feishu:c1",
        goal="fresh task",
        metadata={"flow_id": "flow-fresh"},
    )
    store.transition(stale.job_id, "running", progress_message="卡住了")
    store.transition(fresh.job_id, "waiting_tool", progress_message="等工具")

    stale_now = (
        datetime.fromisoformat(stale.updated_at) + timedelta(hours=2)
    ).astimezone(timezone.utc)
    report = store.run_maintenance(now=stale_now, retention_seconds=60)

    assert report["orphaned_jobs_pruned"] == 1
    assert stale.job_id in report["orphaned_job_ids"]
    assert fresh.job_id not in report["orphaned_job_ids"]
    assert store.get(stale.job_id) is None
    assert store.get(fresh.job_id) is not None


def test_orchestrator_job_cleanup_command_reports_runtime_maintenance(tmp_path) -> None:
    store = RuntimeJobStore(tmp_path / "jobs.db")
    stale = store.create(chat_key="feishu:c1", goal="stale task")
    store.transition(stale.job_id, "running", progress_message="卡住了")

    agent = object.__new__(OrchestratorAgent)
    agent._initialized = True
    agent._init_lock = None
    agent._interactive_sessions = None
    agent._recent_flow_ids_by_chat = {"feishu:c1": "flow-orphan"}
    agent._recent_flows_by_chat = {"feishu:c1": ["flow-orphan"]}
    agent._handoff_locks = {}
    agent._background_tasks = set()
    agent._runtime_job_store = store
    agent.resource_manager = None
    agent.gateway = None
    agent.tape_store = None
    agent.memory_store = None
    agent.config = type("Config", (), {"system": type("System", (), {})()})()

    import asyncio

    response = asyncio.run(agent.process_task("@job cleanup", chat_key="feishu:c1"))

    assert "[Runtime Maintenance]" in response.text
    assert "orphaned_jobs_pruned=1" in response.text
    assert "unmatched_recent_flows=1" in response.text
