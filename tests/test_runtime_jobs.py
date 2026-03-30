from __future__ import annotations

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
