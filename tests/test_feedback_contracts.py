from __future__ import annotations

import asyncio

from babybot.feedback_events import RuntimeFeedbackEvent, render_runtime_feedback_event
from babybot.interactive_sessions.manager import InteractiveSessionManager
from babybot.runtime_job_store import RuntimeJobStore


class _Backend:
    async def start(self, chat_key: str):
        return {"chat_key": chat_key, "backend": "claude"}

    async def send(self, handle, message):
        del handle, message
        from babybot.interactive_sessions.types import InteractiveReply

        return InteractiveReply(text="ok")

    async def stop(self, handle, reason: str = "user_stop") -> None:
        del handle, reason

    def status(self, handle) -> dict[str, str]:
        return {"backend": handle["backend"]}


def test_feedback_renderer_handles_every_canonical_job_state() -> None:
    for state in (
        "queued",
        "planning",
        "running",
        "waiting_tool",
        "waiting_user",
        "repairing",
        "completed",
        "failed",
        "cancelled",
    ):
        rendered = render_runtime_feedback_event(
            RuntimeFeedbackEvent(
                job_id="job-1",
                flow_id="flow-1",
                task_id="task-1",
                state=state,
                stage="job",
                message="状态变更",
            )
        )
        assert rendered


def test_session_summary_and_job_status_are_independently_consistent(tmp_path) -> None:
    manager = InteractiveSessionManager(
        backends={"claude": _Backend()},
        max_age_seconds=7200,
    )
    store = RuntimeJobStore(tmp_path / "jobs.db")
    job = store.create(chat_key="feishu:c1", goal="demo")
    store.transition(job.job_id, "waiting_user", progress_message="等用户确认")

    async def _run() -> tuple[dict[str, object], str]:
        await manager.start(chat_key="feishu:c1", backend_name="claude")
        summary = manager.summary()
        loaded = store.get(job.job_id)
        return summary, loaded.state if loaded is not None else ""

    summary, state = asyncio.run(_run())

    assert summary["active_count"] == 1
    assert state == "waiting_user"
