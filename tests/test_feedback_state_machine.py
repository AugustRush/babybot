from __future__ import annotations

from babybot.feedback_events import (
    RuntimeFeedbackEvent,
    feedback_dedupe_key,
    normalize_runtime_feedback_event,
    render_runtime_feedback_event,
)


def test_runtime_feedback_dedupes_by_event_identity_not_text() -> None:
    first = RuntimeFeedbackEvent(
        job_id="job-1",
        flow_id="flow-1",
        task_id="task-a",
        state="running",
        stage="worker",
        message="处理中",
    )
    second = RuntimeFeedbackEvent(
        job_id="job-1",
        flow_id="flow-1",
        task_id="task-b",
        state="running",
        stage="worker",
        message="处理中",
    )

    assert feedback_dedupe_key(first) != feedback_dedupe_key(second)


def test_runtime_feedback_collapses_adjacent_active_states_for_same_task() -> None:
    queued = RuntimeFeedbackEvent(
        job_id="job-1",
        flow_id="flow-1",
        task_id="task-a",
        state="queued",
        stage="worker",
        message="下载模型",
    )
    started = RuntimeFeedbackEvent(
        job_id="job-1",
        flow_id="flow-1",
        task_id="task-a",
        state="running",
        stage="worker",
        message="下载模型",
    )

    assert feedback_dedupe_key(queued) == feedback_dedupe_key(started)


def test_render_runtime_feedback_event_surfaces_failed_state_without_error() -> None:
    event = RuntimeFeedbackEvent(
        job_id="job-1",
        flow_id="flow-1",
        task_id="task-a",
        state="failed",
        stage="worker",
        message="阶段结束",
        error="",
    )

    rendered = render_runtime_feedback_event(event)

    assert "失败" in rendered


def test_normalize_runtime_feedback_event_maps_succeeded_to_completed() -> None:
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "job-1",
            "flow_id": "flow-1",
            "task_id": "task-a",
            "event": "succeeded",
            "payload": {
                "description": "查询天气",
                "output": "杭州多云",
            },
        }
    )

    assert normalized.state == "completed"
    assert normalized.stage == "task"
