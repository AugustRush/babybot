from __future__ import annotations

from babybot.feedback_events import (
    RuntimeFeedbackEvent,
    _sanitize_message,
    feedback_dedupe_key,
    normalize_runtime_feedback_event,
    progress_spinner,
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

    # The warning symbol should be present for any failed state.
    assert "⚠" in rendered
    # The original message should be preserved.
    assert "阶段结束" in rendered


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


def test_progress_spinner_cycles_through_symbols() -> None:
    symbols = set()
    for i in range(20):
        symbols.add(progress_spinner(i))
    # Should cycle through multiple distinct symbols.
    assert len(symbols) > 1


def test_progress_spinner_is_deterministic() -> None:
    # Same counter always produces same symbol.
    assert progress_spinner(0) == progress_spinner(0)
    assert progress_spinner(3) == progress_spinner(3)


def test_sanitize_message_strips_task_id_succeeded() -> None:
    raw = "task task_2_1527df succeeded"
    assert _sanitize_message(raw) == ""


def test_sanitize_message_strips_task_id_failed() -> None:
    raw = "task task_0_abc123 failed"
    assert _sanitize_message(raw) == ""


def test_sanitize_message_preserves_user_text() -> None:
    assert _sanitize_message("正在查询天气") == "正在查询天气"
    assert _sanitize_message("子任务已完成") == "子任务已完成"


def test_render_running_state_includes_spinner() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="running",
        stage="task",
        message="查询中",
    )
    rendered = render_runtime_feedback_event(event, spinner_counter=0)
    # Should contain a spinner symbol followed by the message.
    assert "查询中" in rendered
    assert "…" in rendered
    # The first spinner symbol is ⏳
    assert rendered.startswith("⏳")


def test_render_completed_state_has_checkmark() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="completed",
        stage="task",
    )
    rendered = render_runtime_feedback_event(event)
    assert "✅" in rendered


def test_render_failed_state_has_warning_and_error() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="failed",
        stage="task",
        error="连接超时",
    )
    rendered = render_runtime_feedback_event(event)
    assert "⚠" in rendered
    assert "连接超时" in rendered


def test_normalize_strips_internal_task_id_from_message() -> None:
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "job-1",
            "flow_id": "flow-1",
            "task_id": "task_2_1527df",
            "event": "succeeded",
            "payload": {
                "message": "task task_2_1527df succeeded",
                "status": "succeeded",
            },
        }
    )
    # The raw internal message should be stripped.
    assert "task_2_1527df" not in normalized.message


def test_normalize_started_event_derives_message_from_resource_id() -> None:
    """When a 'started' event has no message, resource_id is used to build one."""
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "job-1",
            "flow_id": "flow-1",
            "task_id": "task_0_abc",
            "event": "started",
            "payload": {
                "resource_id": "skill.text-to-image",
                "description": "[执行型子任务] ...",  # must NOT appear in message
            },
        }
    )
    assert normalized.state == "running"
    assert "text-to-image" in normalized.message
    assert "执行型子任务" not in normalized.message


def test_normalize_started_event_without_resource_id_falls_back_to_default() -> None:
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "job-1",
            "flow_id": "flow-1",
            "task_id": "task_0_abc",
            "event": "started",
            "payload": {},
        }
    )
    assert normalized.state == "running"
    # No resource_id → message is empty, render falls back to "执行中"
    rendered = render_runtime_feedback_event(normalized)
    assert "执行中" in rendered
