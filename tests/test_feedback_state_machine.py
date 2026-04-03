from __future__ import annotations

from babybot.feedback_events import (
    RuntimeFeedbackEvent,
    _extract_task_label,
    _sanitize_message,
    _truncate_output,
    feedback_dedupe_key,
    normalize_runtime_feedback_event,
    progress_spinner,
    render_runtime_feedback_event,
    render_stage_result,
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
    # The first spinner symbol is ⠋ (Braille frame 0)
    assert rendered.startswith("⠋")


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


# ---------------------------------------------------------------------------
# output_summary and task_label extraction
# ---------------------------------------------------------------------------


def test_truncate_output_short_text_unchanged() -> None:
    assert _truncate_output("hello") == "hello"


def test_truncate_output_long_text_truncated() -> None:
    long = "x" * 1000
    result = _truncate_output(long)
    assert len(result) < 900
    assert "共 1000 字" in result


def test_truncate_output_empty_returns_empty() -> None:
    assert _truncate_output("") == ""
    assert _truncate_output("   ") == ""


def test_extract_task_label_first_line() -> None:
    desc = "查询杭州天气\n详情: 获取今天的天气预报"
    assert _extract_task_label(desc) == "查询杭州天气"


def test_extract_task_label_strips_phase_prefix() -> None:
    assert _extract_task_label("[Research] 搜索相关文档") == "搜索相关文档"
    assert _extract_task_label("[执行] 生成图片") == "生成图片"
    assert _extract_task_label("[Implementation] deploy") == "deploy"


def test_extract_task_label_truncates_long_text() -> None:
    long_desc = "a" * 200
    result = _extract_task_label(long_desc)
    assert len(result) <= 124  # 120 + "…"
    assert result.endswith("…")


def test_extract_task_label_empty_returns_empty() -> None:
    assert _extract_task_label("") == ""
    assert _extract_task_label("   ") == ""


# ---------------------------------------------------------------------------
# normalize: output_summary and task_label from payload
# ---------------------------------------------------------------------------


def test_normalize_succeeded_event_extracts_output_summary() -> None:
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "j1",
            "flow_id": "f1",
            "task_id": "t1",
            "event": "succeeded",
            "payload": {
                "output": "杭州今日多云，气温 18-24°C，南风 3 级",
                "task_description": "查询杭州天气",
            },
        }
    )
    assert normalized.output_summary == "杭州今日多云，气温 18-24°C，南风 3 级"
    assert normalized.task_label == "查询杭州天气"


def test_normalize_succeeded_event_with_long_output_truncates() -> None:
    long_output = "结果 " * 300  # > 800 chars
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "j1",
            "flow_id": "f1",
            "task_id": "t1",
            "event": "succeeded",
            "payload": {"output": long_output},
        }
    )
    assert len(normalized.output_summary) < len(long_output)
    assert "共" in normalized.output_summary


def test_normalize_non_terminal_event_has_no_output_summary() -> None:
    """output_summary should only be populated for terminal events."""
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "j1",
            "flow_id": "f1",
            "task_id": "t1",
            "event": "started",
            "payload": {"output": "should not appear"},
        }
    )
    assert normalized.output_summary == ""


def test_normalize_task_description_not_leaked_as_message() -> None:
    """task_description goes into task_label only, never into message."""
    normalized = normalize_runtime_feedback_event(
        {
            "job_id": "j1",
            "flow_id": "f1",
            "task_id": "t1",
            "event": "succeeded",
            "payload": {
                "task_description": "这是系统内部的任务描述",
                "message": "",
            },
        }
    )
    assert "这是系统内部的任务描述" not in normalized.message
    assert normalized.task_label == "这是系统内部的任务描述"


# ---------------------------------------------------------------------------
# render_stage_result
# ---------------------------------------------------------------------------


def test_render_stage_result_completed_with_output() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="completed",
        stage="task",
        task_label="查询杭州天气",
        output_summary="杭州多云，18-24°C",
    )
    result = render_stage_result(event)
    assert "✅" in result
    assert "查询杭州天气" in result
    assert "杭州多云，18-24°C" in result


def test_render_stage_result_completed_without_output_returns_empty() -> None:
    """No output_summary → nothing to show as a separate message."""
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="completed",
        stage="task",
        task_label="某任务",
        output_summary="",
    )
    assert render_stage_result(event) == ""


def test_render_stage_result_failed_with_error() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="failed",
        stage="task",
        task_label="生成图片",
        error="API 超时",
    )
    result = render_stage_result(event)
    assert "⚠" in result
    assert "生成图片" in result
    assert "API 超时" in result


def test_render_stage_result_failed_without_error_shows_header() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="failed",
        stage="task",
    )
    result = render_stage_result(event)
    assert "⚠" in result


def test_render_stage_result_non_terminal_returns_empty() -> None:
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="running",
        stage="task",
        output_summary="some output",
    )
    assert render_stage_result(event) == ""


def test_render_stage_result_strips_spinner_from_label() -> None:
    """Spinner symbols in task_label (via message fallback) should be stripped."""
    event = RuntimeFeedbackEvent(
        job_id="j1",
        flow_id="f1",
        task_id="t1",
        state="completed",
        stage="task",
        message="⠋ 正在调用 weather",
        output_summary="天气结果",
    )
    result = render_stage_result(event)
    assert "⠋" not in result
    assert "天气结果" in result
