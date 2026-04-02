"""Canonical runtime feedback events and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeFeedbackEvent:
    job_id: str
    flow_id: str
    task_id: str
    state: str
    stage: str
    message: str = ""
    error: str = ""
    progress: float | None = None


_ACTIVE_FEEDBACK_STATES = frozenset(
    {"queued", "planning", "running", "waiting_tool", "waiting_user", "repairing"}
)


def feedback_dedupe_key(
    event: RuntimeFeedbackEvent,
) -> tuple[str, str, str, str, str, str, str]:
    state_group = "active" if event.state in _ACTIVE_FEEDBACK_STATES else event.state
    progress = ""
    if isinstance(event.progress, (int, float)):
        progress = f"{float(event.progress):.4f}"
    return (
        event.job_id,
        event.task_id,
        event.stage,
        state_group,
        event.message,
        progress,
        event.error,
    )


def normalize_runtime_feedback_event(raw: Any) -> RuntimeFeedbackEvent:
    if isinstance(raw, dict):
        payload = dict(raw.get("payload") or {})
        event_name = str(raw.get("event", "") or "").strip().lower()
        job_id = str(raw.get("job_id", "") or payload.get("job_id", "") or "").strip()
        flow_id = str(raw.get("flow_id", "") or "").strip()
        task_id = str(raw.get("task_id", "") or "").strip()
    else:
        payload = dict(getattr(raw, "payload", {}) or {})
        event_name = str(getattr(raw, "event", "") or "").strip().lower()
        job_id = str(
            getattr(raw, "job_id", "") or payload.get("job_id", "") or ""
        ).strip()
        flow_id = str(getattr(raw, "flow_id", "") or "").strip()
        task_id = str(getattr(raw, "task_id", "") or "").strip()
    state = str(payload.get("state", "") or "").strip().lower()
    if not state:
        state = {
            "queued": "queued",
            "started": "running",
            "progress": "running",
            "succeeded": "completed",
            "completed": "completed",
            "failed": "failed",
            "dead_lettered": "failed",
            "stalled": "failed",
            "cancelled": "cancelled",
        }.get(event_name, "running")
    stage = str(payload.get("stage", "") or "").strip() or (
        "job" if job_id and not task_id else "task"
    )
    message = str(
        payload.get("message", "")
        or payload.get("description", "")
        or payload.get("status", "")
        or ""
    ).strip()
    error = str(payload.get("error", "") or "").strip()
    progress = payload.get("progress")
    if isinstance(progress, (int, float)):
        progress = max(0.0, min(1.0, float(progress)))
    else:
        progress = None
    return RuntimeFeedbackEvent(
        job_id=job_id,
        flow_id=flow_id,
        task_id=task_id,
        state=state,
        stage=stage,
        message=message,
        error=error,
        progress=progress,
    )


def render_runtime_feedback_event(event: RuntimeFeedbackEvent) -> str:
    progress_hint = ""
    if isinstance(event.progress, (int, float)) and event.state not in {
        "completed",
        "cancelled",
    }:
        pct = int(float(event.progress) * 100)
        progress_hint = f" ({pct}%)"
    if event.state == "queued":
        return event.message or "排队中…"
    if event.state == "planning":
        label = event.message or "规划中…"
        return f"{label}{progress_hint}"
    if event.state in {"running", "waiting_tool", "waiting_user", "repairing"}:
        label = event.message or "执行中…"
        return f"{label}{progress_hint}"
    if event.state == "completed":
        return event.message or "已完成"
    if event.state == "cancelled":
        return event.message or "已取消"
    if event.state == "failed":
        label = event.message or "执行失败"
        if event.error:
            return f"{label}\n原因：{event.error}"
        return label
    return ""
