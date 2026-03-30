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


def feedback_dedupe_key(event: RuntimeFeedbackEvent) -> tuple[str, str, str, str]:
    return (event.job_id, event.task_id, event.stage, event.state)


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
        job_id = str(getattr(raw, "job_id", "") or payload.get("job_id", "") or "").strip()
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
    progress_pct = ""
    if isinstance(event.progress, (int, float)):
        progress_pct = f" ({int(float(event.progress) * 100)}%)"
    if event.state in {"queued", "planning", "running", "waiting_tool", "waiting_user", "repairing"}:
        label = event.message or event.stage or "处理中"
        return f"处理中：{label}{progress_pct}"
    if event.state == "completed":
        label = event.message or event.stage or "阶段完成"
        return f"阶段完成：{label}"
    if event.state == "cancelled":
        label = event.message or event.stage or "任务已取消"
        return f"任务已取消：{label}"
    if event.state == "failed":
        label = event.message or event.stage or "任务失败"
        if event.error:
            return f"阶段失败：{label}\n原因：{event.error}"
        return f"阶段失败：{label}"
    return ""
