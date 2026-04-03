"""Canonical runtime feedback events and rendering helpers."""

from __future__ import annotations

import re
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

# Spinner symbols cycled across successive progress updates to give users
# a sense of "alive" activity.  The caller passes a monotonically
# increasing counter and we pick the symbol at ``counter % len``.
_PROGRESS_SPINNERS = ("⏳", "⌛", "⚙", "✦", "◉", "◎")

# Regex that matches internal task-id patterns like "task task_0_abc123 succeeded".
_TASK_ID_RE = re.compile(r"\btask[\s_]+[a-zA-Z0-9_]+\b")


def progress_spinner(counter: int = 0) -> str:
    """Return a spinner symbol for the given counter value."""
    return _PROGRESS_SPINNERS[counter % len(_PROGRESS_SPINNERS)]


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


def _sanitize_message(message: str) -> str:
    """Strip internal task IDs and other non-user-facing fragments."""
    if not message:
        return ""
    # Remove patterns like "task task_0_abc123 succeeded/failed"
    cleaned = _TASK_ID_RE.sub("", message).strip()
    # Remove leftover "succeeded" / "failed" words that were part of the pattern
    cleaned = re.sub(r"\b(succeeded|failed)\b", "", cleaned).strip()
    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


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
    # Use only "message" or "status" — never "description", which may contain
    # the full task system-prompt and must not be shown to end users.
    message = str(payload.get("message", "") or payload.get("status", "") or "").strip()
    # For "started" events that carry no user-facing message, derive one from
    # resource_id (e.g. "skill.weather" → "正在调用 weather") so the progress
    # card updates visibly when a subtask begins executing.
    if not message and event_name == "started":
        resource_id = str(payload.get("resource_id", "") or "").strip()
        if resource_id:
            # Strip the type prefix (e.g. "skill.", "mcp.", "group.") to get the name.
            resource_name = (
                resource_id.split(".", 1)[-1] if "." in resource_id else resource_id
            )
            message = f"正在调用 {resource_name}"
    # Sanitize: strip internal task IDs from the message.
    message = _sanitize_message(message)
    # Truncate to avoid leaking unexpectedly long internal strings.
    if len(message) > 200:
        message = message[:200] + "…"
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


def render_runtime_feedback_event(
    event: RuntimeFeedbackEvent,
    spinner_counter: int = 0,
) -> str:
    """Render a runtime event into user-facing text.

    Args:
        event: The normalized feedback event.
        spinner_counter: Monotonically increasing counter used to pick
            a rotating spinner symbol for active states.
    """
    spinner = progress_spinner(spinner_counter)

    progress_hint = ""
    if isinstance(event.progress, (int, float)) and event.state not in {
        "completed",
        "cancelled",
    }:
        pct = int(float(event.progress) * 100)
        progress_hint = f" ({pct}%)"

    if event.state == "queued":
        label = event.message or "排队中"
        return f"{spinner} {label}…"
    if event.state == "planning":
        label = event.message or "规划中"
        return f"{spinner} {label}…{progress_hint}"
    if event.state in {"running", "waiting_tool", "waiting_user", "repairing"}:
        label = event.message or "执行中"
        return f"{spinner} {label}…{progress_hint}"
    if event.state == "completed":
        return event.message or "✅ 已完成"
    if event.state == "cancelled":
        return event.message or "⊘ 已取消"
    if event.state == "failed":
        label = event.message or "执行失败"
        if event.error:
            return f"⚠ {label}\n原因：{event.error}"
        return f"⚠ {label}"
    return ""
