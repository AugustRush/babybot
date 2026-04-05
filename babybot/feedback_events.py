"""Canonical runtime feedback events and rendering helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Maximum characters of subtask output shown in the stage result card.
_OUTPUT_SUMMARY_MAX_CHARS = 800
# Maximum characters of task_description shown as the stage label.
_TASK_DESC_MAX_CHARS = 120


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
    # output_summary: truncated result text of a completed subtask.
    # Populated only for terminal events (succeeded / failed) that carry output.
    output_summary: str = ""
    # task_label: a short, user-facing description of what the subtask did.
    # Derived from task_description payload field (first line, truncated).
    task_label: str = ""


def runtime_event_primary_label(event: RuntimeFeedbackEvent) -> str:
    """Return the canonical user-facing label for a runtime event."""
    label = str(event.task_label or "").strip()
    if label:
        return label
    message = str(event.message or "").strip()
    if message:
        return message
    stage = str(event.stage or "").strip()
    if stage and stage not in {"task", "job"}:
        return stage
    return ""


_ACTIVE_FEEDBACK_STATES = frozenset(
    {"queued", "planning", "running", "waiting_tool", "waiting_user", "repairing"}
)

# Spinner symbols cycled across successive progress updates to give users
# a sense of "alive" activity.  The caller passes a monotonically
# increasing counter and we pick the symbol at ``counter % len``.
# Braille 10-frame sequence — visually continuous rotation effect.
_PROGRESS_SPINNERS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

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


def _extract_task_label(raw_description: str) -> str:
    """Return a short, user-facing label from a raw task description.

    Takes the first non-empty line and strips common phase prefixes like
    ``[Research]``, ``[执行]``, etc., then truncates.
    """
    if not raw_description:
        return ""
    lines = [line.strip() for line in raw_description.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[0]
    if candidate == "[执行型子任务]":
        for line in lines[1:]:
            if line.startswith("原始子任务："):
                candidate = line.split("：", 1)[1].strip()
                break
            if line.startswith("原始子任务:"):
                candidate = line.split(":", 1)[1].strip()
                break
        else:
            return ""
    candidate = re.sub(r"^\[[^\]]{1,30}\]\s*", "", candidate).strip()
    if len(candidate) > _TASK_DESC_MAX_CHARS:
        candidate = candidate[:_TASK_DESC_MAX_CHARS] + "…"
    return candidate


def _build_notebook_feedback_message(payload: dict[str, Any]) -> str:
    phase = str(payload.get("notebook_phase", "") or "").strip()
    owner = str(payload.get("notebook_owner", "") or "").strip()
    next_action = str(payload.get("notebook_next_action", "") or "").strip()
    completed_steps = payload.get("notebook_completed_steps") or ()
    blockers = payload.get("notebook_blockers") or ()

    parts: list[str] = []
    if phase:
        parts.append(f"当前阶段：{phase}")
    if owner:
        parts.append(f"当前负责人：{owner}")
    if isinstance(completed_steps, (list, tuple)):
        cleaned = [str(item).strip() for item in completed_steps if str(item).strip()]
        if cleaned:
            parts.append(f"已完成：{'；'.join(cleaned[:3])}")
    if isinstance(blockers, (list, tuple)):
        cleaned = [str(item).strip() for item in blockers if str(item).strip()]
        if cleaned:
            parts.append(f"阻塞：{'；'.join(cleaned[:2])}")
    if next_action:
        parts.append(f"下一步：{next_action}")
    return " | ".join(parts)


def _truncate_output(output: str) -> str:
    """Truncate subtask output for display in a stage result card."""
    if not output:
        return ""
    text = output.strip()
    if len(text) > _OUTPUT_SUMMARY_MAX_CHARS:
        return text[:_OUTPUT_SUMMARY_MAX_CHARS] + f"\n…（共 {len(text)} 字）"
    return text


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
    notebook_feedback = _build_notebook_feedback_message(payload)
    if notebook_feedback:
        message = notebook_feedback
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

    # output_summary: only populated for terminal events that carry output.
    output_summary = ""
    if event_name in {"succeeded", "completed"}:
        raw_output = str(payload.get("output", "") or "").strip()
        output_summary = _truncate_output(raw_output)

    # task_label: user-readable description of what the subtask did.
    # Prefer explicit user_label. Fall back to task_description only when needed.
    task_label = _extract_task_label(str(payload.get("user_label", "") or "").strip())
    if not task_label:
        task_label = _extract_task_label(
            str(payload.get("task_description", "") or "").strip()
        )
    if not task_label:
        task_label = _extract_task_label(str(payload.get("description", "") or "").strip())

    return RuntimeFeedbackEvent(
        job_id=job_id,
        flow_id=flow_id,
        task_id=task_id,
        state=state,
        stage=stage,
        message=message,
        error=error,
        progress=progress,
        output_summary=output_summary,
        task_label=task_label,
    )


def render_runtime_feedback_event(
    event: RuntimeFeedbackEvent,
    spinner_counter: int = 0,
) -> str:
    """Render a runtime event into user-facing progress text.

    This is used for the unified lifecycle card (patch updates).
    For terminal events with output, callers should use
    ``render_stage_result`` to produce a separate stage result message.

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
        label = runtime_event_primary_label(event) or "排队中"
        return f"{spinner} {label}…"
    if event.state == "planning":
        label = runtime_event_primary_label(event) or "规划中"
        return f"{spinner} {label}…{progress_hint}"
    if event.state in {"running", "waiting_tool", "waiting_user", "repairing"}:
        label = runtime_event_primary_label(event) or "执行中"
        return f"{spinner} {label}…{progress_hint}"
    if event.state == "completed":
        return runtime_event_primary_label(event) or "✅ 已完成"
    if event.state == "cancelled":
        return runtime_event_primary_label(event) or "⊘ 已取消"
    if event.state == "failed":
        label = runtime_event_primary_label(event) or "执行失败"
        if event.error:
            return f"⚠ {label}\n原因：{event.error}"
        return f"⚠ {label}"
    return ""


def render_stage_result(event: RuntimeFeedbackEvent) -> str:
    """Render a completed subtask as an independent stage result message.

    Returns an empty string when there is no output worth showing
    (e.g. the event has no output_summary and no meaningful error).
    This text is sent as a separate message, not patched into the
    lifecycle card.
    """
    if event.state not in {"completed", "failed"}:
        return ""

    # Build header line from task_label or fallback resource name from message.
    label = runtime_event_primary_label(event)
    # Strip leading spinner / status symbols that may be in message.
    # Covers Braille spinner frames, status icons, and legacy symbols.
    label = re.sub(r"^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏✅⚠⊘]\s*", "", label).strip()

    if event.state == "completed":
        header = f"✅ {label}" if label else "✅ 阶段完成"
        if event.output_summary:
            return f"{header}\n\n{event.output_summary}"
        return ""  # No output → nothing interesting to show separately.

    # Failed state.
    header = f"⚠ {label} 失败" if label else "⚠ 阶段失败"
    if event.error:
        return f"{header}\n\n原因：{event.error}"
    return header
