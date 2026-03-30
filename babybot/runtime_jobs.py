"""Runtime job models for long-running orchestration work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .feedback_events import normalize_runtime_feedback_event


JobState = Literal[
    "queued",
    "planning",
    "running",
    "waiting_tool",
    "waiting_user",
    "repairing",
    "completed",
    "failed",
    "cancelled",
]

JOB_STATES: tuple[JobState, ...] = (
    "queued",
    "planning",
    "running",
    "waiting_tool",
    "waiting_user",
    "repairing",
    "completed",
    "failed",
    "cancelled",
)

ACTIVE_JOB_STATES: tuple[JobState, ...] = (
    "queued",
    "planning",
    "running",
    "waiting_tool",
    "waiting_user",
    "repairing",
)

TERMINAL_JOB_STATES: tuple[JobState, ...] = (
    "completed",
    "failed",
    "cancelled",
)


@dataclass(frozen=True)
class RuntimeJob:
    job_id: str
    chat_key: str
    goal: str
    plan_id: str = ""
    state: JobState = "queued"
    progress_message: str = ""
    result_text: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def project_job_state_from_runtime_event(event_payload: dict[str, Any]) -> tuple[JobState, str]:
    normalized = normalize_runtime_feedback_event(event_payload)
    state: JobState = (
        normalized.state if normalized.state in JOB_STATES else "running"
    )
    if normalized.task_id and normalized.stage not in {"job", "interactive_session"}:
        if state == "completed":
            state = "running"
        elif state == "failed":
            state = "repairing"
        elif state == "cancelled":
            state = "running"
    progress_message = normalized.message or normalized.stage or state
    return state, progress_message
