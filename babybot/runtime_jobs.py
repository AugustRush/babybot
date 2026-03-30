"""Runtime job models for long-running orchestration work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
