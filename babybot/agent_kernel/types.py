"""Core types for the minimal orchestration kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TaskStatus = Literal["pending", "running", "succeeded", "failed", "blocked", "skipped"]


@dataclass(frozen=True)
class ToolLease:
    """Least-privilege tool policy for one task."""

    include_groups: tuple[str, ...] = ()
    include_tools: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()


@dataclass
class TaskContract:
    """Generic task unit in an execution plan."""

    task_id: str
    description: str
    deps: tuple[str, ...] = ()
    lease: ToolLease = field(default_factory=ToolLease)
    timeout_s: float | None = None
    retries: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    """DAG of executable tasks."""

    tasks: tuple[TaskContract, ...]
    rationale: str = ""


@dataclass
class TaskResult:
    """Execution result of one task."""

    task_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""
    artifacts: tuple[str, ...] = ()
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalResult:
    """Final result contract for a complete goal."""

    conclusion: str
    evidence: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    task_results: dict[str, TaskResult] = field(default_factory=dict)


@dataclass
class RunPolicy:
    """Execution policy for orchestration runtime."""

    max_parallel: int = 4
    default_timeout_s: float | None = None
    default_retries: int = 0


@dataclass
class ExecutionContext:
    """Runtime context shared by planner, executor, and synthesizer."""

    session_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append({"event": event, **payload})
