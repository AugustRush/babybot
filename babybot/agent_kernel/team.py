# babybot/agent_kernel/team.py
"""Agent team primitives: Mailbox for inter-agent messaging, SharedTaskList for coordination."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = ["Mailbox", "MailMessage", "SharedTaskList", "TeamTask"]


@dataclass
class MailMessage:
    sender: str
    content: str


@dataclass
class TeamTask:
    task_id: str
    description: str
    deps: list[str] = field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    assigned_to: str = ""
    output: str = ""


class Mailbox:
    """In-process async mailbox for agent-to-agent messaging."""

    def __init__(self) -> None:
        self._boxes: dict[str, list[MailMessage]] = {}
        self._events: dict[str, asyncio.Event] = {}

    def _ensure(self, recipient: str) -> None:
        if recipient not in self._boxes:
            self._boxes[recipient] = []
            self._events[recipient] = asyncio.Event()

    def send(self, sender: str, recipient: str, content: str) -> None:
        self._ensure(recipient)
        self._boxes[recipient].append(MailMessage(sender=sender, content=content))
        self._events[recipient].set()

    def broadcast(self, sender: str, recipients: list[str], content: str) -> None:
        for r in recipients:
            if r != sender:
                self.send(sender, r, content)

    def receive(self, recipient: str) -> list[MailMessage]:
        self._ensure(recipient)
        messages = list(self._boxes[recipient])
        self._boxes[recipient].clear()
        self._events[recipient].clear()
        return messages

    async def wait_for_message(
        self, recipient: str, timeout: float | None = None
    ) -> MailMessage:
        self._ensure(recipient)
        if not self._boxes[recipient]:
            self._events[recipient].clear()
            await asyncio.wait_for(self._events[recipient].wait(), timeout=timeout)
        return self._boxes[recipient].pop(0)


class SharedTaskList:
    """Shared task list with dependency tracking for agent teams."""

    def __init__(self) -> None:
        self._tasks: dict[str, TeamTask] = {}
        self._order: list[str] = []

    def add(self, task: TeamTask) -> None:
        self._tasks[task.task_id] = task
        self._order.append(task.task_id)

    def _deps_met(self, task: TeamTask) -> bool:
        return all(
            self._tasks.get(dep, TeamTask(task_id=dep, description="")).status
            == "completed"
            for dep in task.deps
        )

    def claim(self, agent_id: str) -> TeamTask | None:
        for tid in self._order:
            task = self._tasks[tid]
            if task.status == "pending" and self._deps_met(task):
                task.status = "in_progress"
                task.assigned_to = agent_id
                return task
        return None

    def complete(self, task_id: str, output: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = "completed"
            task.output = output

    def fail(self, task_id: str, error: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = "failed"
            task.output = error

    def get_status(self) -> dict[str, TeamTask]:
        return dict(self._tasks)

    def all_done(self) -> bool:
        return all(t.status in ("completed", "failed") for t in self._tasks.values())

    def summary(self) -> str:
        lines = []
        for tid in self._order:
            t = self._tasks[tid]
            marker = {
                "pending": " ",
                "in_progress": ">",
                "completed": "x",
                "failed": "!",
            }
            lines.append(f"[{marker.get(t.status, '?')}] {t.task_id}: {t.description}")
        return "\n".join(lines)
