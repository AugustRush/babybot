# babybot/agent_kernel/team.py
"""Agent team primitives: Mailbox for inter-agent messaging, SharedTaskList for coordination."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "Mailbox",
    "MailMessage",
    "SharedTaskList",
    "TeamTask",
    "TeamRunner",
    "DebateResult",
]


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


@dataclass
class DebateResult:
    """Outcome of a structured multi-agent debate."""

    topic: str
    rounds: int
    transcript: list[dict[str, str]]
    summary: str


class TeamRunner:
    """Runs structured multi-agent interactions (e.g., debates).

    The runner alternates between agents, passing each agent the previous
    agent's output.  An optional judge function can signal early convergence.
    """

    def __init__(
        self,
        executor: Any,  # async callable(agent_id, prompt, context) -> str
        max_rounds: int = 5,
    ) -> None:
        self._executor = executor
        self._max_rounds = max_rounds

    async def run_debate(
        self,
        topic: str,
        agents: list[dict[str, str]],
        judge: Any | None = None,
        on_turn: Any | None = None,
    ) -> DebateResult:
        """Run a structured debate.

        *on_turn*, when provided, is called after each agent turn with
        ``(agent_id, role, round_num, output_text)``.  It may be a
        regular or async callable.
        """
        transcript: list[dict[str, str]] = []
        last_output = ""
        agent_ids = [a["id"] for a in agents]
        logger.info(
            "Debate started: topic=%r agents=%s max_rounds=%d",
            topic, agent_ids, self._max_rounds,
        )

        for round_num in range(1, self._max_rounds + 1):
            logger.info("Debate round %d/%d started", round_num, self._max_rounds)
            for agent in agents:
                prompt_parts = [
                    f"Topic: {topic}",
                    f"Your role: {agent['role']} \u2014 {agent['description']}",
                    f"Round: {round_num}/{self._max_rounds}",
                ]
                if last_output:
                    prompt_parts.append(f"Previous argument:\n{last_output}")
                prompt_parts.append("Present your argument:")

                prompt = "\n".join(prompt_parts)
                agent_exec = agent.get("executor", self._executor)
                exec_ctx: dict[str, Any] = {}
                if agent.get("system_prompt"):
                    exec_ctx["system_prompt"] = agent["system_prompt"]
                output = await agent_exec(agent["id"], prompt, exec_ctx)
                logger.info(
                    "Debate turn: round=%d agent=%s role=%s output_len=%d",
                    round_num, agent["id"], agent["role"], len(output),
                )
                transcript.append(
                    {
                        "round": str(round_num),
                        "agent": agent["id"],
                        "role": agent["role"],
                        "content": output,
                    }
                )
                last_output = output

                if on_turn is not None:
                    import inspect as _inspect
                    _result = on_turn(agent["id"], agent["role"], round_num, output)
                    if _inspect.isawaitable(_result):
                        await _result

            # Check convergence via optional judge
            if judge is not None:
                converged, reason = judge(transcript)
                if converged:
                    logger.info(
                        "Debate converged at round %d: %s",
                        round_num, reason[:200],
                    )
                    return DebateResult(
                        topic=topic,
                        rounds=round_num,
                        transcript=transcript,
                        summary=reason,
                    )

        # Max rounds reached -- summarize
        logger.info(
            "Debate completed: topic=%r rounds=%d total_turns=%d",
            topic, self._max_rounds, len(transcript),
        )
        summary_parts = [
            f"Debate on '{topic}' completed after {self._max_rounds} rounds."
        ]
        for entry in transcript[-len(agents) :]:
            summary_parts.append(f"- {entry['role']}: {entry['content'][:200]}")

        return DebateResult(
            topic=topic,
            rounds=self._max_rounds,
            transcript=transcript,
            summary="\n".join(summary_parts),
        )
