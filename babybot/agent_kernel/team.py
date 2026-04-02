# babybot/agent_kernel/team.py
"""Agent team primitives: Mailbox for inter-agent messaging, SharedTaskList for coordination."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from .execution_constraints import TeamExecutionPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "Mailbox",
    "MailMessage",
    "SharedTaskList",
    "TeamTask",
    "TeamRunner",
    "CooperativeResult",
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
    completed: bool = True
    termination_reason: str = "completed"


@dataclass
class CooperativeResult:
    """Outcome of a cooperative (task-based) team execution."""

    topic: str
    tasks_completed: int
    tasks_failed: int
    tasks_total: int
    task_outputs: dict[str, str]
    mailbox_log: list[dict[str, str]]
    summary: str
    completed: bool = True
    termination_reason: str = "completed"


class TeamRunner:
    """Runs structured multi-agent debate-style interactions.

    The runner alternates between agents, passing each agent the previous
    agent's output.  An optional judge function can signal early convergence.
    """

    def __init__(
        self,
        executor: Any,  # async callable(agent_id, prompt, context) -> str
        max_rounds: int = 5,
        policy: TeamExecutionPolicy | None = None,
    ) -> None:
        self._executor = executor
        self._policy = policy or TeamExecutionPolicy(max_rounds=max_rounds)
        self._max_rounds = max(1, int(self._policy.max_rounds or max_rounds))

    @staticmethod
    def _build_partial_summary(
        topic: str,
        transcript: list[dict[str, str]],
        agents: list[dict[str, str]],
        reason: str,
    ) -> str:
        summary_parts = [
            f"Partial debate summary for '{topic}' ({reason}).",
        ]
        if not transcript:
            summary_parts.append(
                "No agent completed a full turn before the budget was exhausted."
            )
            return "\n".join(summary_parts)
        summary_parts.append("Latest completed arguments:")
        for entry in transcript[-len(agents) :]:
            summary_parts.append(f"- {entry['role']}: {entry['content'][:200]}")
        return "\n".join(summary_parts)

    def _partial_result(
        self,
        *,
        topic: str,
        transcript: list[dict[str, str]],
        agents: list[dict[str, str]],
        rounds: int,
        reason: str,
    ) -> DebateResult:
        return DebateResult(
            topic=topic,
            rounds=rounds,
            transcript=transcript,
            summary=self._build_partial_summary(topic, transcript, agents, reason),
            completed=False,
            termination_reason=reason,
        )

    async def run_debate(
        self,
        topic: str,
        agents: list[dict[str, str]],
        judge: Any | None = None,
        on_turn: Any | None = None,
        on_round_start: Any | None = None,
    ) -> DebateResult:
        """Run a structured debate.

        *on_turn*, when provided, is called after each agent turn with
        ``(agent_id, role, round_num, output_text)``.  It may be a
        regular or async callable.
        """
        transcript: list[dict[str, str]] = []
        last_output = ""
        agent_ids = [a["id"] for a in agents]
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + float(self._policy.max_total_seconds)
            if self._policy.max_total_seconds is not None
            else None
        )
        logger.info(
            "Debate started: topic=%r agents=%s max_rounds=%d",
            topic,
            agent_ids,
            self._max_rounds,
        )

        for round_num in range(1, self._max_rounds + 1):
            if deadline is not None and loop.time() >= deadline:
                return self._partial_result(
                    topic=topic,
                    transcript=transcript,
                    agents=agents,
                    rounds=max(0, round_num - 1),
                    reason="budget_exhausted",
                )
            logger.info("Debate round %d/%d started", round_num, self._max_rounds)
            if on_round_start is not None:
                import inspect as _inspect

                _result = on_round_start(round_num, self._max_rounds)
                if _inspect.isawaitable(_result):
                    await _result
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
                if deadline is not None and loop.time() >= deadline:
                    return self._partial_result(
                        topic=topic,
                        transcript=transcript,
                        agents=agents,
                        rounds=max(0, round_num - 1),
                        reason="budget_exhausted",
                    )
                try:
                    if self._policy.max_turn_seconds is not None:
                        output = await asyncio.wait_for(
                            agent_exec(agent["id"], prompt, exec_ctx),
                            timeout=float(self._policy.max_turn_seconds),
                        )
                    else:
                        output = await agent_exec(agent["id"], prompt, exec_ctx)
                except asyncio.TimeoutError:
                    if self._policy.on_budget_exhausted == "raise_timeout":
                        raise
                    return self._partial_result(
                        topic=topic,
                        transcript=transcript,
                        agents=agents,
                        rounds=max(0, round_num - 1),
                        reason="turn_timeout",
                    )
                logger.info(
                    "Debate turn: round=%d agent=%s role=%s output_len=%d",
                    round_num,
                    agent["id"],
                    agent["role"],
                    len(output),
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
                        round_num,
                        reason[:200],
                    )
                    return DebateResult(
                        topic=topic,
                        rounds=round_num,
                        transcript=transcript,
                        summary=reason,
                        completed=True,
                        termination_reason="judge_converged",
                    )

        # Max rounds reached -- summarize
        logger.info(
            "Debate completed: topic=%r rounds=%d total_turns=%d",
            topic,
            self._max_rounds,
            len(transcript),
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

    async def run_cooperative(
        self,
        topic: str,
        agents: list[dict[str, str]],
        tasks: list[dict[str, Any]],
        *,
        on_task_complete: Any | None = None,
    ) -> CooperativeResult:
        """Run cooperative task-based execution.

        Each agent claims tasks from a SharedTaskList and executes them.
        Agents communicate results via Mailbox broadcasts so downstream
        tasks (declared via ``deps``) can use upstream outputs.

        Args:
            topic: High-level goal description.
            agents: Agent definitions (same format as ``run_debate``).
            tasks: List of task dicts with ``task_id``, ``description``,
                   and optional ``deps`` (list of upstream task_ids).
            on_task_complete: Optional callback ``(agent_id, task_id, output)``
                             invoked after each task finishes.
        """
        import inspect as _inspect

        mailbox = Mailbox()
        task_list = SharedTaskList()

        for task_def in tasks:
            task_list.add(
                TeamTask(
                    task_id=task_def["task_id"],
                    description=task_def["description"],
                    deps=list(task_def.get("deps") or []),
                )
            )

        agent_ids = [a["id"] for a in agents]
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + float(self._policy.max_total_seconds)
            if self._policy.max_total_seconds is not None
            else None
        )
        mailbox_log: list[dict[str, str]] = []

        logger.info(
            "Cooperative started: topic=%r agents=%s tasks=%d",
            topic,
            agent_ids,
            len(tasks),
        )

        async def _agent_worker(agent: dict[str, str]) -> None:
            """Worker loop: claim → execute → broadcast result → repeat."""
            aid = agent["id"]
            agent_exec = agent.get("executor", self._executor)

            while True:
                if deadline is not None and loop.time() >= deadline:
                    break

                claimed = task_list.claim(aid)
                if claimed is None:
                    if task_list.all_done():
                        break
                    # No task available yet — wait a bit and retry.
                    await asyncio.sleep(0.05)
                    if task_list.all_done():
                        break
                    continue

                # Build prompt with upstream context from mailbox + completed deps.
                upstream_msgs = mailbox.receive(aid)
                upstream_context = ""
                # Combine mailbox messages with direct dep outputs from task list.
                upstream_parts: list[str] = []
                if upstream_msgs:
                    upstream_parts.extend(
                        f"  [{m.sender}]: {m.content[:300]}" for m in upstream_msgs
                    )
                # Also inject completed dep task outputs directly — this ensures
                # context flows even when the same agent executed both tasks
                # (broadcasts exclude the sender).
                for dep_tid in claimed.deps:
                    dep_task = task_list._tasks.get(dep_tid)
                    if dep_task and dep_task.status == "completed" and dep_task.output:
                        dep_snippet = dep_task.output[:300]
                        dep_line = f"  [dep:{dep_tid}]: {dep_snippet}"
                        if dep_line not in upstream_parts:
                            upstream_parts.append(dep_line)
                if upstream_parts:
                    upstream_context = (
                        "\n上游任务结果：\n" + "\n".join(upstream_parts) + "\n"
                    )

                prompt_parts = [
                    f"Topic: {topic}",
                    f"Your role: {agent['role']} — {agent['description']}",
                    f"Task [{claimed.task_id}]: {claimed.description}",
                ]
                if upstream_context:
                    prompt_parts.append(upstream_context)
                prompt_parts.append("Execute the task and return results:")

                prompt = "\n".join(prompt_parts)
                exec_ctx: dict[str, Any] = {}
                if agent.get("system_prompt"):
                    exec_ctx["system_prompt"] = agent["system_prompt"]

                try:
                    if self._policy.max_turn_seconds is not None:
                        output = await asyncio.wait_for(
                            agent_exec(aid, prompt, exec_ctx),
                            timeout=float(self._policy.max_turn_seconds),
                        )
                    else:
                        output = await agent_exec(aid, prompt, exec_ctx)
                except (asyncio.TimeoutError, Exception) as exc:
                    error_msg = f"Task {claimed.task_id} failed: {exc}"
                    logger.warning(
                        "Cooperative task failed: agent=%s task=%s error=%s",
                        aid,
                        claimed.task_id,
                        str(exc)[:200],
                    )
                    task_list.fail(claimed.task_id, error=error_msg)
                    mailbox.broadcast(
                        aid,
                        agent_ids,
                        f"[FAILED] task={claimed.task_id}: {error_msg[:200]}",
                    )
                    msg_entry = {
                        "sender": aid,
                        "type": "task_failed",
                        "task_id": claimed.task_id,
                        "content": error_msg[:200],
                    }
                    mailbox_log.append(msg_entry)
                    continue

                task_list.complete(claimed.task_id, output=output)
                logger.info(
                    "Cooperative task done: agent=%s task=%s output_len=%d",
                    aid,
                    claimed.task_id,
                    len(output),
                )

                # Broadcast result to all agents so dependents can use it.
                broadcast_content = f"[DONE] task={claimed.task_id}: {output[:500]}"
                mailbox.broadcast(aid, agent_ids, broadcast_content)
                msg_entry = {
                    "sender": aid,
                    "type": "task_completed",
                    "task_id": claimed.task_id,
                    "content": output[:500],
                }
                mailbox_log.append(msg_entry)

                if on_task_complete is not None:
                    _result = on_task_complete(aid, claimed.task_id, output)
                    if _inspect.isawaitable(_result):
                        await _result

        # Run all agent workers concurrently.
        max_agents = self._policy.max_agents or len(agents)
        active_agents = agents[:max_agents]
        worker_tasks = [
            asyncio.create_task(_agent_worker(agent)) for agent in active_agents
        ]

        # Wait for all workers to finish or deadline.
        if deadline is not None:
            remaining = max(0.1, deadline - loop.time())
            done, pending = await asyncio.wait(worker_tasks, timeout=remaining)
            for p in pending:
                p.cancel()
        else:
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Collect results.
        status = task_list.get_status()
        task_outputs = {
            tid: t.output for tid, t in status.items() if t.status == "completed"
        }
        completed_count = sum(1 for t in status.values() if t.status == "completed")
        failed_count = sum(1 for t in status.values() if t.status == "failed")
        all_done = task_list.all_done()

        termination_reason = "completed" if all_done else "deadline_exceeded"
        if not all_done and any(t.status == "pending" for t in status.values()):
            termination_reason = "deadline_exceeded"

        summary_parts = [
            f"Cooperative execution on '{topic}': "
            f"{completed_count}/{len(tasks)} completed, {failed_count} failed.",
        ]
        for tid, t in status.items():
            marker = {
                "completed": "x",
                "failed": "!",
                "in_progress": ">",
                "pending": " ",
            }
            summary_parts.append(
                f"[{marker.get(t.status, '?')}] {tid}: {t.output[:150] if t.output else t.status}"
            )

        logger.info(
            "Cooperative finished: topic=%r completed=%d failed=%d total=%d",
            topic,
            completed_count,
            failed_count,
            len(tasks),
        )

        return CooperativeResult(
            topic=topic,
            tasks_completed=completed_count,
            tasks_failed=failed_count,
            tasks_total=len(tasks),
            task_outputs=task_outputs,
            mailbox_log=mailbox_log,
            summary="\n".join(summary_parts),
            completed=all_done and failed_count == 0,
            termination_reason=termination_reason,
        )
