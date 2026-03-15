"""Reference single-agent executor built on model/tools/skills ports."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .model import ModelMessage, ModelProvider, ModelRequest, ModelResponse, ModelToolCall
from .skills import SkillPack, merge_leases, merge_prompts
from .tools import ToolContext, ToolRegistry
from .types import ExecutionContext, TaskContract, TaskResult, ToolLease

logger = logging.getLogger(__name__)


@dataclass
class ExecutorPolicy:
    """Policy for one single-agent execution loop."""

    max_steps: int = 8


@dataclass
class SingleAgentExecutor:
    """ExecutorPort implementation.

    It runs a compact agent loop:
    model -> tool_calls -> tool_results -> model ... -> final text
    """

    model: ModelProvider
    tools: ToolRegistry
    skill_resolver: Callable[[TaskContract, ExecutionContext], SkillPack | Iterable[SkillPack] | None] | None = None
    policy: ExecutorPolicy = field(default_factory=ExecutorPolicy)

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        skills = self._resolve_skills(task, context)
        base_lease = task.lease
        for skill in skills:
            base_lease = merge_leases(base_lease, skill.tool_lease)

        system_prompt = merge_prompts(skills)
        messages: list[ModelMessage] = []
        if system_prompt:
            messages.append(ModelMessage(role="system", content=system_prompt))

        # Inject history from tape (anchor summary + recent entries + BM25 recall)
        tape = context.state.get("tape")
        if tape is not None:
            history_budget = context.state.get("context_history_tokens", 2000)
            tape_store = context.state.get("tape_store")
            messages.extend(_build_history_messages(
                tape, history_budget,
                query=task.description,
                tape_store=tape_store,
            ))

        messages.append(ModelMessage(role="user", content=task.description))

        available_tools = self.tools.tool_schemas(base_lease)
        tool_names = [t["function"]["name"] for t in available_tools]
        if not tool_names:
            # Debug: dump registry contents when no tools found
            all_tools = {n: rt.group for n, rt in self.tools._tools.items()}
            logger.warning(
                "Executor NO TOOLS task=%s lease=%s registry_tools=%s",
                task.task_id, base_lease, all_tools,
            )
        logger.info(
            "Executor start task=%s max_steps=%d tools=%s lease_groups=%s include_tools=%s exclude_tools=%s",
            task.task_id, self.policy.max_steps, tool_names,
            list(base_lease.include_groups),
            list(base_lease.include_tools),
            list(base_lease.exclude_tools),
        )

        tool_context = ToolContext(session_id=context.session_id, state=context.state)
        heartbeat = context.state.get("heartbeat")
        for step in range(1, max(1, self.policy.max_steps) + 1):
            if heartbeat is not None:
                heartbeat.beat()
            context.emit("executor.step", task_id=task.task_id, step=step)
            logger.info("Executor step=%d/%d task=%s", step, self.policy.max_steps, task.task_id)

            response = await self.model.generate(
                ModelRequest(
                    messages=tuple(messages),
                    tools=available_tools,
                    metadata={"task_id": task.task_id, "step": step},
                ),
                context,
            )

            if response.tool_calls:
                logger.info(
                    "Executor tool_calls task=%s step=%d calls=%s",
                    task.task_id, step,
                    [tc.name for tc in response.tool_calls],
                )
                messages.append(
                    ModelMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )
                for tool_call in response.tool_calls:
                    registered = self.tools.get(tool_call.name)
                    if not registered or not self._tool_allowed(registered.group, tool_call.name, base_lease):
                        tool_output = f"Tool unavailable: {tool_call.name}"
                        logger.warning(
                            "Executor tool unavailable task=%s tool=%s",
                            task.task_id, tool_call.name,
                        )
                    else:
                        logger.info(
                            "Executor invoke task=%s tool=%s args_keys=%s",
                            task.task_id, tool_call.name,
                            list(tool_call.arguments.keys()),
                        )
                        started = time.perf_counter()
                        result = await registered.tool.invoke(tool_call.arguments, tool_context)
                        elapsed = time.perf_counter() - started
                        tool_output = result.content if result.ok else f"Tool error: {result.error}"
                        logger.info(
                            "Executor tool done task=%s tool=%s ok=%s elapsed=%.2fs output_len=%d",
                            task.task_id, tool_call.name, result.ok,
                            elapsed, len(tool_output),
                        )
                    if heartbeat is not None:
                        heartbeat.beat()
                    messages.append(
                        ModelMessage(
                            role="tool",
                            name=tool_call.name,
                            content=tool_output,
                            tool_call_id=tool_call.call_id,
                        )
                    )
                continue

            text = response.text.strip()
            if text:
                logger.info(
                    "Executor final answer task=%s step=%d output_len=%d",
                    task.task_id, step, len(text),
                )
                return TaskResult(task_id=task.task_id, status="succeeded", output=text)

        logger.warning(
            "Executor exhausted steps task=%s max_steps=%d",
            task.task_id, self.policy.max_steps,
        )
        return TaskResult(
            task_id=task.task_id,
            status="failed",
            error=f"No terminal answer within {self.policy.max_steps} steps.",
            metadata={"history": [self._dump_message(message) for message in messages]},
        )

    def _resolve_skills(self, task: TaskContract, context: ExecutionContext) -> list[SkillPack]:
        resolver = self.skill_resolver
        if resolver is None:
            return []
        result = resolver(task, context)
        if not result:
            return []
        if isinstance(result, SkillPack):
            return [result]
        return [item for item in result if isinstance(item, SkillPack)]

    @staticmethod
    def _tool_allowed(group: str, name: str, lease: ToolLease) -> bool:
        include_groups = set(lease.include_groups)
        include_tools = set(lease.include_tools)
        exclude_tools = set(lease.exclude_tools)
        if name in exclude_tools:
            return False
        if include_tools and name not in include_tools:
            return False
        if include_groups and group not in include_groups:
            return False
        return True

    @staticmethod
    def _dump_message(message: ModelMessage) -> dict[str, str]:
        payload = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.call_id, "name": tc.name} for tc in message.tool_calls
            ]
        return payload


class EchoModelProvider:
    """Tiny debug model provider for local wiring tests."""

    async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
        last = request.messages[-1] if request.messages else ModelMessage(role="assistant", content="")
        if last.role == "tool":
            return ModelResponse(text=f"Observed: {last.content}")
        if request.tools:
            fn = request.tools[0]["function"]["name"]
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c1",
                        name=fn,
                        arguments={"input": last.content},
                    ),
                ),
            )
        return ModelResponse(text=last.content)


def _build_history_messages(
    tape: object,
    token_budget: int,
    query: str = "",
    tape_store: object | None = None,
) -> list[ModelMessage]:
    """Build history context messages from a Tape.

    Three sections (all sharing token_budget):
    1. Anchor summary → system message
    2. BM25 cross-anchor recall → [相关历史] system message
    3. Recent entries since anchor → user/assistant messages
    """
    messages: list[ModelMessage] = []

    last_anchor = getattr(tape, "last_anchor", None)
    if last_anchor is None:
        return messages
    anchor = last_anchor()

    # 1. Anchor summary → system message (with structured fields if available)
    if anchor is not None:
        state = anchor.payload.get("state", {})
        summary = state.get("summary", "") if isinstance(state, dict) else ""
        if summary:
            parts = [f"[对话背景]\n{summary}"]
            entities = state.get("entities")
            if entities and isinstance(entities, list):
                parts.append(f"关键实体: {', '.join(entities)}")
            intent = state.get("user_intent")
            if intent:
                parts.append(f"用户意图: {intent}")
            pending = state.get("pending")
            if pending:
                parts.append(f"待办: {pending}")
            messages.append(ModelMessage(role="system", content="\n".join(parts)))

    # Collect recent entries (for both section 2 exclusion and section 3)
    entries_since = getattr(tape, "entries_since_anchor", None)
    if entries_since is None:
        return messages

    recent = entries_since()
    msg_entries = [e for e in recent if e.kind == "message"]
    # Exclude the last user message (it's the current turn, added by executor)
    if msg_entries and msg_entries[-1].payload.get("role") == "user":
        msg_entries = msg_entries[:-1]

    budget_remaining = max(0, int(token_budget))

    # 2. BM25 cross-anchor recall — search for relevant entries before the anchor
    search_fn = getattr(tape_store, "search_relevant", None) if tape_store else None
    chat_id = getattr(tape, "chat_id", None)
    if search_fn and chat_id and query:
        recent_ids = {e.entry_id for e in recent}
        recall_budget = budget_remaining // 4  # Reserve up to 25% for recall
        try:
            recalled = search_fn(chat_id, query, limit=5, exclude_ids=recent_ids)
        except Exception:
            recalled = []

        if recalled:
            recall_lines: list[str] = []
            recall_tokens = 0
            for entry in recalled:
                est = max(1, int(entry.token_estimate))
                if recall_tokens + est > recall_budget:
                    break
                role = entry.payload.get("role", "?")
                content = entry.payload.get("content", "")
                recall_lines.append(f"{role}: {content}")
                recall_tokens += est
            if recall_lines:
                messages.append(ModelMessage(
                    role="system",
                    content="[相关历史]\n" + "\n".join(recall_lines),
                ))
                budget_remaining -= recall_tokens

    # 3. Recent entries → hybrid recency+relevance scoring
    if msg_entries and query:
        from ..context import _extract_keywords
        kws = _extract_keywords(query)
    else:
        kws = []

    n = len(msg_entries)
    scored_entries: list[tuple[float, int, object]] = []
    for idx, entry in enumerate(msg_entries):
        # Recency: linear 0→1, most recent = 1.0
        recency = (idx + 1) / n if n else 0.0
        # Relevance: fraction of keywords found in content
        if kws:
            content = entry.payload.get("content", "")
            hits = sum(1 for kw in kws if kw in content)
            relevance = hits / len(kws)
        else:
            relevance = 0.0
        # Weighted blend: recency dominates (0.6) to preserve conversation flow
        score = 0.6 * recency + 0.4 * relevance
        scored_entries.append((score, idx, entry))

    # Sort by score descending, greedily pick within budget
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    picked_indices: set[int] = set()
    for score, idx, entry in scored_entries:
        if budget_remaining <= 0:
            break
        est = max(1, int(entry.token_estimate))
        if est > budget_remaining:
            continue
        budget_remaining -= est
        picked_indices.add(idx)

    # Emit in original chronological order
    for idx, entry in enumerate(msg_entries):
        if idx in picked_indices:
            messages.append(ModelMessage(
                role=entry.payload["role"],
                content=entry.payload["content"],
            ))

    return messages
