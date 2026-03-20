"""Reference single-agent executor built on model/tools/skills ports."""

from __future__ import annotations

import json as _json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .loop_guard import LoopGuard, LoopGuardConfig
from .model import ModelMessage, ModelProvider, ModelRequest, ModelResponse, ModelToolCall
from .skills import SkillPack, merge_leases, merge_prompts
from .tools import ToolContext, ToolRegistry
from .types import ExecutionContext, TaskContract, TaskResult, ToolLease
from ..context import _extract_keywords

logger = logging.getLogger(__name__)


@dataclass
class ExecutorPolicy:
    """Policy for one single-agent execution loop."""

    max_steps: int = 8
    max_continuations: int = 2
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)


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

        tape = context.state.get("tape")
        if tape is not None:
            history_budget = context.state.get("context_history_tokens", 2000)
            tape_store = context.state.get("tape_store")
            messages.extend(_build_history_messages(
                tape, history_budget,
                query=task.description,
                tape_store=tape_store,
            ))

        media_paths = context.state.get("media_paths") or ()
        messages.append(ModelMessage(
            role="user",
            content=task.description,
            images=tuple(media_paths),
        ))

        available_tools = self.tools.tool_schemas(base_lease)
        tool_names = [t["function"]["name"] for t in available_tools]
        if not tool_names:
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

        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        max_model_tokens = int(context.state.get("max_model_tokens", 0) or 0)
        loop_guard = LoopGuard(self.policy.loop_guard)
        blocked_tool_names: set[str] = set()

        tool_context = ToolContext(session_id=context.session_id, state=context.state)
        heartbeat = context.state.get("heartbeat")
        for step in range(1, max(1, self.policy.max_steps) + 1):
            if heartbeat is not None:
                heartbeat.beat()
            context.emit("executor.step", task_id=task.task_id, step=step)
            logger.info(
                "Executor step=%d/%d task=%s",
                step, self.policy.max_steps, task.task_id,
            )

            messages = loop_guard.compress_messages(messages)
            if blocked_tool_names:
                step_tools = [
                    t for t in available_tools
                    if t["function"]["name"] not in blocked_tool_names
                ]
            else:
                step_tools = available_tools
            response = await self.model.generate(
                ModelRequest(
                    messages=tuple(messages),
                    tools=step_tools,
                    metadata={"task_id": task.task_id, "step": step},
                ),
                context,
            )
            self._accumulate_usage(usage_totals, response)
            budget_error = self._check_token_budget(usage_totals, max_model_tokens)
            if budget_error:
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=budget_error,
                    metadata=self._build_usage_metadata(
                        usage_totals,
                        extra={"max_model_tokens": max_model_tokens},
                    ),
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
                # Phase 1: serial validation — lightweight checks
                error_results: list[tuple[ModelToolCall, str]] = []
                valid_calls: list[tuple[ModelToolCall, Any]] = []

                for tool_call in response.tool_calls:
                    verdict = loop_guard.check_call(tool_call.name, tool_call.arguments)
                    if verdict.blocked:
                        logger.warning(
                            "Executor loop guard blocked task=%s tool=%s reason=%s",
                            task.task_id, tool_call.name, verdict.reason,
                        )
                        blocked_tool_names.add(tool_call.name)
                        remaining = [
                            t["function"]["name"] for t in available_tools
                            if t["function"]["name"] not in blocked_tool_names
                        ]
                        hint = (
                            f"Loop guard: {verdict.reason}"
                            f"\nTool '{tool_call.name}' is now disabled for this task."
                            f"\nYou MUST use a different tool. Available tools: {remaining}"
                        )
                        error_results.append((tool_call, hint))
                        continue

                    registered = self.tools.get(tool_call.name)
                    if not registered or not self._tool_allowed(
                        registered.group, tool_call.name, base_lease,
                    ):
                        logger.warning(
                            "Executor tool unavailable task=%s tool=%s",
                            task.task_id, tool_call.name,
                        )
                        error_results.append((tool_call, (
                            f"Tool unavailable: {tool_call.name}"
                            "\n[Hint: This tool is not available. Use a different tool or approach.]"
                        )))
                        continue

                    if (
                        isinstance(tool_call.arguments, dict)
                        and tool_call.arguments.get("__tool_argument_parse_error__")
                    ):
                        logger.warning(
                            "Executor invalid arguments JSON task=%s tool=%s",
                            task.task_id, tool_call.name,
                        )
                        error_results.append((tool_call, (
                            f"Tool argument JSON parse error for {tool_call.name}: "
                            f"{tool_call.arguments.get('__raw_arguments__', '')}"
                            "\n[Hint: Fix the JSON syntax in your tool arguments.]"
                        )))
                        continue

                    tool_call.arguments = self._cast_tool_arguments(
                        schema=registered.tool.schema,
                        args=tool_call.arguments,
                    )
                    validation_error = self._validate_tool_arguments(
                        schema=registered.tool.schema,
                        args=tool_call.arguments,
                    )
                    if validation_error:
                        logger.warning(
                            "Executor argument validation failed task=%s tool=%s error=%s",
                            task.task_id, tool_call.name, validation_error,
                        )
                        error_results.append((tool_call, (
                            f"Tool argument validation failed for {tool_call.name}: {validation_error}"
                            "\n[Hint: Check parameter types and values against the tool schema.]"
                        )))
                        continue

                    valid_calls.append((tool_call, registered))

                # Build a map of tool_call index → result message for ordering
                tool_result_map: dict[str, ModelMessage] = {}

                # Append error results immediately
                for tc, err_output in error_results:
                    tool_result_map[tc.call_id] = ModelMessage(
                        role="tool",
                        name=tc.name,
                        content=err_output,
                        tool_call_id=tc.call_id,
                    )

                # Phase 2: parallel execution of validated tool calls
                if valid_calls:
                    import asyncio as _aio

                    def _collect_media(paths: list[str]) -> None:
                        if not paths:
                            return
                        bucket = context.state.setdefault("media_paths_collected", [])
                        existing = set(bucket)
                        for path in paths:
                            if path and path not in existing:
                                bucket.append(path)
                                existing.add(path)

                    async def _invoke_one(
                        tc: ModelToolCall,
                        reg: Any,
                    ) -> tuple[ModelToolCall, str, list[str]]:
                        logger.info(
                            "Executor invoke task=%s tool=%s args_keys=%s",
                            task.task_id, tc.name,
                            list(tc.arguments.keys()),
                        )
                        started = time.perf_counter()
                        result = await reg.tool.invoke(tc.arguments, tool_context)
                        elapsed = time.perf_counter() - started
                        output = result.content if result.ok else (
                            f"Tool error: {result.error}"
                            "\n[Hint: Analyze the error and try a different approach instead of retrying the same way.]"
                        )
                        if result.ok:
                            logger.info(
                                "Executor tool done task=%s tool=%s ok=%s elapsed=%.2fs output_len=%d",
                                task.task_id, tc.name, result.ok,
                                elapsed, len(output),
                            )
                        else:
                            logger.warning(
                                "Executor tool failed task=%s tool=%s elapsed=%.2fs error=%s",
                                task.task_id,
                                tc.name,
                                elapsed,
                                (result.error or "")[:500],
                            )
                        return tc, output, list(result.artifacts or [])

                    done = await _aio.gather(*[_invoke_one(tc, reg) for tc, reg in valid_calls])
                    for tc, output, artifacts in done:
                        _collect_media(artifacts)
                        tool_result_map[tc.call_id] = ModelMessage(
                            role="tool",
                            name=tc.name,
                            content=output,
                            tool_call_id=tc.call_id,
                        )

                # Append tool results in original tool_calls order
                for tool_call in response.tool_calls:
                    if tool_call.call_id in tool_result_map:
                        messages.append(tool_result_map[tool_call.call_id])

                if heartbeat is not None:
                    heartbeat.beat()
                continue

            text = response.text
            if response.finish_reason == "length":
                text = await self._continue_from_truncation(
                    initial_text=text,
                    messages=messages,
                    context=context,
                    task_id=task.task_id,
                    step=step,
                    usage_totals=usage_totals,
                    max_model_tokens=max_model_tokens,
                )
                if text is None:
                    return TaskResult(
                        task_id=task.task_id,
                        status="failed",
                        error=(
                            f"Model token budget exceeded "
                            f"({usage_totals['total_tokens']}/{max_model_tokens})."
                        ),
                        metadata=self._build_usage_metadata(
                            usage_totals,
                            extra={"max_model_tokens": max_model_tokens},
                        ),
                    )

            text = text.strip()
            if text:
                logger.info(
                    "Executor final answer task=%s step=%d output_len=%d",
                    task.task_id, step, len(text),
                )
                return TaskResult(
                    task_id=task.task_id,
                    status="succeeded",
                    output=text,
                    metadata=self._build_usage_metadata(usage_totals),
                )

        logger.warning(
            "Executor exhausted steps task=%s max_steps=%d",
            task.task_id, self.policy.max_steps,
        )
        return TaskResult(
            task_id=task.task_id,
            status="failed",
            error=f"No terminal answer within {self.policy.max_steps} steps.",
            metadata=self._build_usage_metadata(
                usage_totals,
                extra={"history": [self._dump_message(message) for message in messages]},
            ),
        )

    async def _continue_from_truncation(
        self,
        *,
        initial_text: str,
        messages: list[ModelMessage],
        context: ExecutionContext,
        task_id: str,
        step: int,
        usage_totals: dict[str, int],
        max_model_tokens: int,
    ) -> str | None:
        text = initial_text or ""
        finish_reason = "length"
        for continuation_idx in range(1, max(1, self.policy.max_continuations) + 1):
            if finish_reason != "length":
                break
            messages.append(ModelMessage(role="assistant", content=text))
            messages.append(ModelMessage(role="user", content="Continue from where you left off."))
            response = await self.model.generate(
                ModelRequest(
                    messages=tuple(messages),
                    metadata={
                        "task_id": task_id,
                        "step": step,
                        "continuation": continuation_idx,
                    },
                ),
                context,
            )
            self._accumulate_usage(usage_totals, response)
            if self._check_token_budget(usage_totals, max_model_tokens):
                return None
            text += response.text or ""
            finish_reason = response.finish_reason or "stop"
        return text

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
        if include_tools or include_groups:
            in_tools = name in include_tools if include_tools else False
            in_groups = group in include_groups if include_groups else False
            if not (in_tools or in_groups):
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

    @staticmethod
    def _usage_from_response(response: ModelResponse) -> tuple[int, int, int]:
        usage = response.metadata.get("usage", {}) if response.metadata else {}
        if not isinstance(usage, dict):
            return 0, 0, 0
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        return prompt, completion, total

    @classmethod
    def _accumulate_usage(
        cls,
        totals: dict[str, int],
        response: ModelResponse,
    ) -> None:
        prompt, completion, total = cls._usage_from_response(response)
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["total_tokens"] += total

    @staticmethod
    def _check_token_budget(totals: dict[str, int], max_model_tokens: int) -> str:
        if max_model_tokens <= 0:
            return ""
        if totals["total_tokens"] <= max_model_tokens:
            return ""
        return (
            f"Model token budget exceeded "
            f"({totals['total_tokens']}/{max_model_tokens})."
        )

    @staticmethod
    def _build_usage_metadata(
        totals: dict[str, int],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
        }
        if extra:
            metadata.update(extra)
        return metadata

    @staticmethod
    def _cast_tool_arguments(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        """Best-effort type coercion based on the tool schema."""
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            return args
        casted = dict(args)
        for name, value in casted.items():
            prop = properties.get(name)
            if not isinstance(prop, dict):
                continue
            expected = prop.get("type")
            if not expected or value is None:
                continue
            if expected == "integer" and isinstance(value, str):
                try:
                    casted[name] = int(value)
                except ValueError:
                    pass
            elif expected == "number" and isinstance(value, str):
                try:
                    casted[name] = float(value)
                except ValueError:
                    pass
            elif expected == "boolean" and isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    casted[name] = True
                elif value.lower() in ("false", "0", "no"):
                    casted[name] = False
            elif expected == "array" and isinstance(value, str):
                try:
                    casted[name] = _json.loads(value)
                except (_json.JSONDecodeError, ValueError):
                    pass
        return casted

    @staticmethod
    def _validate_tool_arguments(
        schema: dict[str, Any],
        args: Any,
    ) -> str | None:
        if not isinstance(args, dict):
            return "arguments must be a JSON object"

        required = schema.get("required") or []
        if isinstance(required, list):
            missing = [name for name in required if name not in args]
            if missing:
                return f"missing required fields: {', '.join(missing)}"

        properties = schema.get("properties") or {}
        additional_allowed = schema.get("additionalProperties", True)
        if isinstance(properties, dict) and additional_allowed is False:
            extra = [key for key in args if key not in properties]
            if extra:
                return f"unexpected fields: {', '.join(extra)}"

        if isinstance(properties, dict):
            for name, value in args.items():
                prop = properties.get(name)
                if not isinstance(prop, dict):
                    continue
                expected = prop.get("type")
                if not expected:
                    continue
                if expected == "string" and not isinstance(value, str):
                    return f"field '{name}' must be string"
                if expected == "integer" and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    return f"field '{name}' must be integer"
                if expected == "number" and (
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                ):
                    return f"field '{name}' must be number"
                if expected == "boolean" and not isinstance(value, bool):
                    return f"field '{name}' must be boolean"
                if expected == "array" and not isinstance(value, list):
                    return f"field '{name}' must be array"
                if expected == "object" and not isinstance(value, dict):
                    return f"field '{name}' must be object"
        return None


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
            anchor_text = "\n".join(parts)
            anchor_cost = len(anchor_text) // 3
            messages.append(ModelMessage(role="system", content=anchor_text))
            budget_remaining = max(0, int(token_budget) - anchor_cost)
        else:
            budget_remaining = max(0, int(token_budget))
    else:
        budget_remaining = max(0, int(token_budget))

    # Collect recent entries (for both section 2 exclusion and section 3)
    entries_since = getattr(tape, "entries_since_anchor", None)
    if entries_since is None:
        return messages

    recent = entries_since()
    msg_entries = [e for e in recent if e.kind == "message"]
    # Exclude the last user message (it's the current turn, added by executor)
    if msg_entries and msg_entries[-1].payload.get("role") == "user":
        msg_entries = msg_entries[:-1]

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
