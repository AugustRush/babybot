"""Orchestrator built on lightweight kernel — DAG-driven multi-agent mode."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .agent_kernel import ExecutionContext
from .agent_kernel.dynamic_orchestrator import (
    DynamicOrchestrator,
    InMemoryChildTaskBus,
)
from .config import Config
from .context import Tape, TapeStore, _extract_keywords
from .context_views import build_context_view
from .memory_store import HybridMemoryStore
from .heartbeat import TaskHeartbeatRegistry
from .interactive_sessions import InteractiveSessionManager
from .interactive_sessions.backends import ClaudeInteractiveBackend
from .model_gateway import OpenAICompatibleGateway
from .orchestration_policy_store import OrchestrationPolicyStore
from .orchestration_policy_types import PolicyDecisionRecord, PolicyOutcomeRecord
from .resource import ResourceManager

if TYPE_CHECKING:
    from .heartbeat import Heartbeat

logger = logging.getLogger(__name__)
StreamTextCallback = Callable[[str], Awaitable[None] | None]

_SUMMARIZE_PROMPT = (
    "请将以下对话历史浓缩为 JSON 格式（用中文填写），严格按以下结构输出，不要输出其他内容：\n"
    '{"summary":"不超过200字的摘要，保留关键事实和已完成操作",'
    '"entities":["提到的关键实体，如人名、物品、话题等，最多5个"],'
    '"user_intent":"用户当前最可能的意图，一句话",'
    '"pending":"未完成的事项，如无则为空字符串",'
    '"next_steps":["建议的下一步，最多3条"],'
    '"artifacts":["重要产物文件名或标识，最多5条"],'
    '"open_questions":["仍需用户确认的问题，最多3条"],'
    '"decisions":["已经确认的重要决定，最多3条"]}\n\n'
)


@dataclass
class TaskResponse:
    """Structured response from process_task with text and optional media."""

    text: str = ""
    media_paths: list[str] = field(default_factory=list)


class OrchestratorAgent:
    _FLOW_CACHE_LIMIT = 256
    _HANDOFF_LOCK_LIMIT = 256
    _DYNAMIC_ORCHESTRATOR_PARAMETERS: set[str] | None = None

    """Orchestrator — dynamic multi-agent mode via DynamicOrchestrator."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.model.validate()
        self.resource_manager = ResourceManager(self.config)
        self.gateway = OpenAICompatibleGateway(self.config)
        self._interactive_sessions = self._build_interactive_session_manager()
        self.tape_store = TapeStore(
            db_path=self.config.home_dir / "memory" / "context.db",
            max_chats=self.config.system.context_max_chats,
        )
        self.memory_store = HybridMemoryStore(
            db_path=self.config.home_dir / "memory" / "context.db",
            memory_dir=self.config.home_dir / "memory",
        )
        self.memory_store.ensure_bootstrap()
        self.resource_manager.memory_store = self.memory_store
        self.resource_manager.set_observability_provider(self)
        self._policy_store = OrchestrationPolicyStore(
            self.config.home_dir / "memory" / "policy.db"
        )
        self._child_task_bus = InMemoryChildTaskBus()
        self._task_heartbeat_registry = TaskHeartbeatRegistry()
        self._handoff_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._recent_flow_ids_by_chat: OrderedDict[str, str] = OrderedDict()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _build_interactive_session_manager(self) -> InteractiveSessionManager:
        return InteractiveSessionManager(
            backends={
                "claude": ClaudeInteractiveBackend(
                    workspace_root=self.config.workspace_dir,
                )
            },
            max_age_seconds=self.config.system.interactive_session_max_age_seconds,
        )

    def _remember_flow_id(self, chat_key: str, flow_id: str) -> None:
        recent = getattr(self, "_recent_flow_ids_by_chat", None)
        if not isinstance(recent, OrderedDict):
            recent = OrderedDict(recent or {})
            self._recent_flow_ids_by_chat = recent
        self._recent_flow_ids_by_chat.pop(chat_key, None)
        self._recent_flow_ids_by_chat[chat_key] = flow_id
        while len(self._recent_flow_ids_by_chat) > self._FLOW_CACHE_LIMIT:
            self._recent_flow_ids_by_chat.popitem(last=False)

    def _get_handoff_lock(self, chat_key: str) -> asyncio.Lock:
        if not isinstance(self._handoff_locks, OrderedDict):
            self._handoff_locks = OrderedDict(self._handoff_locks)
        lock = self._handoff_locks.pop(chat_key, None)
        if lock is None:
            lock = asyncio.Lock()
        self._handoff_locks[chat_key] = lock
        while len(self._handoff_locks) > self._HANDOFF_LOCK_LIMIT:
            self._handoff_locks.popitem(last=False)
        return lock

    def _spawn_background_task(
        self,
        coro: Awaitable[Any],
        *,
        label: str,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        background_tasks = getattr(self, "_background_tasks", None)
        if background_tasks is None:
            background_tasks = set()
            self._background_tasks = background_tasks
        background_tasks.add(task)

        def _on_done(done: asyncio.Task[Any]) -> None:
            getattr(self, "_background_tasks", set()).discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Background task failed: %s", label)

        task.add_done_callback(_on_done)
        return task

    def _policy_learning_enabled(self) -> bool:
        system = getattr(getattr(self, "config", None), "system", None)
        return bool(getattr(system, "policy_learning_enabled", False))

    @staticmethod
    def _build_policy_state_features(
        user_input: str,
        *,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        text = str(user_input or "").strip()
        task_shape = "single_step"
        if any(token in text for token in ("然后", "再", "并且", "同时", "先")):
            task_shape = "multi_step"
        return {
            "task_shape": task_shape,
            "input_length": len(text),
            "has_media": bool(media_paths),
        }

    def _record_policy_decision(self, record: PolicyDecisionRecord) -> None:
        if not self._policy_learning_enabled() or not record.chat_key:
            return
        self._policy_store.record_decision(
            flow_id=record.flow_id,
            chat_key=record.chat_key,
            decision_kind=record.decision_kind,
            action_name=record.action_name,
            state_features=record.state_features,
        )

    def _record_policy_outcome(self, record: PolicyOutcomeRecord) -> None:
        if not self._policy_learning_enabled() or not record.chat_key:
            return
        self._policy_store.record_outcome(
            flow_id=record.flow_id,
            chat_key=record.chat_key,
            final_status=record.final_status,
            reward=record.reward,
            outcome=record.outcome,
        )

    def _persist_policy_events(
        self,
        *,
        flow_id: str,
        chat_key: str,
        events: list[dict[str, Any]],
    ) -> None:
        if not self._policy_learning_enabled() or not chat_key:
            return
        for event in events:
            if event.get("event") != "policy_decision":
                continue
            self._record_policy_decision(
                PolicyDecisionRecord(
                    flow_id=flow_id,
                    chat_key=chat_key,
                    decision_kind=str(event.get("decision_kind", "") or "").strip(),
                    action_name=str(event.get("action_name", "") or "").strip(),
                    state_features=dict(event.get("state_features") or {}),
                )
            )

    @staticmethod
    def _policy_reward(events: list[dict[str, Any]], final_status: str) -> float:
        reward = 1.0 if final_status == "succeeded" else -1.0
        retry_count = sum(1 for event in events if event.get("event") == "retrying")
        dead_letter_count = sum(
            1 for event in events if event.get("event") == "dead_lettered"
        )
        stalled_count = sum(1 for event in events if event.get("event") == "stalled")
        reward -= 0.15 * retry_count
        reward -= 0.25 * dead_letter_count
        reward -= 0.2 * stalled_count
        return max(-1.0, min(1.0, reward))

    async def _answer_with_dag(
        self,
        user_input: str,
        tape: Tape | None = None,
        heartbeat: Heartbeat | None = None,
        media_paths: list[str] | None = None,
        stream_callback: StreamTextCallback | None = None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None = None,
        send_intermediate_message: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, list[str]]:
        orchestrator_kwargs: dict[str, Any] = {
            "resource_manager": self.resource_manager,
            "gateway": self.gateway,
        }
        parameters = self._DYNAMIC_ORCHESTRATOR_PARAMETERS
        if parameters is None:
            try:
                parameters = set(inspect.signature(DynamicOrchestrator).parameters)
            except (TypeError, ValueError):
                parameters = set()
            self._DYNAMIC_ORCHESTRATOR_PARAMETERS = parameters
        optional_kwargs = {
            "child_task_bus": getattr(self, "_child_task_bus", None),
            "task_heartbeat_registry": getattr(self, "_task_heartbeat_registry", None),
            "task_stale_after_s": float(self.config.system.idle_timeout),
            "max_steps": getattr(self.config.system, "orchestrator_max_steps", 30),
        }
        for key, value in optional_kwargs.items():
            if key not in parameters or value is None:
                continue
            orchestrator_kwargs[key] = value

        orchestrator = DynamicOrchestrator(**orchestrator_kwargs)
        flow_id = f"orchestrator:{uuid.uuid4().hex[:12]}"
        chat_key = getattr(tape, "chat_id", "") if tape is not None else ""
        if chat_key:
            self._remember_flow_id(chat_key, flow_id)

        context = ExecutionContext(
            session_id=flow_id,
            state={
                k: v
                for k, v in [
                    ("tape", tape),
                    ("tape_store", self.tape_store if tape else None),
                    ("memory_store", self.memory_store if tape else None),
                    ("heartbeat", heartbeat),
                    ("media_paths", media_paths),
                    ("original_goal", user_input),
                    (
                        "context_history_tokens",
                        self.config.system.context_history_tokens,
                    ),
                    ("stream_callback", stream_callback),
                    ("runtime_event_callback", runtime_event_callback),
                    ("send_intermediate_message", send_intermediate_message),
                ]
                if v is not None
            },
        )
        if chat_key:
            context.emit(
                "policy_decision",
                decision_kind="decomposition",
                action_name="baseline",
                state_features=self._build_policy_state_features(
                    user_input,
                    media_paths=media_paths,
                ),
            )

        logger.info("DynamicOrchestrator created, starting run flow_id=%s", flow_id)
        try:
            if heartbeat is not None:
                result = await heartbeat.watch(
                    orchestrator.run(goal=user_input, context=context),
                )
            else:
                result = await orchestrator.run(goal=user_input, context=context)
        except Exception as exc:
            self._persist_policy_events(
                flow_id=flow_id,
                chat_key=chat_key,
                events=list(context.events),
            )
            self._record_policy_outcome(
                PolicyOutcomeRecord(
                    flow_id=flow_id,
                    chat_key=chat_key,
                    final_status="failed",
                    reward=self._policy_reward(context.events, "failed"),
                    outcome={"error": str(exc)},
                )
            )
            raise
        self._persist_policy_events(
            flow_id=flow_id,
            chat_key=chat_key,
            events=list(context.events),
        )
        self._record_policy_outcome(
            PolicyOutcomeRecord(
                flow_id=flow_id,
                chat_key=chat_key,
                final_status="succeeded",
                reward=self._policy_reward(context.events, "succeeded"),
                outcome={
                    "task_result_count": len(
                        getattr(result, "task_results", {}) or {}
                    ),
                },
            )
        )

        text = result.conclusion or "任务完成，但没有可返回的结果。"
        collected_media = context.state.get("media_paths_collected", [])
        dedup_media = sorted(set(collected_media))

        return text, dedup_media

    def inspect_runtime_flow(self, flow_id: str = "", chat_key: str = "") -> str:
        resolved_flow_id = flow_id.strip()
        resolved_chat_key = chat_key.strip()
        if not resolved_flow_id and resolved_chat_key:
            resolved_flow_id = self._recent_flow_ids_by_chat.get(resolved_chat_key, "")
        if not resolved_flow_id:
            return "暂无可观测的 flow。"
        snapshot = self._task_heartbeat_registry.snapshot(resolved_flow_id)
        events = self._child_task_bus.events_for(resolved_flow_id)
        parts = ["[Runtime Flow]", f"flow_id={resolved_flow_id}"]
        if resolved_chat_key:
            parts.append(f"chat_key={resolved_chat_key}")
        if snapshot:
            lines = []
            for task_id, state in sorted(snapshot.items()):
                lines.append(
                    f"- task_id={task_id} status={state.get('status', '')} progress={state.get('progress', None)}"
                )
            parts.append("[Tasks]\n" + "\n".join(lines))
        if events:
            lines = []
            for event in events[-12:]:
                payload = dict(event.payload or {})
                status = str(payload.get("status", "") or "")
                progress = payload.get("progress")
                desc = str(payload.get("description", "") or "")
                suffix = []
                if desc:
                    suffix.append(desc)
                if status:
                    suffix.append(f"status={status}")
                if progress is not None:
                    suffix.append(f"progress={progress}")
                lines.append(
                    f"- task_id={event.task_id} event={event.event}"
                    + (f" ({', '.join(suffix)})" if suffix else "")
                )
            parts.append("[Recent Events]\n" + "\n".join(lines))
        if len(parts) == 2:
            parts.append("暂无 task/event 快照。")
        return "\n".join(parts)

    def inspect_chat_context(self, chat_key: str, query: str = "") -> str:
        if not chat_key:
            return "缺少 chat_key。"
        view = build_context_view(
            memory_store=self.memory_store, chat_id=chat_key, query=query
        )
        records = self.memory_store.list_memories(chat_id=chat_key)
        parts = ["[Chat Context]", f"chat_key={chat_key}"]
        if query:
            parts.append(f"query={query}")
        if view.hot:
            parts.append("[Hot Context]\n- " + "\n- ".join(view.hot))
        if view.warm:
            parts.append("[Warm Context]\n- " + "\n- ".join(view.warm))
        if view.cold:
            parts.append("[Cold Context]\n- " + "\n- ".join(view.cold))
        if records:
            lines = [
                f"- memory_type={record.memory_type} key={record.key} tier={record.tier} status={record.status} confidence={record.confidence:.2f} summary={record.summary}"
                for record in records[:12]
            ]
            parts.append("[Memory Records]\n" + "\n".join(lines))
        tape = self.tape_store.get_or_create(chat_key)
        anchor = tape.last_anchor()
        if anchor is not None:
            summary = str((anchor.payload.get("state") or {}).get("summary", "") or "")
            if summary:
                parts.append(f"[Tape Summary]\n{summary}")
        return "\n".join(parts)

    async def process_task(
        self,
        user_input: str,
        chat_key: str = "",
        heartbeat: Heartbeat | None = None,
        media_paths: list[str] | None = None,
        stream_callback: StreamTextCallback | None = None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None = None,
        send_intermediate_message: Callable[[str], Awaitable[None]] | None = None,
    ) -> TaskResponse:
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    logger.info("Initializing resource manager...")
                    await self.resource_manager.initialize_async()
                    self._initialized = True
                    logger.info("Resource manager initialized")
        if heartbeat is not None:
            heartbeat.beat()

        if chat_key and getattr(self, "_interactive_sessions", None) is not None:
            control = self._parse_interactive_session_command(user_input)
            if control is not None:
                return await self._handle_interactive_session_command(
                    chat_key, control
                )
            if self._interactive_sessions.has_active_session(chat_key):
                return await self._handle_interactive_session_message(
                    chat_key, user_input
                )

        # --- Tape context ---
        tape: Tape | None = None
        if chat_key:
            logger.info("Loading tape for chat_key=%s", chat_key)
            tape = self.tape_store.get_or_create(chat_key)
            logger.info("Tape loaded, observing user message...")
            if hasattr(self, "memory_store"):
                self.memory_store.observe_user_message(chat_key, user_input)
            pending_entries = []
            # Ensure bootstrap anchor exists
            if tape.last_anchor() is None:
                anchor = tape.append("anchor", {"name": "session/start", "state": {}})
                pending_entries.append(anchor)
            # Append user message
            content_for_tape = user_input
            if media_paths:
                content_for_tape = f"{user_input}\n[附带 {len(media_paths)} 张图片]"
            user_entry = tape.append(
                "message", {"role": "user", "content": content_for_tape}
            )
            pending_entries.append(user_entry)
            self.tape_store.save_entries(chat_key, pending_entries)
            logger.info("Tape entries saved, proceeding to _answer_with_dag")

        wrapped_runtime_event_callback = runtime_event_callback
        if tape is not None and chat_key:

            async def _record_runtime_event(event: Any) -> None:
                payload: dict[str, Any]
                if isinstance(event, dict):
                    payload = dict(event)
                else:
                    payload = {
                        "event": getattr(event, "event", ""),
                        "task_id": getattr(event, "task_id", ""),
                        "flow_id": getattr(event, "flow_id", ""),
                        "payload": dict(getattr(event, "payload", {}) or {}),
                    }
                entry = tape.append(
                    "event",
                    {
                        "event": str(payload.get("event", "") or ""),
                        "payload": dict(payload.get("payload") or {}),
                    },
                    {
                        "task_id": str(payload.get("task_id", "") or ""),
                        "flow_id": str(payload.get("flow_id", "") or ""),
                    },
                )
                self.tape_store.save_entry(chat_key, entry)
                if hasattr(self, "memory_store"):
                    self.memory_store.observe_runtime_event(chat_key, payload)
                if runtime_event_callback is not None:
                    maybe = runtime_event_callback(event)
                    if inspect.isawaitable(maybe):
                        await maybe

            wrapped_runtime_event_callback = _record_runtime_event

        try:
            logger.info("Starting _answer_with_dag")
            text, collected_media = await self._answer_with_dag(
                user_input,
                tape=tape,
                heartbeat=heartbeat,
                media_paths=media_paths,
                stream_callback=stream_callback,
                runtime_event_callback=wrapped_runtime_event_callback,
                send_intermediate_message=send_intermediate_message,
            )
            if heartbeat is not None:
                heartbeat.beat()

            # Append assistant response
            if tape and chat_key:
                asst_entry = tape.append(
                    "message", {"role": "assistant", "content": text}
                )
                self.tape_store.save_entry(chat_key, asst_entry)
                # Fire-and-forget async handoff check
                self._spawn_background_task(
                    self._maybe_handoff(tape, chat_key),
                    label=f"handoff:{chat_key}",
                )

            return TaskResponse(text=text, media_paths=collected_media)
        except Exception as exc:
            logger.exception("Error processing task")
            return TaskResponse(text=f"处理任务时出错：{exc}")

    async def _maybe_handoff(self, tape: Tape, chat_key: str) -> None:
        """Check if entries since last anchor exceed threshold; if so, create a new anchor."""
        lock = self._get_handoff_lock(chat_key)
        try:
            async with lock:
                threshold = self.config.system.context_compact_threshold

                # Collect entries once, compute tokens from them
                old_entries = tape.entries_since_anchor()
                if not old_entries:
                    return
                total_tokens = sum(e.token_estimate for e in old_entries)
                if total_tokens <= threshold:
                    return

                # Build text to summarize
                lines: list[str] = []
                for e in old_entries:
                    if e.kind == "message":
                        role = e.payload.get("role", "?")
                        content = e.payload.get("content", "")
                        lines.append(f"{role}: {content}")
                if not lines:
                    return

                history_text = "\n".join(lines)
                raw_summary = await self.gateway.complete(
                    _SUMMARIZE_PROMPT, history_text
                )

                # Parse structured JSON from LLM, fallback to plain summary
                structured: dict[str, Any] = {}
                try:
                    # Strip markdown code fences if present
                    text = raw_summary.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    structured = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    structured = {"summary": raw_summary.strip()}

                summary_text = structured.get("summary", raw_summary.strip())
                entities = structured.get("entities", [])
                next_steps = structured.get("next_steps", [])
                artifacts = structured.get("artifacts", [])
                open_questions = structured.get("open_questions", [])
                decisions = structured.get("decisions", [])
                if not isinstance(entities, list):
                    entities = []
                if not isinstance(next_steps, list):
                    next_steps = []
                if not isinstance(artifacts, list):
                    artifacts = []
                if not isinstance(open_questions, list):
                    open_questions = []
                if not isinstance(decisions, list):
                    decisions = []

                source_ids = [e.entry_id for e in old_entries]

                # Detect topic shift: compare current segment keywords vs previous anchor summary
                phase = "continuation"
                prev_anchor = tape.last_anchor()
                if prev_anchor:
                    prev_summary = (prev_anchor.payload.get("state") or {}).get(
                        "summary", ""
                    )
                    prev_kws = set(_extract_keywords(prev_summary, max_keywords=12))
                    # Collect recent user messages for keyword comparison
                    recent_user_text = " ".join(
                        e.payload.get("content", "")
                        for e in old_entries
                        if e.kind == "message" and e.payload.get("role") == "user"
                    )
                    curr_kws = set(_extract_keywords(recent_user_text, max_keywords=12))
                    if prev_kws and curr_kws:
                        overlap = len(prev_kws & curr_kws) / max(
                            len(prev_kws), len(curr_kws)
                        )
                        if overlap < 0.15:
                            phase = "topic_shift"
                            logger.info(
                                "Topic shift detected chat_key=%s overlap=%.2f",
                                chat_key,
                                overlap,
                            )

                anchor = tape.append(
                    "anchor",
                    {
                        "name": f"compact/{tape.turn_count()}",
                        "state": {
                            "summary": summary_text,
                            "entities": entities,
                            "user_intent": structured.get("user_intent", ""),
                            "pending": structured.get("pending", ""),
                            "next_steps": [str(item) for item in next_steps[:3]],
                            "artifacts": [str(item) for item in artifacts[:5]],
                            "open_questions": [
                                str(item) for item in open_questions[:3]
                            ],
                            "decisions": [str(item) for item in decisions[:3]],
                            "phase": phase,
                            "source_ids": source_ids,
                            "turn_count": tape.turn_count(),
                        },
                    },
                )
                self.tape_store.save_entry(chat_key, anchor)
                if hasattr(self, "memory_store"):
                    self.memory_store.observe_anchor_state(
                        chat_key,
                        anchor.payload.get("state") or {},
                        source_ids=source_ids,
                    )
                tape.compact_entries()
                logger.info(
                    "Handoff created anchor chat_key=%s entry_id=%d summarized=%d entries",
                    chat_key,
                    anchor.entry_id,
                    len(source_ids),
                )
        except Exception:
            logger.exception("Error in _maybe_handoff for chat_key=%s", chat_key)

    @staticmethod
    def _parse_interactive_session_command(
        user_input: str,
    ) -> dict[str, str] | None:
        text = (user_input or "").strip()
        if not text.lower().startswith("@session"):
            return None
        parts = text.split()
        action = parts[1].lower() if len(parts) >= 2 else "status"
        backend_name = parts[2].lower() if len(parts) >= 3 else ""
        return {"action": action, "backend_name": backend_name}

    async def _handle_interactive_session_command(
        self, chat_key: str, control: dict[str, str]
    ) -> TaskResponse:
        manager = self._interactive_sessions
        action = control.get("action", "")
        backend_name = control.get("backend_name", "")

        if action == "start":
            if not backend_name:
                return TaskResponse(text="用法：@session start <backend>")
            session = await manager.start(chat_key=chat_key, backend_name=backend_name)
            label = session.backend_name.capitalize()
            return TaskResponse(
                text=(
                    f"{label} 会话已启动（session_id={session.session_id}）。"
                    "后续消息将直接发送到该交互会话。"
                )
            )
        if action == "stop":
            stopped = await manager.stop(chat_key, reason="user_stop")
            return TaskResponse(
                text="交互会话已关闭。" if stopped else "当前没有活动中的交互会话。"
            )
        if action == "status":
            status = manager.status(chat_key)
            if status is None:
                return TaskResponse(text="当前没有活动中的交互会话。")
            return TaskResponse(
                text=(
                    f"当前交互会话：{status.backend_name} "
                    f"(session_id={status.session_id})"
                )
            )
        return TaskResponse(text="支持的命令：@session start <backend> / status / stop")

    async def _handle_interactive_session_message(
        self, chat_key: str, user_input: str
    ) -> TaskResponse:
        reply = await self._interactive_sessions.send(chat_key, user_input)
        return TaskResponse(
            text=reply.text,
            media_paths=list(reply.media_paths or []),
        )

    def reset(self) -> None:
        manager = getattr(self, "_interactive_sessions", None)
        if manager is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(manager.stop_all(reason="reset"))
            else:
                self._spawn_background_task(
                    manager.stop_all(reason="reset"),
                    label="interactive-reset",
                )
        if self.resource_manager is not None:
            self.resource_manager.reset()
        if self.tape_store is not None:
            self.tape_store.clear()
        self._initialized = False

    def get_status(self) -> dict[str, Any]:
        resource_manager = getattr(self, "resource_manager", None)
        interactive_manager = getattr(self, "_interactive_sessions", None)
        status = {
            "resource_manager": "initialized",
            "available_tools": len(resource_manager.get_available_tools())
            if resource_manager is not None
            else 0,
            "resources": resource_manager.search_resources()
            if resource_manager is not None
            else [],
        }
        if interactive_manager is not None:
            status["interactive_sessions"] = interactive_manager.summary()
        return status
