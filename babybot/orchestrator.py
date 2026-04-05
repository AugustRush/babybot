"""Orchestrator built on lightweight kernel — DAG-driven multi-agent mode."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import contextlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .agent_kernel import ExecutionContext, RuntimeState
from .agent_kernel.plan_notebook import PlanNotebook
from .agent_kernel.plan_notebook_store import PlanNotebookStore
from .agent_kernel.dynamic_orchestrator import (
    DynamicOrchestrator,
    InMemoryChildTaskBus,
)
from .orchestrator_prompts import build_orchestrator_config
from .agent_kernel.execution_constraints import (
    build_execution_constraint_hints,
    infer_execution_constraints,
)
from .config import Config
from .context import Tape, TapeStore, _extract_keywords
from .context_views import build_context_view
from .execution_plan import build_execution_plan, compile_execution_plan_to_notebook
from .memory_store import HybridMemoryStore
from .heartbeat import TaskHeartbeatRegistry
from .interactive_sessions import InteractiveSessionManager
from .interactive_sessions.backends import ClaudeInteractiveBackend
from .interactive_sessions.types import (
    InteractiveOutputCallback,
    InteractiveRequest,
)
from .model_gateway import OpenAICompatibleGateway
from .orchestration_policy import ConservativePolicySelector, build_policy_state_bucket
from .orchestration_policy_store import OrchestrationPolicyStore
from .orchestration_router import (
    RoutingDecision,
    build_routing_intent_bucket,
    build_routing_snapshot,
    match_rule_based_routing,
    route_mode_to_contract_mode,
    route_mode_to_step_kind,
    route_task,
    should_call_model_router,
)
from .orchestration_policy_types import PolicyDecisionRecord, PolicyOutcomeRecord
from .resource import ResourceManager
from .runtime_job_store import RuntimeJobStore
from .runtime_jobs import JOB_STATES, project_job_state_from_runtime_event
from .runtime_feedback_commands import parse_policy_command
from .task_contract import build_task_contract
from .task_evaluator import TaskEvaluationInput, TaskEvaluator

if TYPE_CHECKING:
    from .heartbeat import Heartbeat

logger = logging.getLogger(__name__)
StreamTextCallback = Callable[[str], Awaitable[None] | None]

_SUMMARIZE_PROMPT = (
    "CRITICAL: 仅输出纯文本 JSON，禁止调用任何工具，禁止任何 markdown 代码块包裹。\n\n"
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
    is_error: bool = False


class _NullPolicyStore:
    """Fallback store used by tests and partial agent doubles."""

    @staticmethod
    def summarize_action_stats(
        *,
        decision_kind: str,
        state_bucket: str | None = None,
    ) -> dict[str, dict[str, float | int]]:
        del decision_kind, state_bucket
        return {}


_NULL_POLICY_STORE = _NullPolicyStore()


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
        self._runtime_job_store = RuntimeJobStore(
            self.config.home_dir / "memory" / "jobs.db"
        )
        self._plan_notebook_store = PlanNotebookStore(
            self.config.home_dir / "memory" / "plan_notebooks.db"
        )
        self._child_task_bus = InMemoryChildTaskBus()
        self._task_heartbeat_registry = TaskHeartbeatRegistry()
        self._handoff_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._recent_flow_ids_by_chat: OrderedDict[str, str] = OrderedDict()
        self._recent_flows_by_chat: OrderedDict[str, list[str]] = OrderedDict()
        self._recent_policy_decisions_by_flow: OrderedDict[
            str, list[dict[str, Any]]
        ] = OrderedDict()
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

    @staticmethod
    def _ensure_ordered_mapping(mapping: Any) -> OrderedDict[str, Any]:
        if isinstance(mapping, OrderedDict):
            return mapping
        normalized: OrderedDict[str, Any] = OrderedDict()
        if isinstance(mapping, dict):
            normalized.update(mapping)
        return normalized

    @staticmethod
    def _callable_accepts_kwarg(callable_obj: Any, name: str) -> bool:
        try:
            parameters = inspect.signature(callable_obj).parameters.values()
        except (TypeError, ValueError):
            return False
        for parameter in parameters:
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
            if parameter.name == name:
                return True
        return False

    def _remember_flow_id(self, chat_key: str, flow_id: str) -> None:
        recent_flow_ids_by_chat = self._ensure_ordered_mapping(
            getattr(self, "_recent_flow_ids_by_chat", OrderedDict())
        )
        self._recent_flow_ids_by_chat = recent_flow_ids_by_chat
        recent_flows_by_chat = self._ensure_ordered_mapping(
            getattr(self, "_recent_flows_by_chat", OrderedDict())
        )
        self._recent_flows_by_chat = recent_flows_by_chat
        recent_flow_ids_by_chat.pop(chat_key, None)
        recent_flow_ids_by_chat[chat_key] = flow_id
        while len(recent_flow_ids_by_chat) > self._FLOW_CACHE_LIMIT:
            recent_flow_ids_by_chat.popitem(last=False)
        history = [
            item
            for item in recent_flows_by_chat.get(chat_key, [])
            if item != flow_id
        ]
        history.insert(0, flow_id)
        recent_flows_by_chat[chat_key] = history[:5]
        while len(recent_flows_by_chat) > self._FLOW_CACHE_LIMIT:
            recent_flows_by_chat.popitem(last=False)

    @staticmethod
    def _build_notebook_completion_summary(
        notebook: PlanNotebook,
        *,
        final_text: str,
    ) -> dict[str, Any]:
        existing = dict(notebook.completion_summary or {})
        final_summary = str(existing.get("final_summary", "") or final_text or "").strip()
        decision_register = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in (
                        existing.get("decision_register")
                        or [
                            decision.summary
                            for node in notebook.nodes.values()
                            for decision in node.decisions[-3:]
                        ]
                    )
                    if str(item).strip()
                ]
            )
        )
        artifact_manifest = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in (
                        existing.get("artifact_manifest")
                        or [
                            artifact.path
                            for node in notebook.nodes.values()
                            for artifact in node.artifacts
                        ]
                    )
                    if str(item).strip()
                ]
            )
        )
        node_summaries = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in (
                        existing.get("node_summaries")
                        or [
                            f"{node.title}: {node.latest_summary or node.result_text or node.objective}"
                            for node in notebook.nodes.values()
                            if node.node_id != notebook.root_node_id
                            and str(node.latest_summary or node.result_text or node.objective).strip()
                        ]
                    )
                    if str(item).strip()
                ]
            )
        )
        open_followups = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in (
                        existing.get("open_followups")
                        or [
                            checkpoint.message
                            for node in notebook.nodes.values()
                            for checkpoint in node.checkpoints
                            if checkpoint.status == "open"
                        ]
                        + [
                            issue.title
                            for node in notebook.nodes.values()
                            for issue in node.issues
                            if issue.status == "open"
                        ]
                    )
                    if str(item).strip()
                ]
            )
        )
        search_terms = list(
            dict.fromkeys(
                [
                    term
                    for term in (
                        list(existing.get("search_terms") or [])
                        + _extract_keywords(notebook.goal)
                        + _extract_keywords(final_summary)
                        + _extract_keywords(" ".join(decision_register[:6]))
                        + _extract_keywords(" ".join(node_summaries[:6]))
                    )
                    if str(term).strip()
                ]
            )
        )
        return {
            "final_summary": final_summary,
            "decision_register": decision_register,
            "artifact_manifest": artifact_manifest,
            "open_followups": open_followups,
            "node_summaries": node_summaries,
            "search_terms": search_terms,
            "updated_at": time.time(),
        }

    def _persist_notebook_completion(
        self,
        *,
        chat_key: str,
        context: ExecutionContext,
        final_text: str,
    ) -> None:
        notebook = RuntimeState(context).notebook_binding().notebook
        if notebook is None:
            return
        completion_summary = self._build_notebook_completion_summary(
            notebook,
            final_text=final_text,
        )
        notebook.set_completion_summary(completion_summary)
        notebook_store = getattr(self, "_plan_notebook_store", None)
        if notebook_store is not None:
            notebook_store.save_notebook(notebook, chat_key=chat_key)
        memory_store = getattr(self, "memory_store", None)
        if memory_store is not None and chat_key:
            memory_store.observe_notebook_completion(
                chat_id=chat_key,
                notebook_id=notebook.notebook_id,
                completion_summary=completion_summary,
            )

    def _get_handoff_lock(self, chat_key: str) -> asyncio.Lock:
        handoff_locks = self._ensure_ordered_mapping(
            getattr(self, "_handoff_locks", OrderedDict())
        )
        self._handoff_locks = handoff_locks
        lock = handoff_locks.pop(chat_key, None)
        if lock is None:
            lock = asyncio.Lock()
        handoff_locks[chat_key] = lock
        while len(handoff_locks) > self._HANDOFF_LOCK_LIMIT:
            handoff_locks.popitem(last=False)
        return lock

    def _remember_flow_policy_decisions(
        self,
        flow_id: str,
        events: list[dict[str, Any]],
    ) -> None:
        if not flow_id:
            return
        policy_decisions_by_flow = self._ensure_ordered_mapping(
            getattr(self, "_recent_policy_decisions_by_flow", OrderedDict())
        )
        self._recent_policy_decisions_by_flow = policy_decisions_by_flow
        decisions: list[dict[str, Any]] = []
        for event in events:
            if event.get("event") != "policy_decision":
                continue
            decisions.append(
                {
                    "decision_kind": str(event.get("decision_kind", "") or "").strip(),
                    "action_name": str(event.get("action_name", "") or "").strip(),
                    "state_bucket": str(event.get("state_bucket", "") or "").strip(),
                    "explain": str(event.get("explain", "") or "").strip(),
                }
            )
        policy_decisions_by_flow.pop(flow_id, None)
        policy_decisions_by_flow[flow_id] = decisions
        while len(policy_decisions_by_flow) > self._FLOW_CACHE_LIMIT:
            policy_decisions_by_flow.popitem(last=False)

    def _spawn_background_task(
        self,
        coro: Awaitable[Any],
        *,
        label: str,
    ) -> asyncio.Task[Any]:
        if not hasattr(self, "_background_tasks") or self._background_tasks is None:
            self._background_tasks = set()
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Background task failed: %s", label)

        task.add_done_callback(_on_done)
        return task

    def _policy_learning_enabled(self) -> bool:
        return bool(getattr(self.config.system, "policy_learning_enabled", False))

    def _routing_enabled(self) -> bool:
        return bool(getattr(self.config.system, "routing_enabled", True))

    def _reflection_enabled(self) -> bool:
        return bool(getattr(self.config.system, "reflection_enabled", False))

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
        policy_store = getattr(self, "_policy_store", None)
        if policy_store is None:
            return
        policy_store.record_decision(
            flow_id=record.flow_id,
            chat_key=record.chat_key,
            decision_kind=record.decision_kind,
            action_name=record.action_name,
            state_features=record.state_features,
        )

    def _record_policy_outcome(self, record: PolicyOutcomeRecord) -> None:
        if not self._policy_learning_enabled() or not record.chat_key:
            return
        policy_store = getattr(self, "_policy_store", None)
        if policy_store is None:
            return
        policy_store.record_outcome(
            flow_id=record.flow_id,
            chat_key=record.chat_key,
            final_status=record.final_status,
            reward=record.reward,
            outcome=record.outcome,
        )

    def _select_decomposition_action(
        self,
        *,
        user_input: str,
        media_paths: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        features = self._build_policy_state_features(
            user_input,
            media_paths=media_paths,
        )
        selector = self._policy_selector()
        action = selector.choose_decomposition(features=features)
        return action.name, features, action.hint

    @staticmethod
    def _estimate_independent_subtasks(user_input: str) -> int:
        text = str(user_input or "").strip()
        count = 1
        for token in ("同时", "分别", "并行", "并且"):
            count += text.count(token)
        return max(1, count)

    def _build_scheduling_state_features(
        self,
        user_input: str,
        *,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        features = self._build_policy_state_features(
            user_input,
            media_paths=media_paths,
        )
        features["independent_subtasks"] = self._estimate_independent_subtasks(
            user_input
        )
        return features

    def _policy_selector(self) -> ConservativePolicySelector:
        policy_store = getattr(self, "_policy_store", None)
        if policy_store is None or not hasattr(policy_store, "summarize_action_stats"):
            policy_store = _NULL_POLICY_STORE
        return ConservativePolicySelector(
            policy_store,
            min_samples=int(
                getattr(self.config.system, "policy_learning_min_samples", 0) or 0
            ),
            explore_ratio=float(
                getattr(self.config.system, "policy_learning_explore_ratio", -1.0)
                or -1.0
            ),
        )

    def choose_scheduling_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        selection = self._policy_selector().select_scheduling(features=features)
        return {
            "action_name": selection.action.name,
            "hint": selection.action.hint,
            "explain": selection.explain,
            "state_bucket": selection.state_bucket,
        }

    def choose_worker_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        if not self._policy_learning_enabled():
            return {
                "action_name": "allow_worker",
                "hint": "",
                "explain": "policy_learning_disabled",
                "state_bucket": "disabled",
            }
        selection = self._policy_selector().select_worker_gate(features=features)
        return {
            "action_name": selection.action.name,
            "hint": selection.action.hint,
            "explain": selection.explain,
            "state_bucket": selection.state_bucket,
        }

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
        self._remember_flow_policy_decisions(flow_id, events)

    @staticmethod
    def _routing_policy_hints(decision: RoutingDecision | None) -> list[str]:
        if decision is None:
            return []
        hints: list[str] = []
        explain = str(decision.explain or "").strip()
        if explain:
            hints.append(f"路由判定：{explain}")
        execution_hints = {
            "direct_execute": "路由建议：信息充分时可直接执行，但仍需避免猜测。",
            "analyze_first": "路由建议：先分析边界与依赖，再开始执行。",
            "retrieve_first": "路由建议：先补足上下文或外部信息，再执行。",
            "verify_first": "路由建议：在收尾前优先做局部验证。",
        }
        parallel_hints = {
            "serial": "路由建议：默认串行推进，优先保证收敛。",
            "bounded_parallel": "路由建议：仅对明确独立子任务做有限并行。",
        }
        worker_hints = {
            "allow": "路由建议：必要时可启用少量 worker。",
            "deny": "路由建议：优先在当前主链路完成，不轻易扩散 worker。",
        }
        for mapping, key in (
            (execution_hints, decision.execution_style),
            (parallel_hints, decision.parallelism_hint),
            (worker_hints, decision.worker_hint),
        ):
            hint = mapping.get(str(key or "").strip())
            if hint and hint not in hints:
                hints.append(hint)
        return hints

    @staticmethod
    def _debug_policy_line(
        label: str,
        value: str,
        *,
        explain: str = "",
    ) -> str:
        normalized_label = str(label or "").strip() or "unknown"
        normalized_value = str(value or "").strip() or "-"
        normalized_explain = str(explain or "").strip()
        if not normalized_explain:
            return f"{normalized_label}={normalized_value}"
        return f"{normalized_label}={normalized_value}; explain={normalized_explain}"

    def _build_debug_policy_summary(
        self,
        *,
        flow_id: str,
        decomposition_action: str,
        decomposition_hint: str,
        route_mode: str,
        router_skip_reason: str,
        routing_decision: RoutingDecision | None,
        scheduling_policy: dict[str, Any],
        worker_policy: dict[str, Any],
    ) -> str:
        routing_value = (
            "/".join(
                part
                for part in (
                    str(routing_decision.decision_source or "").strip(),
                    str(routing_decision.route_mode or "").strip(),
                    str(routing_decision.execution_style or "").strip(),
                )
                if part
            )
            if routing_decision is not None
            else f"fallback/{str(route_mode or '').strip() or 'tool_workflow'}"
        )
        routing_explain = (
            str(routing_decision.explain or "").strip()
            if routing_decision is not None
            else f"router_skip_reason={str(router_skip_reason or 'fallback').strip()}"
        )
        lines = [
            "调试：编排决策",
            f"flow_id={str(flow_id or '').strip() or '-'}",
            self._debug_policy_line(
                "decomposition",
                str(decomposition_action or "").strip(),
                explain=decomposition_hint,
            ),
            self._debug_policy_line(
                "routing",
                routing_value,
                explain=routing_explain,
            ),
            self._debug_policy_line(
                "scheduling",
                str(scheduling_policy.get("action_name", "") or "").strip(),
                explain=str(scheduling_policy.get("explain", "") or ""),
            ),
            self._debug_policy_line(
                "worker",
                str(worker_policy.get("action_name", "") or "").strip(),
                explain=str(worker_policy.get("explain", "") or ""),
            ),
        ]
        return "\n".join(line for line in lines if line.strip())

    async def _maybe_send_debug_policy_summary(
        self,
        *,
        send_intermediate_message: Callable[[str], Awaitable[None]] | None,
        flow_id: str,
        decomposition_action: str,
        decomposition_hint: str,
        route_mode: str,
        router_skip_reason: str,
        routing_decision: RoutingDecision | None,
        scheduling_policy: dict[str, Any],
        worker_policy: dict[str, Any],
    ) -> None:
        if send_intermediate_message is None:
            return
        if not bool(getattr(self.config.system, "debug_runtime_feedback", False)):
            return
        text = self._build_debug_policy_summary(
            flow_id=flow_id,
            decomposition_action=decomposition_action,
            decomposition_hint=decomposition_hint,
            route_mode=route_mode,
            router_skip_reason=router_skip_reason,
            routing_decision=routing_decision,
            scheduling_policy=scheduling_policy,
            worker_policy=worker_policy,
        )
        if not text.strip():
            return
        await send_intermediate_message(text)

    @staticmethod
    def _format_reflection_hint(payload: dict[str, Any]) -> str:
        return (
            "历史反思："
            f"曾出现 {str(payload.get('failure_pattern', '') or 'unknown')}，"
            f"下次优先考虑 {str(payload.get('recommended_action', '') or 'safe_action')} "
            f"(置信度 {float(payload.get('confidence', 0.0) or 0.0):.2f})。"
        )

    @staticmethod
    def _select_reflection_override(
        hints: list[dict[str, Any]],
        *,
        allowed_actions: set[str],
    ) -> str:
        for payload in hints:
            action = str(payload.get("recommended_action", "") or "").strip()
            if action in allowed_actions:
                return action
        return ""

    def _maybe_override_policy_from_reflection(
        self,
        payload: dict[str, Any],
        *,
        preferred_action: str,
        hint_prefix: str,
    ) -> dict[str, Any]:
        if not preferred_action:
            return payload
        explain = str(payload.get("explain", "") or "")
        if (
            "insufficient_samples" not in explain
            and payload.get("state_bucket") != "global_default"
        ):
            return payload
        return {
            "action_name": preferred_action,
            "hint": f"{hint_prefix}{preferred_action}",
            "explain": f"reflection_override; preferred_action={preferred_action}; base={explain}",
            "state_bucket": str(payload.get("state_bucket", "") or "reflection"),
        }

    @staticmethod
    def _maybe_soften_policy_from_guardrail(
        payload: dict[str, Any],
        *,
        soften_default: bool,
        current_default_action: str,
        softened_action: str,
        hint: str,
    ) -> dict[str, Any]:
        if not soften_default:
            return payload
        action_name = str(payload.get("action_name", "") or "").strip()
        explain = str(payload.get("explain", "") or "")
        if action_name != current_default_action:
            return payload
        if (
            "insufficient_samples" not in explain
            and payload.get("state_bucket") != "global_default"
        ):
            return payload
        return {
            "action_name": softened_action,
            "hint": hint,
            "explain": f"reflection_guardrail_softened; base={explain}",
            "state_bucket": str(payload.get("state_bucket", "") or "guardrail"),
        }

    @staticmethod
    def _routing_decision_from_reflection(
        payload: dict[str, Any] | None,
    ) -> RoutingDecision | None:
        if not isinstance(payload, dict):
            return None
        route_mode = str(payload.get("route_mode", "") or "").strip()
        recommended_action = str(payload.get("recommended_action", "") or "").strip()
        if route_mode not in {"tool_workflow", "answer", "debate"}:
            return None
        execution_style = (
            recommended_action
            if recommended_action
            in {
                "direct_execute",
                "analyze_first",
                "retrieve_first",
                "verify_first",
            }
            else "analyze_first"
        )
        parallelism_hint = (
            recommended_action
            if recommended_action in {"serial", "bounded_parallel"}
            else "serial"
        )
        worker_hint = (
            "allow" if recommended_action in {"allow", "allow_worker"} else "deny"
        )
        samples = int(payload.get("samples", 0) or 0)
        effective_samples = float(payload.get("effective_samples", 0.0) or 0.0)
        min_samples_required = int(payload.get("min_samples_required", 0) or 0)
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        return RoutingDecision(
            route_mode=route_mode,
            need_clarification=False,
            execution_style=execution_style,
            parallelism_hint=parallelism_hint,
            worker_hint=worker_hint,
            explain=(
                "稳定成功经验直达"
                + f"(samples={samples}, effective_samples={effective_samples:.2f}, "
                + f"min_samples={min_samples_required}, confidence={confidence:.2f})"
            ),
            decision_source="reflection",
        )

    def _routing_decision_from_intent_cache(
        self,
        payload: dict[str, Any] | None,
        *,
        goal: str,
    ) -> RoutingDecision | None:
        if not isinstance(payload, dict):
            return None
        route_mode = str(payload.get("route_mode", "") or "").strip()
        execution_style = str(payload.get("execution_style", "") or "").strip()
        if route_mode not in {"tool_workflow", "answer", "debate"}:
            return None
        if execution_style not in {
            "direct_execute",
            "analyze_first",
            "retrieve_first",
            "verify_first",
        }:
            execution_style = "analyze_first"
        subtasks = int(
            self._build_scheduling_state_features(goal).get("independent_subtasks", 1)
            or 1
        )
        return RoutingDecision(
            route_mode=route_mode,
            need_clarification=False,
            execution_style=execution_style,
            parallelism_hint="bounded_parallel" if subtasks >= 2 else "serial",
            worker_hint="deny",
            explain=(
                "稳定意图桶直达"
                + f"(samples={int(payload.get('samples', 0) or 0)}, "
                + f"wins={int(payload.get('wins', 0) or 0)})"
            ),
            decision_source="intent_cache",
        )

    def _record_routing_telemetry(
        self,
        *,
        chat_key: str,
        flow_id: str,
        route_mode: str,
        resolved_router_model: str,
        routing_latency_ms: float,
        routing_decision: RoutingDecision | None,
        router_skip_reason: str,
        should_use_model_router: bool,
        intent_bucket: str,
        reflection_hints_payload: list[dict[str, Any]],
        scheduling_overridden: bool,
        worker_overridden: bool,
        execution_style_guardrail_reduced: bool,
        relaxed_reflection_route_payload: Any,
        guardrail_softened_scheduling: bool,
        guardrail_softened_worker: bool,
    ) -> None:
        policy_store = getattr(self, "_policy_store", None)
        if not chat_key or not hasattr(policy_store, "record_runtime_telemetry"):
            return
        policy_store.record_runtime_telemetry(
            flow_id=flow_id,
            chat_key=chat_key,
            route_mode=route_mode,
            router_model=resolved_router_model,
            router_latency_ms=routing_latency_ms,
            router_fallback=routing_decision is None,
            router_source=(
                routing_decision.decision_source
                if routing_decision is not None
                else (
                    f"skipped:{router_skip_reason}"
                    if self._routing_enabled() and not should_use_model_router
                    else "fallback"
                )
            ),
            execution_style=(
                str(routing_decision.execution_style or "")
                if routing_decision is not None
                else ""
            ),
            intent_bucket=intent_bucket,
            reflection_hint_count=len(reflection_hints_payload),
            reflection_override_count=int(scheduling_overridden)
            + int(worker_overridden),
            execution_style_reflection_count=int(
                routing_decision is not None
                and routing_decision.decision_source == "reflection"
            ),
            parallelism_reflection_count=int(scheduling_overridden),
            worker_reflection_count=int(worker_overridden),
            execution_style_guardrail_reduce_count=int(
                execution_style_guardrail_reduced
                and relaxed_reflection_route_payload is not None
                and routing_decision is None
            ),
            parallelism_guardrail_soften_count=int(guardrail_softened_scheduling),
            worker_guardrail_soften_count=int(guardrail_softened_worker),
        )

    async def _evaluate_task_run_async(
        self,
        *,
        chat_key: str,
        route_mode: str,
        state_features: dict[str, Any],
        routing_decision: RoutingDecision | None,
        final_status: str,
        outcome: dict[str, Any],
    ) -> None:
        if not self._reflection_enabled():
            return
        policy_store = getattr(self, "_policy_store", None)
        if not hasattr(policy_store, "record_reflection"):
            return
        TaskEvaluator(policy_store).evaluate(
            TaskEvaluationInput(
                chat_key=chat_key,
                route_mode=route_mode,
                state_features=state_features,
                execution_style=(
                    routing_decision.execution_style
                    if routing_decision is not None
                    else ""
                ),
                parallelism_hint=(
                    routing_decision.parallelism_hint
                    if routing_decision is not None
                    else ""
                ),
                worker_hint=(
                    routing_decision.worker_hint if routing_decision is not None else ""
                ),
                final_status=final_status,
                outcome=outcome,
            )
        )

    @staticmethod
    def _task_result_outcome_details(result: Any | None) -> dict[str, int]:
        task_results = dict(getattr(result, "task_results", {}) or {})
        retry_count = 0
        dead_letter_count = 0
        stalled_count = 0
        tool_call_count = 0
        tool_failure_count = 0
        loop_guard_block_count = 0
        max_step_exhausted_count = 0
        for task_result in task_results.values():
            attempts = max(1, int(getattr(task_result, "attempts", 1) or 1))
            retry_count += max(0, attempts - 1)
            metadata = dict(getattr(task_result, "metadata", {}) or {})
            if metadata.get("dead_lettered") is True:
                dead_letter_count += 1
            error_text = str(getattr(task_result, "error", "") or "").strip().lower()
            if (
                metadata.get("stalled") is True
                or metadata.get("error_type") == "stalled"
                or "heartbeat stalled" in error_text
            ):
                stalled_count += 1
            tool_call_count += max(0, int(metadata.get("tool_call_count", 0) or 0))
            tool_failure_count += max(
                0, int(metadata.get("tool_failure_count", 0) or 0)
            )
            loop_guard_block_count += max(
                0, int(metadata.get("loop_guard_block_count", 0) or 0)
            )
            max_step_exhausted_count += max(
                0, int(metadata.get("max_step_exhausted_count", 0) or 0)
            )
        return {
            "task_result_count": len(task_results),
            "retry_count": retry_count,
            "dead_letter_count": dead_letter_count,
            "stalled_count": stalled_count,
            "tool_call_count": tool_call_count,
            "tool_failure_count": tool_failure_count,
            "loop_guard_block_count": loop_guard_block_count,
            "max_step_exhausted_count": max_step_exhausted_count,
        }

    @classmethod
    def _policy_reward(
        cls,
        events: list[dict[str, Any]],
        final_status: str,
        *,
        result: Any | None = None,
    ) -> float:
        reward = 1.0 if final_status == "succeeded" else -1.0
        event_retry_count = sum(
            1 for event in events if event.get("event") == "retrying"
        )
        event_dead_letter_count = sum(
            1 for event in events if event.get("event") == "dead_lettered"
        )
        event_stalled_count = sum(
            1 for event in events if event.get("event") == "stalled"
        )
        result_details = cls._task_result_outcome_details(result)
        retry_count = max(event_retry_count, int(result_details["retry_count"]))
        dead_letter_count = max(
            event_dead_letter_count, int(result_details["dead_letter_count"])
        )
        stalled_count = max(event_stalled_count, int(result_details["stalled_count"]))
        reward -= 0.15 * retry_count
        reward -= 0.25 * dead_letter_count
        reward -= 0.2 * stalled_count
        reward -= 0.08 * int(result_details["tool_failure_count"] > 0)
        reward -= 0.1 * int(result_details["loop_guard_block_count"] > 0)
        reward -= 0.18 * int(result_details["max_step_exhausted_count"] > 0)
        return max(-1.0, min(1.0, reward))

    @classmethod
    def _policy_outcome_details(
        cls,
        events: list[dict[str, Any]],
        *,
        result: Any | None = None,
        error: str | None = None,
        execution_elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        event_retry_count = sum(
            1 for event in events if event.get("event") == "retrying"
        )
        event_dead_letter_count = sum(
            1 for event in events if event.get("event") == "dead_lettered"
        )
        event_stalled_count = sum(
            1 for event in events if event.get("event") == "stalled"
        )
        result_details = cls._task_result_outcome_details(result)
        payload = {
            "retry_count": max(event_retry_count, int(result_details["retry_count"])),
            "dead_letter_count": max(
                event_dead_letter_count, int(result_details["dead_letter_count"])
            ),
            "stalled_count": max(
                event_stalled_count, int(result_details["stalled_count"])
            ),
            "task_result_count": int(result_details["task_result_count"]),
            "executor_step_count": sum(
                1 for event in events if event.get("event") == "executor.step"
            ),
            "tool_call_count": int(result_details["tool_call_count"]),
            "tool_failure_count": int(result_details["tool_failure_count"]),
            "loop_guard_block_count": int(result_details["loop_guard_block_count"]),
            "max_step_exhausted_count": int(result_details["max_step_exhausted_count"]),
        }
        if execution_elapsed_ms is not None:
            payload["execution_elapsed_ms"] = round(
                max(0.0, float(execution_elapsed_ms)), 2
            )
        if error:
            payload["error"] = error
        return payload

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
        optional_kwargs = {
            "child_task_bus": getattr(self, "_child_task_bus", None),
            "task_heartbeat_registry": getattr(self, "_task_heartbeat_registry", None),
            "task_stale_after_s": float(self.config.system.idle_timeout),
            "max_steps": getattr(self.config.system, "orchestrator_max_steps", 30),
            "default_task_timeout_s": float(
                getattr(self.config.system, "subtask_timeout", 0) or 0
            )
            or None,
        }
        for key, value in optional_kwargs.items():
            if value is None or not self._callable_accepts_kwarg(
                DynamicOrchestrator, key
            ):
                continue
            orchestrator_kwargs[key] = value

        if self._callable_accepts_kwarg(DynamicOrchestrator, "config"):
            orchestrator_kwargs["config"] = build_orchestrator_config()
        orchestrator = DynamicOrchestrator(**orchestrator_kwargs)
        flow_id = f"orchestrator:{uuid.uuid4().hex[:12]}"
        chat_key = getattr(tape, "chat_id", "") if tape is not None else ""
        if chat_key:
            self._remember_flow_id(chat_key, flow_id)

        decomposition_action, decomposition_features, decomposition_hint = (
            self._select_decomposition_action(
                user_input=user_input,
                media_paths=media_paths,
            )
        )
        scheduling_features = self._build_scheduling_state_features(
            user_input,
            media_paths=media_paths,
        )
        execution_constraints = await infer_execution_constraints(
            self.gateway,
            user_input,
            heartbeat=heartbeat,
            default_max_total_seconds=(
                float(getattr(self.config.system, "timeout", 0) or 0) or None
            ),
        )
        runtime_job_store = getattr(self, "_runtime_job_store", None)
        runtime_job = (
            runtime_job_store.latest_for_chat(chat_key)
            if runtime_job_store is not None and chat_key
            else None
        )
        routing_snapshot = build_routing_snapshot(
            chat_key=chat_key,
            goal=user_input,
            tape=tape,
            memory_store=getattr(self, "memory_store", None) if tape else None,
            runtime_job=runtime_job,
            recent_flow_ids=list(
                (getattr(self, "_recent_flows_by_chat", {}) or {}).get(chat_key, [])
            ),
            execution_constraints=execution_constraints,
        )
        routing_decision = None
        configured_router_model = str(
            getattr(self.config.system, "routing_model_name", "") or ""
        ).strip()
        fallback_router_model = str(
            getattr(getattr(self.config, "model", None), "model_name", "") or ""
        ).strip()
        resolved_router_model = configured_router_model or fallback_router_model
        routing_timeout = float(getattr(self.config.system, "routing_timeout", 2.0) or 2.0)
        runtime_telemetry_store = getattr(self, "_policy_store", None)
        intent_bucket = build_routing_intent_bucket(
            user_input,
            has_media=bool(media_paths),
        )
        reflection_guardrails = {
            "samples": 0,
            "execution_style": {"injection_level": "normal", "soften_default": False},
            "parallelism": {"injection_level": "normal", "soften_default": False},
            "worker": {"injection_level": "normal", "soften_default": False},
        }
        if hasattr(runtime_telemetry_store, "recommend_reflection_guardrails"):
            reflection_guardrails = (
                runtime_telemetry_store.recommend_reflection_guardrails(
                    chat_key=chat_key or None
                )
            )
        if (
            self._reflection_enabled()
            and chat_key
            and hasattr(runtime_telemetry_store, "recommend_route_from_reflections")
        ):
            execution_style_guardrail_reduced = (
                str(
                    reflection_guardrails.get("execution_style", {}).get(
                        "injection_level", "normal"
                    )
                )
                == "reduced"
            )
            relaxed_reflection_route_payload = (
                runtime_telemetry_store.recommend_route_from_reflections(
                    chat_key=chat_key,
                    state_features=scheduling_features,
                    min_confidence=0.55,
                )
            )
            reflection_route_payload = relaxed_reflection_route_payload
            if execution_style_guardrail_reduced:
                reflection_route_payload = (
                    runtime_telemetry_store.recommend_route_from_reflections(
                        chat_key=chat_key,
                        state_features=scheduling_features,
                        min_confidence=0.72,
                    )
                )
            routing_decision = self._routing_decision_from_reflection(
                reflection_route_payload
            )
        else:
            execution_style_guardrail_reduced = False
            relaxed_reflection_route_payload = None
        if (
            routing_decision is None
            and chat_key
            and hasattr(runtime_telemetry_store, "recommend_route_from_intent_bucket")
            and intent_bucket.startswith("other|")
        ):
            routing_decision = self._routing_decision_from_intent_cache(
                runtime_telemetry_store.recommend_route_from_intent_bucket(
                    chat_key=chat_key,
                    intent_bucket=intent_bucket,
                ),
                goal=user_input,
            )
        if self._routing_enabled() and routing_decision is None:
            routing_decision = match_rule_based_routing(routing_snapshot)
        should_use_model_router = False
        router_skip_reason = "disabled"
        if self._routing_enabled() and routing_decision is None:
            should_use_model_router, router_skip_reason = should_call_model_router(
                routing_snapshot,
                intent_bucket=intent_bucket,
            )
        if should_use_model_router and hasattr(
            runtime_telemetry_store, "recommend_router_timeout"
        ):
            recommendation = runtime_telemetry_store.recommend_router_timeout(
                base_timeout=routing_timeout,
                chat_key=chat_key or None,
                router_model=resolved_router_model,
            )
            routing_timeout = float(
                recommendation.get("timeout_seconds", routing_timeout)
                or routing_timeout
            )
        routing_started = time.perf_counter()
        if should_use_model_router:
            routing_decision = await route_task(
                self.gateway,
                routing_snapshot,
                heartbeat=heartbeat,
                model_name=configured_router_model,
                timeout=routing_timeout,
            )
        routing_latency_ms = (time.perf_counter() - routing_started) * 1000.0
        task_contract = build_task_contract(
            user_input=user_input,
            chat_key=chat_key,
            execution_constraints=execution_constraints,
            route_mode_override=(
                route_mode_to_contract_mode(routing_decision.route_mode)
                if routing_decision is not None
                else None
            ),
            allow_clarification_override=(
                True
                if routing_decision is not None and routing_decision.need_clarification
                else None
            ),
            metadata_overrides=(
                {"routing_decision": routing_decision.model_dump()}
                if routing_decision is not None
                else None
            ),
        )
        route_mode = (
            route_mode_to_step_kind(routing_decision.route_mode)
            if routing_decision is not None
            else ("debate" if task_contract.mode == "debate" else "tool_workflow")
        )
        reflection_hints_payload: list[dict[str, Any]] = []
        if self._reflection_enabled() and hasattr(
            runtime_telemetry_store, "list_reflection_hints"
        ):
            reflection_hints_payload = runtime_telemetry_store.list_reflection_hints(
                route_mode=route_mode,
                state_features=scheduling_features,
                limit=int(getattr(self.config.system, "reflection_max_hints", 3) or 3),
            )
        scheduling_policy = self.choose_scheduling_policy(features=scheduling_features)
        worker_policy = self.choose_worker_policy(features=scheduling_features)
        guardrail_softened_scheduling = False
        guardrail_softened_worker = False
        if int(scheduling_features.get("independent_subtasks", 1) or 1) >= 2:
            softened_scheduling_policy = self._maybe_soften_policy_from_guardrail(
                scheduling_policy,
                soften_default=bool(
                    reflection_guardrails.get("parallelism", {}).get("soften_default")
                ),
                current_default_action="serial",
                softened_action="bounded_parallel",
                hint="guardrail 放宽默认调度：bounded_parallel",
            )
            guardrail_softened_scheduling = softened_scheduling_policy.get(
                "action_name"
            ) != scheduling_policy.get("action_name")
            scheduling_policy = softened_scheduling_policy
            softened_worker_policy = self._maybe_soften_policy_from_guardrail(
                worker_policy,
                soften_default=bool(
                    reflection_guardrails.get("worker", {}).get("soften_default")
                ),
                current_default_action="deny_worker",
                softened_action="allow_worker",
                hint="guardrail 放宽默认 worker：allow_worker",
            )
            guardrail_softened_worker = softened_worker_policy.get(
                "action_name"
            ) != worker_policy.get("action_name")
            worker_policy = softened_worker_policy
        scheduling_base_action = str(scheduling_policy.get("action_name", "") or "")
        worker_base_action = str(worker_policy.get("action_name", "") or "")
        scheduling_policy = self._maybe_override_policy_from_reflection(
            scheduling_policy,
            preferred_action=self._select_reflection_override(
                (
                    reflection_hints_payload
                    if str(
                        reflection_guardrails.get("parallelism", {}).get(
                            "injection_level", "normal"
                        )
                    )
                    != "reduced"
                    else []
                ),
                allowed_actions={"serial", "bounded_parallel"},
            ),
            hint_prefix="历史反思建议调度动作：",
        )
        worker_policy = self._maybe_override_policy_from_reflection(
            worker_policy,
            preferred_action=self._select_reflection_override(
                (
                    reflection_hints_payload
                    if str(
                        reflection_guardrails.get("worker", {}).get(
                            "injection_level", "normal"
                        )
                    )
                    != "reduced"
                    else []
                ),
                allowed_actions={"allow_worker", "deny_worker"},
            ),
            hint_prefix="历史反思建议 worker 动作：",
        )
        scheduling_overridden = (
            str(scheduling_policy.get("action_name", "") or "")
            != scheduling_base_action
        )
        worker_overridden = (
            str(worker_policy.get("action_name", "") or "") != worker_base_action
        )
        self._record_routing_telemetry(
            chat_key=chat_key,
            flow_id=flow_id,
            route_mode=route_mode,
            resolved_router_model=resolved_router_model,
            routing_latency_ms=routing_latency_ms,
            routing_decision=routing_decision,
            router_skip_reason=router_skip_reason,
            should_use_model_router=should_use_model_router,
            intent_bucket=intent_bucket,
            reflection_hints_payload=reflection_hints_payload,
            scheduling_overridden=scheduling_overridden,
            worker_overridden=worker_overridden,
            execution_style_guardrail_reduced=execution_style_guardrail_reduced,
            relaxed_reflection_route_payload=relaxed_reflection_route_payload,
            guardrail_softened_scheduling=guardrail_softened_scheduling,
            guardrail_softened_worker=guardrail_softened_worker,
        )
        execution_plan = build_execution_plan(task_contract)
        plan_notebook = compile_execution_plan_to_notebook(
            execution_plan,
            flow_id=flow_id,
            metadata={
                "chat_key": chat_key,
                "routing_decision": (
                    routing_decision.model_dump()
                    if routing_decision is not None
                    and hasattr(routing_decision, "model_dump")
                    else routing_decision
                ),
            },
        )
        policy_hints = [decomposition_hint]
        policy_hints.extend(self._routing_policy_hints(routing_decision))
        policy_hints.extend(build_execution_constraint_hints(execution_constraints))
        for payload in (scheduling_policy, worker_policy):
            hint = str(payload.get("hint", "") or "").strip()
            if hint and hint not in policy_hints:
                policy_hints.append(hint)
        for payload in reflection_hints_payload:
            hint = self._format_reflection_hint(payload)
            if hint not in policy_hints:
                policy_hints.append(hint)
        await self._maybe_send_debug_policy_summary(
            send_intermediate_message=send_intermediate_message,
            flow_id=flow_id,
            decomposition_action=decomposition_action,
            decomposition_hint=decomposition_hint,
            route_mode=route_mode,
            router_skip_reason=router_skip_reason,
            routing_decision=routing_decision,
            scheduling_policy=scheduling_policy,
            worker_policy=worker_policy,
        )

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
                    ("policy_hints", policy_hints),
                    ("routing_snapshot", routing_snapshot),
                    ("routing_decision", routing_decision),
                    ("execution_constraints", execution_constraints),
                    ("task_contract", task_contract),
                    ("execution_plan", execution_plan),
                    ("plan_notebook", plan_notebook),
                    ("plan_notebook_id", plan_notebook.notebook_id),
                    (
                        "current_notebook_node_id",
                        plan_notebook.primary_frontier_node_id(),
                    ),
                    (
                        "notebook_context_budget",
                        max(1200, int(self.config.system.context_history_tokens)),
                    ),
                ]
                if v is not None
            },
        )
        if chat_key:
            context.emit(
                "policy_decision",
                decision_kind="decomposition",
                action_name=decomposition_action,
                state_features=decomposition_features,
                explain=decomposition_hint,
                state_bucket=build_policy_state_bucket(decomposition_features),
            )
        if chat_key and self._policy_learning_enabled():
            worker_action_name = str(worker_policy.get("action_name", "") or "").strip()
            if worker_action_name:
                context.emit(
                    "policy_decision",
                    decision_kind="worker",
                    action_name=worker_action_name,
                    state_features=scheduling_features,
                    explain=str(worker_policy.get("explain", "") or ""),
                    state_bucket=str(worker_policy.get("state_bucket", "") or ""),
                )
        if chat_key and self._policy_learning_enabled():
            scheduling_action_name = str(
                scheduling_policy.get("action_name", "") or ""
            ).strip()
            if scheduling_action_name:
                context.emit(
                    "policy_decision",
                    decision_kind="scheduling",
                    action_name=scheduling_action_name,
                    state_features=scheduling_features,
                    explain=str(scheduling_policy.get("explain", "") or ""),
                    state_bucket=str(scheduling_policy.get("state_bucket", "") or ""),
                )

        logger.info("DynamicOrchestrator created, starting run flow_id=%s", flow_id)
        execution_started = time.perf_counter()
        try:
            if heartbeat is not None:
                result = await heartbeat.watch(
                    orchestrator.run(goal=task_contract.goal, context=context),
                )
            else:
                result = await orchestrator.run(
                    goal=task_contract.goal, context=context
                )
        except Exception as exc:
            self._persist_policy_events(
                flow_id=flow_id,
                chat_key=chat_key,
                events=list(context.events),
            )
            failed_outcome = self._policy_outcome_details(
                context.events,
                error=str(exc),
                execution_elapsed_ms=(time.perf_counter() - execution_started) * 1000.0,
            )
            self._record_policy_outcome(
                PolicyOutcomeRecord(
                    flow_id=flow_id,
                    chat_key=chat_key,
                    final_status="failed",
                    reward=self._policy_reward(context.events, "failed"),
                    outcome=failed_outcome,
                )
            )
            self._spawn_background_task(
                self._evaluate_task_run_async(
                    chat_key=chat_key,
                    route_mode=route_mode,
                    state_features=scheduling_features,
                    routing_decision=routing_decision,
                    final_status="failed",
                    outcome=failed_outcome,
                ),
                label="task-evaluator-failed",
            )
            raise
        self._persist_policy_events(
            flow_id=flow_id,
            chat_key=chat_key,
            events=list(context.events),
        )
        succeeded_outcome = self._policy_outcome_details(
            context.events,
            result=result,
            execution_elapsed_ms=(time.perf_counter() - execution_started) * 1000.0,
        )
        self._record_policy_outcome(
            PolicyOutcomeRecord(
                flow_id=flow_id,
                chat_key=chat_key,
                final_status="succeeded",
                reward=self._policy_reward(context.events, "succeeded", result=result),
                outcome=succeeded_outcome,
            )
        )
        self._spawn_background_task(
            self._evaluate_task_run_async(
                chat_key=chat_key,
                route_mode=route_mode,
                state_features=scheduling_features,
                routing_decision=routing_decision,
                final_status="succeeded",
                outcome=succeeded_outcome,
            ),
            label="task-evaluator-succeeded",
        )

        text = result.conclusion or "任务完成，但没有可返回的结果。"
        self._persist_notebook_completion(
            chat_key=chat_key,
            context=context,
            final_text=text,
        )
        collected_media = RuntimeState(context).collected_media_bucket()
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
        decision_cache = getattr(self, "_recent_policy_decisions_by_flow", {}) or {}
        policy_decisions = list(decision_cache.get(resolved_flow_id, []) or [])
        parts = ["[Runtime Flow]", f"flow_id={resolved_flow_id}"]
        if resolved_chat_key:
            parts.append(f"chat_key={resolved_chat_key}")
        if policy_decisions:
            lines = []
            for item in policy_decisions[-8:]:
                kind = str(item.get("decision_kind", "") or "").strip()
                action = str(item.get("action_name", "") or "").strip()
                state_bucket = str(item.get("state_bucket", "") or "").strip()
                explain = str(item.get("explain", "") or "").strip()
                suffix = []
                if state_bucket:
                    suffix.append(f"bucket={state_bucket}")
                if explain:
                    suffix.append(explain)
                lines.append(
                    f"- decision_kind={kind or '-'} action={action or '-'}"
                    + (f" ({'; '.join(suffix)})" if suffix else "")
                )
            parts.append("[Policy Decisions]\n" + "\n".join(lines))
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

    def inspect_policy(self, chat_key: str = "", decision_kind: str = "") -> str:
        kinds = (
            [decision_kind.strip()]
            if decision_kind.strip()
            else ["decomposition", "scheduling", "worker"]
        )
        parts = ["[Policy]"]
        if chat_key:
            parts.append(f"chat_key={chat_key}")
        for kind in kinds:
            stats = self._policy_store.summarize_action_stats(decision_kind=kind)
            parts.append(f"decision_kind={kind}")
            if not stats:
                parts.append("- no_stats")
                continue
            ranked = sorted(
                stats.items(),
                key=lambda item: (
                    float(
                        item[1].get("effective_samples", item[1].get("samples", 0.0))
                        or 0.0
                    ),
                    float(item[1].get("mean_reward", 0.0) or 0.0),
                ),
                reverse=True,
            )
            for action_name, payload in ranked[:5]:
                parts.append(
                    "- "
                    + f"action={action_name} "
                    + f"samples={int(payload.get('samples', 0) or 0)} "
                    + f"effective_samples={float(payload.get('effective_samples', payload.get('samples', 0.0)) or 0.0):.2f} "
                    + f"mean_reward={float(payload.get('mean_reward', 0.0) or 0.0):.2f} "
                    + f"recent_mean_reward={float(payload.get('recent_mean_reward', 0.0) or 0.0):.2f} "
                    + f"drift_score={float(payload.get('drift_score', 0.0) or 0.0):.2f} "
                    + f"failure_rate={float(payload.get('failure_rate', 0.0) or 0.0):.2f} "
                    + f"avg_execution_elapsed_ms={float(payload.get('avg_execution_elapsed_ms', 0.0) or 0.0):.2f} "
                    + f"avg_tool_call_count={float(payload.get('avg_tool_call_count', 0.0) or 0.0):.2f} "
                    + f"tool_failure_rate={float(payload.get('tool_failure_rate', 0.0) or 0.0):.2f} "
                    + f"loop_guard_block_rate={float(payload.get('loop_guard_block_rate', 0.0) or 0.0):.2f} "
                    + f"max_step_exhausted_rate={float(payload.get('max_step_exhausted_rate', 0.0) or 0.0):.2f} "
                    + f"feedback_score={float(payload.get('feedback_score', 0.0) or 0.0):.2f}"
                )
        telemetry_summary_fn = getattr(
            self._policy_store, "summarize_runtime_telemetry", None
        )
        if callable(telemetry_summary_fn):
            try:
                telemetry = telemetry_summary_fn(chat_key=chat_key or None)
            except TypeError:
                telemetry = telemetry_summary_fn()

            def _format_skip_breakdown(payload: dict[str, object]) -> str:
                breakdown = payload.get("skip_breakdown")
                if not isinstance(breakdown, dict) or not breakdown:
                    return ""
                items: list[str] = []
                for reason, count in sorted(
                    (
                        (str(reason).strip() or "unknown", int(value or 0))
                        for reason, value in breakdown.items()
                    ),
                    key=lambda item: (-item[1], item[0]),
                ):
                    items.append(f"{reason}:{count}")
                return ",".join(items)

            overall = telemetry.get("overall") if isinstance(telemetry, dict) else None
            by_route_mode = (
                telemetry.get("by_route_mode", {})
                if isinstance(telemetry, dict)
                else {}
            )

            # Fields shared by both overall and per-route-mode rows.
            _TELEMETRY_FLOAT_FIELDS = (
                "avg_router_latency_ms",
                "avg_execution_elapsed_ms",
                "avg_task_result_count",
                "avg_executor_step_count",
                "avg_tool_call_count",
                "fallback_rate",
                "skipped_rate",
                "model_route_rate",
                "rule_hit_rate",
                "reflection_route_rate",
                "reflection_match_rate",
                "reflection_override_rate",
                "tool_failure_rate",
                "loop_guard_block_rate",
                "max_step_exhausted_rate",
                "dead_letter_rate",
                "stalled_rate",
                "execution_style_reflection_rate",
                "parallelism_reflection_rate",
                "worker_reflection_rate",
                "execution_style_guardrail_reduce_rate",
                "parallelism_guardrail_soften_rate",
                "worker_guardrail_soften_rate",
                "mean_reward",
            )

            def _format_telemetry_row(
                payload: dict[str, Any],
                *,
                prefix: str = "",
            ) -> str:
                tokens: list[str] = []
                if prefix:
                    tokens.append(prefix)
                tokens.append(f"runs={int(payload.get('runs', 0) or 0)}")
                for field_name in _TELEMETRY_FLOAT_FIELDS:
                    tokens.append(
                        f"{field_name}={float(payload.get(field_name, 0.0) or 0.0):.2f}"
                    )
                skip_bd = _format_skip_breakdown(payload)
                if skip_bd:
                    tokens.append(f"skip_breakdown={skip_bd}")
                return "- " + " ".join(tokens)

            if isinstance(overall, dict) and int(overall.get("runs", 0) or 0) > 0:
                parts.append("[Routing Telemetry]")
                parts.append(_format_telemetry_row(overall))
                if isinstance(by_route_mode, dict):
                    for route_mode, payload in sorted(by_route_mode.items()):
                        if not isinstance(payload, dict):
                            continue
                        parts.append(
                            _format_telemetry_row(
                                payload, prefix=f"route_mode={route_mode}"
                            )
                        )
        return "\n".join(parts)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            logger.info("Initializing resource manager...")
            await self.resource_manager.initialize_async()
            self._initialized = True
            logger.info("Resource manager initialized")

    def _prepare_tape(
        self,
        *,
        chat_key: str,
        user_input: str,
        media_paths: list[str] | None = None,
    ) -> Tape | None:
        if not chat_key or self.tape_store is None:
            return None
        logger.info("Loading tape for chat_key=%s", chat_key)
        tape = self.tape_store.get_or_create(chat_key)
        logger.info("Tape loaded, observing user message...")
        if hasattr(self, "memory_store"):
            self.memory_store.observe_user_message(chat_key, user_input)
        pending_entries = []
        if tape.last_anchor() is None:
            anchor = tape.append("anchor", {"name": "session/start", "state": {}})
            pending_entries.append(anchor)
        content_for_tape = user_input
        if media_paths:
            content_for_tape = f"{user_input}\n[附带 {len(media_paths)} 张图片]"
        user_entry = tape.append(
            "message", {"role": "user", "content": content_for_tape}
        )
        pending_entries.append(user_entry)
        self.tape_store.save_entries(chat_key, pending_entries)
        logger.info("Tape entries saved, proceeding to _answer_with_dag")
        return tape

    def _create_runtime_job(
        self,
        *,
        chat_key: str,
        user_input: str,
        media_paths: list[str] | None = None,
        job_metadata_override: dict[str, Any] | None = None,
    ) -> Any | None:
        runtime_job_store = getattr(self, "_runtime_job_store", None)
        if not chat_key or runtime_job_store is None:
            return None
        runtime_metadata = dict(job_metadata_override or {})
        runtime_metadata.setdefault("media_paths", list(media_paths or []))
        runtime_job = runtime_job_store.create(
            chat_key=chat_key,
            goal=user_input,
            metadata=runtime_metadata,
        )
        runtime_job_store.transition(
            runtime_job.job_id,
            "planning",
            progress_message="已接收任务，准备执行",
        )
        return runtime_job

    @staticmethod
    async def _invoke_callback(
        callback: Callable[[Any], Awaitable[None] | None] | None,
        payload: Any,
    ) -> None:
        """Invoke an optional callback that may return a coroutine or None."""
        if callback is None:
            return
        maybe = callback(payload)
        if inspect.isawaitable(maybe):
            await maybe

    def _build_runtime_event_recorder(
        self,
        *,
        chat_key: str,
        tape: Tape | None,
        runtime_job: Any | None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None,
    ) -> Callable[[Any], Awaitable[None] | None] | None:
        runtime_job_store = getattr(self, "_runtime_job_store", None)
        if runtime_job is None and (tape is None or not chat_key):
            return runtime_event_callback

        async def _record_runtime_event(event: Any) -> None:
            payload = self._runtime_event_payload(event)
            if runtime_job_store is not None and runtime_job is not None:
                payload["job_id"] = str(payload.get("job_id", "") or runtime_job.job_id)
                inner_payload = dict(payload.get("payload") or {})
                inner_payload.setdefault("job_id", runtime_job.job_id)
                payload["payload"] = inner_payload
                state, progress_message = self._job_state_from_runtime_event(payload)
                metadata_update: dict[str, Any] = {}
                flow_id = str(payload.get("flow_id", "") or "").strip()
                task_id = str(payload.get("task_id", "") or "").strip()
                stage = str(inner_payload.get("stage", "") or "").strip()
                if flow_id:
                    metadata_update["flow_id"] = flow_id
                if task_id:
                    metadata_update["last_task_id"] = task_id
                if stage:
                    metadata_update["last_stage"] = stage
                runtime_job_store.transition(
                    runtime_job.job_id,
                    state,
                    progress_message=progress_message,
                    metadata=metadata_update,
                )
            if tape is not None and chat_key:
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
            await self._invoke_callback(runtime_event_callback, payload)

        return _record_runtime_event

    def _record_assistant_reply(
        self,
        *,
        chat_key: str,
        tape: Tape | None,
        text: str,
    ) -> None:
        if not tape or not chat_key:
            return
        asst_entry = tape.append("message", {"role": "assistant", "content": text})
        self.tape_store.save_entry(chat_key, asst_entry)
        self._spawn_background_task(
            self._maybe_handoff(tape, chat_key),
            label=f"handoff:{chat_key}",
        )

    async def process_task(
        self,
        user_input: str,
        chat_key: str = "",
        heartbeat: Heartbeat | None = None,
        media_paths: list[str] | None = None,
        stream_callback: StreamTextCallback | None = None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None = None,
        interactive_output_callback: InteractiveOutputCallback | None = None,
        send_intermediate_message: Callable[[str], Awaitable[None]] | None = None,
        job_metadata_override: dict[str, Any] | None = None,
    ) -> TaskResponse:
        policy_control = self._parse_policy_feedback_command(user_input)
        if policy_control is not None:
            return await self._handle_policy_feedback_command(chat_key, policy_control)
        job_control = self._parse_job_command(user_input)
        if job_control is not None:
            return await self._handle_job_command(chat_key, job_control)
        await self._ensure_initialized()
        if heartbeat is not None:
            heartbeat.beat()

        if chat_key and getattr(self, "_interactive_sessions", None) is not None:
            control = self._parse_interactive_session_command(user_input)
            if control is not None:
                return await self._handle_interactive_session_command(chat_key, control)
            if self._interactive_sessions.has_active_session(chat_key):
                reply = await self._handle_interactive_session_message(
                    chat_key,
                    user_input,
                    media_paths=media_paths,
                    heartbeat=heartbeat,
                    runtime_event_callback=runtime_event_callback,
                    interactive_output_callback=interactive_output_callback,
                )
                if reply is not None:
                    return reply

        tape = self._prepare_tape(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
        )
        runtime_job = self._create_runtime_job(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
            job_metadata_override=job_metadata_override,
        )
        wrapped_runtime_event_callback = self._build_runtime_event_recorder(
            chat_key=chat_key,
            tape=tape,
            runtime_job=runtime_job,
            runtime_event_callback=runtime_event_callback,
        )

        try:
            logger.info("Starting _answer_with_dag")
            runtime_job_store = getattr(self, "_runtime_job_store", None)
            if runtime_job is not None:
                runtime_job_store.transition(
                    runtime_job.job_id,
                    "running",
                    progress_message="编排执行中",
                )
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

            self._record_assistant_reply(chat_key=chat_key, tape=tape, text=text)
            if runtime_job is not None:
                runtime_job_store.transition(
                    runtime_job.job_id,
                    "completed",
                    progress_message="执行完成",
                    result_text=text,
                )

            return TaskResponse(text=text, media_paths=collected_media)
        except Exception as exc:
            runtime_job_store = getattr(self, "_runtime_job_store", None)
            if runtime_job is not None:
                runtime_job_store.transition(
                    runtime_job.job_id,
                    "failed",
                    progress_message="执行失败",
                    error=str(exc),
                )
            logger.exception("Error processing task")
            return TaskResponse(text=f"处理任务时出错：{exc}", is_error=True)

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

    @staticmethod
    def _parse_policy_feedback_command(
        user_input: str,
    ) -> dict[str, str] | None:
        return parse_policy_command(user_input)

    @staticmethod
    def _parse_job_command(
        user_input: str,
    ) -> dict[str, str] | None:
        text = (user_input or "").strip()
        if not text.lower().startswith("@job"):
            return None
        parts = text.split()
        action = parts[1].lower() if len(parts) >= 2 else "status"
        target = parts[2].strip() if len(parts) >= 3 else "latest"
        return {"action": action, "target": target}

    @staticmethod
    def _runtime_event_payload(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            payload = dict(event)
        else:
            payload = {
                "event": getattr(event, "event", ""),
                "task_id": getattr(event, "task_id", ""),
                "flow_id": getattr(event, "flow_id", ""),
                "payload": dict(getattr(event, "payload", {}) or {}),
            }
        payload["payload"] = dict(payload.get("payload") or {})
        return payload

    @staticmethod
    def _job_state_from_runtime_event(event_payload: dict[str, Any]) -> tuple[str, str]:
        state, progress_message = project_job_state_from_runtime_event(event_payload)
        if state not in JOB_STATES:
            return "running", progress_message
        return state, progress_message

    def _resolve_job_target(
        self,
        *,
        chat_key: str,
        target: str,
    ) -> tuple[Any, str]:
        normalized_target = str(target or "latest").strip() or "latest"
        runtime_job_store = getattr(self, "_runtime_job_store", None)
        if runtime_job_store is None:
            return None, "当前未启用运行时作业跟踪。"
        job = (
            runtime_job_store.latest_for_chat(chat_key)
            if normalized_target == "latest"
            else runtime_job_store.get(normalized_target)
        )
        if job is None:
            return None, "未找到对应作业。"
        return job, ""

    def _runtime_maintenance_report(self) -> str:
        runtime_job_store = getattr(self, "_runtime_job_store", None)
        report = (
            runtime_job_store.run_maintenance(retention_seconds=0)
            if runtime_job_store is not None
            else {}
        )
        interactive_sessions = getattr(self, "_interactive_sessions", None)
        stale_sessions = (
            int(interactive_sessions.cleanup())
            if interactive_sessions is not None
            and hasattr(interactive_sessions, "cleanup")
            else 0
        )
        recent_flows = getattr(self, "_recent_flows_by_chat", {}) or {}
        known_flow_ids = (
            runtime_job_store.known_flow_ids() if runtime_job_store is not None else set()
        )
        unmatched_recent_flows = sorted(
            {
                str(flow_id).strip()
                for flows in recent_flows.values()
                for flow_id in flows or []
                if str(flow_id).strip() and str(flow_id).strip() not in known_flow_ids
            }
        )
        lines = [
            "[Runtime Maintenance]",
            f"orphaned_jobs_pruned={int(report.get('orphaned_jobs_pruned', 0) or 0)}",
            f"stale_interactive_sessions={stale_sessions}",
            f"unmatched_recent_flows={len(unmatched_recent_flows)}",
        ]
        orphaned_job_ids = report.get("orphaned_job_ids") or []
        if unmatched_recent_flows:
            lines.append(f"flows={', '.join(unmatched_recent_flows[:10])}")
        if orphaned_job_ids:
            lines.append(
                f"jobs={', '.join(str(item) for item in orphaned_job_ids[:10])}"
            )
        return "\n".join(lines)

    def _resolve_policy_feedback_flow_id(
        self,
        *,
        chat_key: str,
        target: str,
    ) -> tuple[str, str]:
        normalized_target = str(target or "").strip()
        store = getattr(self, "_runtime_job_store", None)
        recent_latest = getattr(self, "_recent_flow_ids_by_chat", {}) or {}
        recent_history = getattr(self, "_recent_flows_by_chat", {}) or {}
        history = [
            str(item).strip()
            for item in recent_history.get(chat_key, []) or []
            if str(item).strip()
        ]
        if normalized_target and normalized_target not in {"latest"}:
            target_job = store.get(normalized_target) if store is not None else None
            if target_job is not None:
                flow_id = str(target_job.metadata.get("flow_id", "") or "").strip()
                if flow_id:
                    return flow_id, ""
                return "", "该 job 尚未关联 flow_id，请指定 flow_id 再反馈。"
        if not history:
            latest = str(recent_latest.get(chat_key, "") or "").strip()
            if latest:
                history = [latest]
        if normalized_target and normalized_target not in {"latest"}:
            return normalized_target, ""
        if not history:
            return "", "当前会话没有可反馈的最近任务。"
        if normalized_target == "latest" and len(history) > 1:
            return "", "最近有多个运行记录，请指定 flow_id 再反馈。"
        return history[0], ""

    async def _handle_policy_feedback_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> TaskResponse:
        if control.get("action", "") == "inspect":
            target = str(control.get("target", "") or "").strip()
            if target and not any(
                target == decision_kind
                for decision_kind in ("decomposition", "scheduling", "worker")
            ):
                return TaskResponse(text=self.inspect_runtime_flow(flow_id=target))
            return TaskResponse(
                text=self.inspect_policy(
                    chat_key=chat_key,
                    decision_kind=target,
                )
            )
        if control.get("action", "") != "feedback":
            return TaskResponse(
                text="支持的命令：@policy feedback <flow_id|latest> good|bad <reason> / @policy inspect [decision_kind|flow_id]"
            )
        rating = str(control.get("rating", "") or "").strip().lower()
        reason = str(control.get("reason", "") or "").strip()
        if rating not in {"good", "bad"} or not reason:
            return TaskResponse(
                text="用法：@policy feedback <flow_id|latest> good|bad <reason>"
            )
        if not chat_key:
            return TaskResponse(text="缺少 chat_key，无法记录策略反馈。")
        flow_id, error = self._resolve_policy_feedback_flow_id(
            chat_key=chat_key,
            target=str(control.get("target", "") or "").strip(),
        )
        if error:
            return TaskResponse(text=error)
        policy_store = getattr(self, "_policy_store", None)
        if policy_store is None:
            return TaskResponse(text="当前未启用策略反馈存储。")
        policy_store.record_feedback(
            flow_id=flow_id,
            chat_key=chat_key,
            rating=rating,
            reason=reason,
        )
        return TaskResponse(text=f"已记录策略反馈：{rating}。")

    async def _handle_job_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> TaskResponse:
        action = str(control.get("action", "") or "status").strip()
        target = str(control.get("target", "") or "latest").strip()
        if action == "cleanup":
            return TaskResponse(text=self._runtime_maintenance_report())
        if action not in {"status", "resume"}:
            return TaskResponse(
                text="支持的命令：@job status <job_id|latest> / @job resume <job_id|latest> / @job cleanup"
            )
        job, error = self._resolve_job_target(chat_key=chat_key, target=target)
        if error:
            return TaskResponse(text=error)
        if job is None:
            return TaskResponse(text="未找到对应作业。")
        if action == "resume":
            if job.state == "completed":
                return TaskResponse(text="该作业已完成，无需恢复。")
            if job.state == "running":
                return TaskResponse(text="该作业仍在运行，请先使用 @job status 查询。")
            runtime_job_store = getattr(self, "_runtime_job_store", None)
            if runtime_job_store is None:
                return TaskResponse(text="当前未启用运行时作业跟踪。")
            runtime_job_store.transition(
                job.job_id,
                "repairing",
                progress_message="准备恢复执行",
                metadata={"resume_requested": True},
            )
            resume_metadata = dict(job.metadata)
            resume_metadata["resumed_from"] = job.job_id
            return await self.process_task(
                job.goal,
                chat_key=job.chat_key or chat_key,
                media_paths=list(job.metadata.get("media_paths") or []),
                job_metadata_override=resume_metadata,
            )
        return TaskResponse(
            text=(
                f"[Job]\njob_id={job.job_id}\nstate={job.state}\n"
                f"progress={job.progress_message or '-'}\n"
                f"flow_id={str(job.metadata.get('flow_id', '') or '-')}"
            )
        )

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
            backend_bits: list[str] = []
            backend_status = dict(status.backend_status or {})
            mode = str(backend_status.get("mode", "") or status.mode).strip()
            if mode:
                backend_bits.append(f"mode={mode}")
            pid = backend_status.get("pid", status.process_pid)
            if pid:
                backend_bits.append(f"pid={pid}")
            alive = backend_status.get("alive")
            if alive is not None:
                backend_bits.append(f"alive={bool(alive)}")
            return TaskResponse(
                text=(
                    f"当前交互会话：{status.backend_name} "
                    f"(session_id={status.session_id}"
                    + (f", {', '.join(backend_bits)}" if backend_bits else "")
                    + ")"
                )
            )
        return TaskResponse(text="支持的命令：@session start <backend> / status / stop")

    async def _handle_interactive_session_message(
        self,
        chat_key: str,
        user_input: str,
        *,
        media_paths: list[str] | None = None,
        heartbeat: Heartbeat | None = None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None = None,
        interactive_output_callback: InteractiveOutputCallback | None = None,
    ) -> TaskResponse | None:
        task_contract = build_task_contract(
            user_input=user_input,
            chat_key=chat_key,
        )
        request = InteractiveRequest(
            text=user_input,
            media_paths=tuple(media_paths or ()),
            contract_mode=task_contract.mode,
        )
        await self._invoke_callback(
            runtime_event_callback,
            {
                "event": "running",
                "task_id": "interactive_session",
                "flow_id": f"interactive:{chat_key}",
                "payload": {
                    "stage": "interactive_session",
                    "state": "running",
                    "message": "交互会话处理中",
                },
            },
        )
        try:
            if heartbeat is not None:
                async with heartbeat.keep_alive():
                    reply = await self._interactive_sessions.send(
                        chat_key,
                        request,
                        output_event_callback=interactive_output_callback,
                    )
            else:
                reply = await self._interactive_sessions.send(
                    chat_key,
                    request,
                    output_event_callback=interactive_output_callback,
                )
        except RuntimeError:
            logger.warning(
                "Interactive session send failed; falling back to DAG chat_key=%s",
                chat_key,
                exc_info=True,
            )
            with contextlib.suppress(Exception):
                await self._interactive_sessions.stop(chat_key, reason="backend_failed")
            return None
        if reply.expired:
            return None
        tape = self._prepare_tape(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
        )
        if tape is not None:
            assistant_entry = tape.append(
                "message", {"role": "assistant", "content": reply.text}
            )
            self.tape_store.save_entries(chat_key, [assistant_entry])
        await self._invoke_callback(
            runtime_event_callback,
            {
                "event": "completed",
                "task_id": "interactive_session",
                "flow_id": f"interactive:{chat_key}",
                "payload": {
                    "stage": "interactive_session",
                    "state": "completed",
                    "message": "交互会话完成",
                },
            },
        )
        return TaskResponse(
            text=reply.text,
            media_paths=list(reply.media_paths or []),
        )

    def reset(self) -> None:
        interactive_sessions = getattr(self, "_interactive_sessions", None)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if interactive_sessions is not None:
                asyncio.run(interactive_sessions.stop_all(reason="reset"))
        else:
            if interactive_sessions is not None:
                self._spawn_background_task(
                    interactive_sessions.stop_all(reason="reset"),
                    label="interactive-reset",
                )
        resource_manager = getattr(self, "resource_manager", None)
        if resource_manager is not None:
            resource_manager.reset()
        tape_store = getattr(self, "tape_store", None)
        if tape_store is not None:
            tape_store.clear()
        self._initialized = False

    def get_status(self) -> dict[str, Any]:
        resource_manager = getattr(self, "resource_manager", None)
        if resource_manager is not None:
            status = {
                "resource_manager": "initialized",
                "available_tools": len(resource_manager.get_available_tools()),
                "resources": resource_manager.search_resources(),
            }
        else:
            status = {
                "resource_manager": "unavailable",
                "available_tools": 0,
                "resources": [],
            }
        interactive_sessions = getattr(self, "_interactive_sessions", None)
        status["interactive_sessions"] = (
            interactive_sessions.summary() if interactive_sessions is not None else {}
        )
        telemetry_summary_fn = getattr(
            getattr(self, "_policy_store", None), "summarize_runtime_telemetry", None
        )
        if callable(telemetry_summary_fn):
            try:
                telemetry = telemetry_summary_fn()
            except TypeError:
                telemetry = telemetry_summary_fn(chat_key=None)
            overall = telemetry.get("overall") if isinstance(telemetry, dict) else None
            if isinstance(overall, dict) and int(overall.get("runs", 0) or 0) > 0:
                status["policy_telemetry"] = overall
        return status
