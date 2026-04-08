"""Orchestrator built on lightweight kernel — DAG-driven multi-agent mode."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
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
from .execution_outcome import (
    build_execution_outcome,
    compute_policy_reward,
    summarize_task_results,
)
from .feedback_events import (
    normalize_runtime_feedback_event,
    runtime_event_primary_label,
)
from .memory_store import HybridMemoryStore
from .heartbeat import TaskHeartbeatRegistry
from .interactive_sessions import InteractiveSessionManager
from .interactive_sessions.backends import ClaudeInteractiveBackend
from .interactive_sessions.types import InteractiveOutputCallback
from .model_gateway import OpenAICompatibleGateway
from .orchestration_policy import ConservativePolicySelector, build_policy_state_bucket
from .orchestration_policy_store import OrchestrationPolicyStore
from .orchestrator_command_dispatch import CommandDispatch
from .orchestrator_handoff import HandoffManager
from .orchestrator_inspect import InspectService
from .orchestrator_interactive_support import OrchestratorInteractiveSessionSupport
from .orchestrator_policy_engine import PolicyEngine
from .orchestrator_routing_cascade import RoutingCascade
from .orchestrator_runtime_support import OrchestratorRuntimeSupport
from .orchestration_routing_result import RoutingResult
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
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # ── Service layer (single-responsibility components) ─────────────
        self._inspect_service = InspectService(
            policy_store=self._policy_store,
            tape_store=self.tape_store,
            memory_store=self.memory_store,
            heartbeat_registry=self._task_heartbeat_registry,
            child_task_bus=self._child_task_bus,
        )
        self._policy_engine = PolicyEngine(
            config=self.config,
            policy_store=self._policy_store,
            inspect_service=self._inspect_service,
        )
        self._handoff_manager = HandoffManager(
            gateway=self.gateway,
            tape_store=self.tape_store,
            memory_store=self.memory_store,
            compact_threshold=self.config.system.context_compact_threshold,
        )
        self._routing_cascade = RoutingCascade(
            policy_engine=self._policy_engine,
            policy_store=self._policy_store,
            gateway=self.gateway,
            config=self.config,
        )

        # ── Command dispatch service ──────────────────────────────────────
        self._command_dispatch = CommandDispatch(
            inspect_service=self._inspect_service,
            policy_store=self._policy_store,
            job_store=self._runtime_job_store,
            process_task_fn=self.process_task,
            runtime_support_fn=self._get_runtime_support,
            interactive_support_fn=self._get_interactive_support,
            response_factory=TaskResponse,
        )

        # Backwards-compatibility shims: keep these attributes pointing at
        # the data owned by InspectService so any code still referencing
        # self._recent_flow_ids_by_chat etc. continues to work during the
        # incremental migration.
        self._recent_flow_ids_by_chat = self._inspect_service._recent_flow_ids_by_chat
        self._recent_flows_by_chat = self._inspect_service._recent_flows_by_chat
        self._recent_policy_decisions_by_flow = (
            self._inspect_service._recent_policy_decisions_by_flow
        )

    def _build_interactive_session_manager(self) -> InteractiveSessionManager:
        return InteractiveSessionManager(
            backends={
                "claude": ClaudeInteractiveBackend(
                    workspace_root=self.config.workspace_dir,
                )
            },
            max_age_seconds=self.config.system.interactive_session_max_age_seconds,
        )

    def _get_runtime_support(self) -> OrchestratorRuntimeSupport:
        handoff_fn = None
        hm = getattr(self, "_handoff_manager", None)
        if hm is not None:
            handoff_fn = hm.maybe_handoff
        else:
            maybe_handoff_method = getattr(self, "_maybe_handoff", None)
            if callable(maybe_handoff_method):
                handoff_fn = maybe_handoff_method
        return OrchestratorRuntimeSupport(
            config=self.config,
            tape_store=getattr(self, "tape_store", None),
            memory_store=getattr(self, "memory_store", None),
            runtime_job_store=getattr(self, "_runtime_job_store", None),
            invoke_callback=self._invoke_callback,
            spawn_background_task=self._spawn_background_task,
            maybe_handoff=handoff_fn,
        )

    def _get_interactive_support(self) -> OrchestratorInteractiveSessionSupport:
        return OrchestratorInteractiveSessionSupport(
            session_manager=self._interactive_sessions,
            invoke_callback=self._invoke_callback,
            prepare_tape=self._prepare_tape,
            tape_store=getattr(self, "tape_store", None),
            response_factory=TaskResponse,
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
        svc = self._get_or_create_inspect_service()
        if svc is not None:
            svc.record_flow(chat_key, flow_id)

    def _remember_flow_policy_decisions(
        self,
        flow_id: str,
        events: list[dict[str, Any]],
    ) -> None:
        svc = self._get_or_create_inspect_service()
        if svc is not None:
            svc.record_policy_decisions(flow_id, events)

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

    def _get_handoff_lock(self, chat_key: str) -> asyncio.Lock:
        """Get per-chat lock for handoff. Delegates to HandoffManager or falls back."""
        hm = getattr(self, "_handoff_manager", None)
        if hm is not None:
            return hm._get_lock(chat_key)
        # Fallback for partial test doubles: manage locks directly on the agent
        handoff_locks = getattr(self, "_handoff_locks", None)
        if not isinstance(handoff_locks, dict):
            handoff_locks = {}
            self._handoff_locks = handoff_locks
        lock = handoff_locks.pop(chat_key, None)
        if lock is None:
            lock = asyncio.Lock()
        handoff_locks[chat_key] = lock
        # LRU eviction
        limit = getattr(self, "_HANDOFF_LOCK_LIMIT", 256)
        while len(handoff_locks) > limit:
            oldest_key = next(iter(handoff_locks))
            del handoff_locks[oldest_key]
        return lock

    @staticmethod
    def _build_notebook_completion_summary(
        notebook: PlanNotebook,
        *,
        final_text: str,
    ) -> dict[str, Any]:
        existing = dict(notebook.completion_summary or {})
        final_summary = str(
            existing.get("final_summary", "") or final_text or ""
        ).strip()
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
                            and str(
                                node.latest_summary
                                or node.result_text
                                or node.objective
                            ).strip()
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

    @staticmethod
    def _collect_response_media(
        *,
        context: ExecutionContext,
        result: Any = None,
    ) -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()

        def _add_many(values: Any) -> None:
            if not isinstance(values, (list, tuple, set)):
                return
            for value in values:
                rendered = str(value or "").strip()
                if rendered and rendered not in seen:
                    seen.add(rendered)
                    collected.append(rendered)

        state_view = RuntimeState(context)
        _add_many(state_view.collected_media_bucket())

        task_results = getattr(result, "task_results", {}) if result is not None else {}
        if isinstance(task_results, dict):
            for task_result in task_results.values():
                _add_many(getattr(task_result, "artifacts", ()))

        notebook = state_view.notebook_binding().notebook
        if notebook is not None:
            for node in notebook.nodes.values():
                for artifact in getattr(node, "artifacts", ()) or ():
                    rendered = str(getattr(artifact, "path", "") or "").strip()
                    if rendered and rendered not in seen:
                        seen.add(rendered)
                        collected.append(rendered)

        return collected

    def _get_or_create_policy_engine(self) -> PolicyEngine | None:
        """Returns _policy_engine, auto-creating a minimal one for partial test doubles."""
        pe = getattr(self, "_policy_engine", None)
        if pe is not None:
            return pe
        config = getattr(self, "config", None)
        if config is None:
            return None
        policy_store = getattr(self, "_policy_store", None)
        inspect_service = getattr(self, "_inspect_service", None)
        pe = PolicyEngine(
            config=config,
            policy_store=policy_store,
            inspect_service=inspect_service,
        )
        self._policy_engine = pe
        return pe

    def _get_or_create_routing_cascade(self) -> RoutingCascade:
        """Returns _routing_cascade, auto-creating for partial test doubles."""
        rc = getattr(self, "_routing_cascade", None)
        if rc is not None:
            return rc
        rc = RoutingCascade(
            policy_engine=self._get_or_create_policy_engine(),
            policy_store=getattr(self, "_policy_store", None),
            gateway=self.gateway,
            config=self.config,
        )
        self._routing_cascade = rc
        return rc

    def _policy_learning_enabled(self) -> bool:
        pe = self._get_or_create_policy_engine()
        return pe.learning_enabled() if pe is not None else False

    def _routing_enabled(self) -> bool:
        pe = self._get_or_create_policy_engine()
        return pe.routing_enabled() if pe is not None else True

    def _reflection_enabled(self) -> bool:
        pe = self._get_or_create_policy_engine()
        return pe.reflection_enabled() if pe is not None else False

    @staticmethod
    def _build_policy_state_features(
        user_input: str,
        *,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        # Kept for backward compat — delegates to PolicyEngine.build_state_features
        return PolicyEngine.build_state_features(user_input, media_paths=media_paths)

    def _record_policy_decision(self, record: PolicyDecisionRecord) -> None:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            pe.record_decision(record)

    def _record_policy_outcome(self, record: PolicyOutcomeRecord) -> None:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            pe.record_outcome(record)

    def _select_decomposition_action(
        self,
        *,
        user_input: str,
        media_paths: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            return pe.select_decomposition(
                user_input=user_input, media_paths=media_paths
            )
        features = PolicyEngine.build_state_features(
            user_input, media_paths=media_paths
        )
        return "direct", features, ""

    @staticmethod
    def _estimate_independent_subtasks(user_input: str) -> int:
        return PolicyEngine.estimate_independent_subtasks(user_input)

    def _build_scheduling_state_features(
        self,
        user_input: str,
        *,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            return pe.build_scheduling_features(user_input, media_paths=media_paths)
        return PolicyEngine.build_state_features(user_input, media_paths=media_paths)

    def _policy_selector(self) -> ConservativePolicySelector:
        # kept for backward compat — internal callers should prefer _policy_engine
        store = getattr(self, "_policy_store", None)
        if store is None or not hasattr(store, "summarize_action_stats"):
            from .orchestrator_policy_engine import _NULL_POLICY_STORE

            store = _NULL_POLICY_STORE
        return ConservativePolicySelector(
            store,
            min_samples=int(
                getattr(self.config.system, "policy_learning_min_samples", 0) or 0
            ),
            explore_ratio=float(
                getattr(self.config.system, "policy_learning_explore_ratio", -1.0)
                or -1.0
            ),
        )

    def choose_scheduling_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            return pe.choose_scheduling_policy(features=features)
        return {
            "action_name": "serial",
            "hint": "",
            "explain": "no_engine",
            "state_bucket": "default",
        }

    def choose_worker_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            return pe.choose_worker_policy(features=features)
        return {
            "action_name": "allow_worker",
            "hint": "",
            "explain": "no_engine",
            "state_bucket": "default",
        }

    def _persist_policy_events(
        self,
        *,
        flow_id: str,
        chat_key: str,
        events: list[dict[str, Any]],
    ) -> None:
        pe = self._get_or_create_policy_engine()
        if pe is not None:
            pe.persist_policy_events(flow_id=flow_id, chat_key=chat_key, events=events)
        else:
            self._remember_flow_policy_decisions(flow_id, events)

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
        pe = self._get_or_create_policy_engine()
        if pe is None:
            return
        pe.record_routing_telemetry(
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
        pe = self._get_or_create_policy_engine()
        if pe is None:
            return
        await pe.evaluate_task_run_async(
            chat_key=chat_key,
            route_mode=route_mode,
            state_features=state_features,
            routing_decision=routing_decision,
            final_status=final_status,
            outcome=outcome,
        )

    @staticmethod
    def _routing_policy_hints(decision: RoutingDecision | None) -> list[str]:
        return RoutingCascade.routing_policy_hints(decision)

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
        return InspectService.build_debug_policy_summary(
            flow_id=flow_id,
            decomposition_action=decomposition_action,
            decomposition_hint=decomposition_hint,
            route_mode=route_mode,
            router_skip_reason=router_skip_reason,
            routing_decision=routing_decision,
            scheduling_policy=scheduling_policy,
            worker_policy=worker_policy,
        )

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
        return RoutingCascade.format_reflection_hint(payload)

    @staticmethod
    def _select_reflection_override(
        hints: list[dict[str, Any]],
        *,
        allowed_actions: set[str],
    ) -> str:
        return RoutingCascade.select_reflection_override(
            hints, allowed_actions=allowed_actions
        )

    def _maybe_override_policy_from_reflection(
        self,
        payload: dict[str, Any],
        *,
        preferred_action: str,
        hint_prefix: str,
    ) -> dict[str, Any]:
        return (
            self._get_or_create_routing_cascade().maybe_override_policy_from_reflection(
                payload, preferred_action=preferred_action, hint_prefix=hint_prefix
            )
        )

    @staticmethod
    def _maybe_soften_policy_from_guardrail(
        payload: dict[str, Any],
        *,
        soften_default: bool,
        current_default_action: str,
        softened_action: str,
        hint: str,
    ) -> dict[str, Any]:
        return RoutingCascade.maybe_soften_policy_from_guardrail(
            payload,
            soften_default=soften_default,
            current_default_action=current_default_action,
            softened_action=softened_action,
            hint=hint,
        )

    @staticmethod
    def _routing_decision_from_reflection(
        payload: dict[str, Any] | None,
    ) -> RoutingDecision | None:
        return RoutingCascade.routing_decision_from_reflection(payload)

    def _routing_decision_from_intent_cache(
        self,
        payload: dict[str, Any] | None,
        *,
        goal: str,
    ) -> RoutingDecision | None:
        return self._get_or_create_routing_cascade().routing_decision_from_intent_cache(
            payload, goal=goal
        )

    @staticmethod
    def _task_result_outcome_details(result: Any | None) -> dict[str, int]:
        return PolicyEngine.summarize_task_results(result)

    @classmethod
    def _policy_reward(
        cls,
        events: list[dict[str, Any]],
        final_status: str,
        *,
        result: Any | None = None,
    ) -> float:
        return compute_policy_reward(
            events,
            final_status=final_status,
            result=result,
        )

    @classmethod
    def _policy_outcome_details(
        cls,
        events: list[dict[str, Any]],
        *,
        result: Any | None = None,
        error: str | None = None,
        execution_elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        return build_execution_outcome(
            events,
            result=result,
            error=error,
            execution_elapsed_ms=execution_elapsed_ms,
        )

    async def _resolve_routing(
        self,
        *,
        chat_key: str,
        user_input: str,
        media_paths: list[str] | None,
        scheduling_features: dict[str, Any],
        execution_constraints: Any,
        routing_snapshot: Any,
        heartbeat: Any,
    ) -> RoutingResult:
        """Routing cascade — delegates to RoutingCascade."""
        return await self._get_or_create_routing_cascade().resolve_routing(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
            scheduling_features=scheduling_features,
            execution_constraints=execution_constraints,
            routing_snapshot=routing_snapshot,
            heartbeat=heartbeat,
        )

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
        routing = await self._resolve_routing(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
            scheduling_features=scheduling_features,
            execution_constraints=execution_constraints,
            routing_snapshot=routing_snapshot,
            heartbeat=heartbeat,
        )
        routing_decision = routing.routing_decision
        route_mode = routing.route_mode
        scheduling_policy = routing.scheduling_policy
        worker_policy = routing.worker_policy
        reflection_hints_payload = routing.reflection_hints_payload
        resolved_router_model = routing.resolved_router_model
        routing_latency_ms = routing.routing_latency_ms
        router_skip_reason = routing.router_skip_reason
        should_use_model_router = routing.should_use_model_router
        intent_bucket = routing.intent_bucket
        scheduling_overridden = routing.scheduling_overridden
        worker_overridden = routing.worker_overridden
        execution_style_guardrail_reduced = routing.execution_style_guardrail_reduced
        relaxed_reflection_route_payload = routing.relaxed_reflection_route_payload
        guardrail_softened_scheduling = routing.guardrail_softened_scheduling
        guardrail_softened_worker = routing.guardrail_softened_worker
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
        dedup_media = self._collect_response_media(context=context, result=result)

        return text, dedup_media

    def _get_or_create_inspect_service(self) -> InspectService | None:
        """Returns _inspect_service, auto-creating a minimal one for partial test doubles."""
        svc = getattr(self, "_inspect_service", None)
        if svc is not None:
            return svc
        policy_store = getattr(self, "_policy_store", None)
        tape_store = getattr(self, "tape_store", None)
        memory_store = getattr(self, "memory_store", None)
        heartbeat_registry = getattr(self, "_task_heartbeat_registry", None)
        child_task_bus = getattr(self, "_child_task_bus", None)

        class _NullRegistry:
            def snapshot(self, _: str = "") -> dict:
                return {}

        class _NullBus:
            def events_for(self, _: str = "") -> list:
                return []

        class _NullMemoryStore:
            def list_memories(self, chat_id: str = "") -> list:
                return []

        class _NullPolicyStoreFallback:
            def summarize_action_stats(self, *, decision_kind: str, **kw) -> dict:
                return {}

        svc = InspectService(
            policy_store=policy_store or _NullPolicyStoreFallback(),
            tape_store=tape_store,
            memory_store=memory_store or _NullMemoryStore(),
            heartbeat_registry=heartbeat_registry or _NullRegistry(),
            child_task_bus=child_task_bus or _NullBus(),
        )
        # Seed caches from any existing agent attributes (test double compatibility)
        existing_flow_ids = getattr(self, "_recent_flow_ids_by_chat", None)
        if existing_flow_ids:
            svc._recent_flow_ids_by_chat.update(existing_flow_ids)
        existing_flows = getattr(self, "_recent_flows_by_chat", None)
        if existing_flows:
            svc._recent_flows_by_chat.update(existing_flows)
        existing_decisions = getattr(self, "_recent_policy_decisions_by_flow", None)
        if existing_decisions:
            svc._recent_policy_decisions_by_flow.update(existing_decisions)
        self._inspect_service = svc
        # Make agent attributes point to the service's dicts (backward compat shim)
        # so that any code still accessing self._recent_flow_ids_by_chat sees live data.
        self._recent_flow_ids_by_chat = svc._recent_flow_ids_by_chat
        self._recent_flows_by_chat = svc._recent_flows_by_chat
        self._recent_policy_decisions_by_flow = svc._recent_policy_decisions_by_flow
        return svc

    def inspect_runtime_flow(self, flow_id: str = "", chat_key: str = "") -> str:
        svc = self._get_or_create_inspect_service()
        if svc is not None:
            return svc.inspect_runtime_flow(flow_id=flow_id, chat_key=chat_key)
        return "暂无可观测的 flow。"

    def inspect_chat_context(self, chat_key: str, query: str = "") -> str:
        svc = self._get_or_create_inspect_service()
        if svc is not None:
            return svc.inspect_chat_context(chat_key=chat_key, query=query)
        return "缺少 chat_key。"

    def inspect_policy(self, chat_key: str = "", decision_kind: str = "") -> str:
        svc = self._get_or_create_inspect_service()
        if svc is not None:
            return svc.inspect_policy(chat_key=chat_key, decision_kind=decision_kind)
        return "[Policy]\n- no_stats"

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
        logger.info("Loading tape for chat_key=%s", chat_key)
        tape = self._get_runtime_support().prepare_tape(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
        )
        if tape is not None:
            logger.info("Tape loaded, observing user message...")
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
        return self._get_runtime_support().create_runtime_job(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
            job_metadata_override=job_metadata_override,
        )

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
        return self._get_runtime_support().build_runtime_event_recorder(
            chat_key=chat_key,
            tape=tape,
            runtime_job=runtime_job,
            runtime_event_callback=runtime_event_callback,
        )

    def _record_assistant_reply(
        self,
        *,
        chat_key: str,
        tape: Tape | None,
        text: str,
    ) -> None:
        self._get_runtime_support().record_assistant_reply(
            chat_key=chat_key,
            tape=tape,
            text=text,
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
        """Delegate to HandoffManager (lock-outside-LLM pattern)."""
        hm = getattr(self, "_handoff_manager", None)
        if hm is None:
            # Auto-create for partial test doubles
            config = getattr(self, "config", None)
            threshold = (
                getattr(
                    getattr(config, "system", None), "context_compact_threshold", 4000
                )
                if config
                else 4000
            )
            hm = HandoffManager(
                gateway=getattr(self, "gateway", None),
                tape_store=getattr(self, "tape_store", None),
                memory_store=getattr(self, "memory_store", None),
                compact_threshold=threshold,
            )
            self._handoff_manager = hm
        await hm.maybe_handoff(tape, chat_key)

    def _get_command_dispatch(self) -> CommandDispatch:
        """Returns _command_dispatch, auto-creating for partial test doubles."""
        cd = getattr(self, "_command_dispatch", None)
        if cd is not None:
            return cd
        cd = CommandDispatch(
            inspect_service=self._get_or_create_inspect_service(),
            policy_store=getattr(self, "_policy_store", None),
            job_store=getattr(self, "_runtime_job_store", None),
            process_task_fn=self.process_task,
            runtime_support_fn=self._get_runtime_support,
            interactive_support_fn=self._get_interactive_support,
            response_factory=TaskResponse,
        )
        self._command_dispatch = cd
        return cd

    @staticmethod
    def _parse_interactive_session_command(
        user_input: str,
    ) -> dict[str, str] | None:
        return CommandDispatch.parse_interactive_command(user_input)

    @staticmethod
    def _parse_policy_feedback_command(
        user_input: str,
    ) -> dict[str, str] | None:
        return CommandDispatch.parse_policy_command(user_input)

    @staticmethod
    def _parse_job_command(
        user_input: str,
    ) -> dict[str, str] | None:
        return CommandDispatch.parse_job_command(user_input)

    @staticmethod
    def _runtime_event_payload(event: Any) -> dict[str, Any]:
        return OrchestratorRuntimeSupport.runtime_event_payload(event)

    @staticmethod
    def _job_state_from_runtime_event(event_payload: dict[str, Any]) -> tuple[str, str]:
        return OrchestratorRuntimeSupport.job_state_from_runtime_event(event_payload)

    def _resolve_job_target(
        self,
        *,
        chat_key: str,
        target: str,
    ) -> tuple[Any, str]:
        return self._get_runtime_support().resolve_job_target(
            chat_key=chat_key,
            target=target,
        )

    def _runtime_maintenance_report(self) -> str:
        recent_flows: dict[str, Any] = {}
        svc = getattr(self, "_inspect_service", None)
        if svc is not None:
            recent_flows = dict(svc._recent_flows_by_chat)
        return self._get_runtime_support().runtime_maintenance_report(
            recent_flows_by_chat=recent_flows,
            interactive_sessions=getattr(self, "_interactive_sessions", None),
        )

    def _resolve_policy_feedback_flow_id(
        self,
        *,
        chat_key: str,
        target: str,
    ) -> tuple[str, str]:
        return self._get_command_dispatch()._resolve_policy_feedback_flow_id(
            chat_key=chat_key, target=target
        )

    async def _handle_policy_feedback_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> TaskResponse:
        return await self._get_command_dispatch().handle_policy_command(
            chat_key, control
        )

    async def _handle_job_command(
        self,
        chat_key: str,
        control: dict[str, str],
    ) -> TaskResponse:
        return await self._get_command_dispatch().handle_job_command(chat_key, control)

    async def _handle_interactive_session_command(
        self, chat_key: str, control: dict[str, str]
    ) -> TaskResponse:
        return await self._get_command_dispatch().handle_interactive_command(
            chat_key, control
        )

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
        return await self._get_command_dispatch().handle_interactive_message(
            chat_key,
            user_input,
            media_paths=media_paths,
            heartbeat=heartbeat,
            runtime_event_callback=runtime_event_callback,
            interactive_output_callback=interactive_output_callback,
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
