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

from .agent_kernel import ExecutionContext
from .agent_kernel.dynamic_orchestrator import (
    DynamicOrchestrator,
    InMemoryChildTaskBus,
)
from .agent_kernel.execution_constraints import (
    build_execution_constraint_hints,
    infer_execution_constraints,
)
from .config import Config
from .context import Tape, TapeStore, _extract_keywords
from .context_views import build_context_view
from .execution_plan import build_execution_plan
from .memory_store import HybridMemoryStore
from .heartbeat import TaskHeartbeatRegistry
from .interactive_sessions import InteractiveSessionManager
from .interactive_sessions.backends import ClaudeInteractiveBackend
from .interactive_sessions.types import InteractiveRequest
from .model_gateway import OpenAICompatibleGateway
from .orchestration_policy import ConservativePolicySelector
from .orchestration_policy_store import OrchestrationPolicyStore
from .orchestration_router import (
    RoutingDecision,
    build_routing_intent_bucket,
    build_routing_snapshot,
    route_mode_to_contract_mode,
    route_mode_to_step_kind,
    route_task,
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
        self._runtime_job_store = RuntimeJobStore(
            self.config.home_dir / "memory" / "jobs.db"
        )
        self._child_task_bus = InMemoryChildTaskBus()
        self._task_heartbeat_registry = TaskHeartbeatRegistry()
        self._handoff_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._recent_flow_ids_by_chat: OrderedDict[str, str] = OrderedDict()
        self._recent_flows_by_chat: OrderedDict[str, list[str]] = OrderedDict()
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
        recent_flows = getattr(self, "_recent_flows_by_chat", None)
        if not isinstance(recent_flows, OrderedDict):
            recent_flows = OrderedDict(recent_flows or {})
            self._recent_flows_by_chat = recent_flows
        history = [item for item in recent_flows.get(chat_key, []) if item != flow_id]
        history.insert(0, flow_id)
        recent_flows[chat_key] = history[:5]
        while len(recent_flows) > self._FLOW_CACHE_LIMIT:
            recent_flows.popitem(last=False)

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

    def _routing_enabled(self) -> bool:
        system = getattr(getattr(self, "config", None), "system", None)
        return bool(getattr(system, "routing_enabled", True))

    def _routing_shadow_eval_enabled(self) -> bool:
        system = getattr(getattr(self, "config", None), "system", None)
        return bool(getattr(system, "routing_shadow_eval_enabled", True))

    def _reflection_enabled(self) -> bool:
        system = getattr(getattr(self, "config", None), "system", None)
        return bool(getattr(system, "reflection_enabled", True))

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
        features["independent_subtasks"] = self._estimate_independent_subtasks(user_input)
        return features

    def _policy_selector(self) -> ConservativePolicySelector:
        store = getattr(self, "_policy_store", None)
        if store is None or not hasattr(store, "summarize_action_stats"):
            class _NullPolicyStore:
                @staticmethod
                def summarize_action_stats(
                    *,
                    decision_kind: str | None = None,
                    state_bucket: str | None = None,
                ) -> dict[str, dict[str, float | int]]:
                    del decision_kind, state_bucket
                    return {}

            store = _NullPolicyStore()
        return ConservativePolicySelector(
            store,
            min_samples=getattr(self.config.system, "policy_learning_min_samples", 8),
            explore_ratio=getattr(
                self.config.system,
                "policy_learning_explore_ratio",
                0.05,
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
        if "insufficient_samples" not in explain and payload.get("state_bucket") != "global_default":
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
        if "insufficient_samples" not in explain and payload.get("state_bucket") != "global_default":
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
            if recommended_action in {
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
            "allow"
            if recommended_action in {"allow", "allow_worker"}
            else "deny"
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
            self._build_scheduling_state_features(goal).get("independent_subtasks", 1) or 1
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

    async def _shadow_evaluate_routing_async(
        self,
        *,
        flow_id: str,
        routing_snapshot: Any,
        actual_decision: RoutingDecision,
        model_name: str,
        timeout: float,
    ) -> None:
        store = getattr(self, "_policy_store", None)
        if (
            store is None
            or not hasattr(store, "record_shadow_routing_eval")
            or actual_decision.decision_source == "model"
        ):
            return
        shadow_decision = await route_task(
            self.gateway,
            routing_snapshot,
            model_name=model_name,
            timeout=max(0.5, min(1.0, float(timeout or 0.0))),
            allow_rule_based=False,
        )
        if shadow_decision is None:
            return
        store.record_shadow_routing_eval(
            flow_id=flow_id,
            agreed=(
                shadow_decision.route_mode == actual_decision.route_mode
                and shadow_decision.execution_style == actual_decision.execution_style
            ),
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
        store = getattr(self, "_policy_store", None)
        if store is None or not hasattr(store, "record_reflection"):
            return
        TaskEvaluator(store).evaluate(
            TaskEvaluationInput(
                chat_key=chat_key,
                route_mode=route_mode,
                state_features=state_features,
                execution_style=(
                    routing_decision.execution_style if routing_decision is not None else ""
                ),
                parallelism_hint=(
                    routing_decision.parallelism_hint if routing_decision is not None else ""
                ),
                worker_hint=(
                    routing_decision.worker_hint if routing_decision is not None else ""
                ),
                final_status=final_status,
                outcome=outcome,
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

    @staticmethod
    def _policy_outcome_details(
        events: list[dict[str, Any]],
        *,
        result: Any | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        retry_count = sum(1 for event in events if event.get("event") == "retrying")
        dead_letter_count = sum(
            1 for event in events if event.get("event") == "dead_lettered"
        )
        stalled_count = sum(1 for event in events if event.get("event") == "stalled")
        payload = {
            "retry_count": retry_count,
            "dead_letter_count": dead_letter_count,
            "stalled_count": stalled_count,
        }
        if result is not None:
            payload["task_result_count"] = len(getattr(result, "task_results", {}) or {})
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
        runtime_job = (
            self._runtime_job_store.latest_for_chat(chat_key)
            if chat_key and getattr(self, "_runtime_job_store", None) is not None
            else None
        )
        routing_snapshot = build_routing_snapshot(
            chat_key=chat_key,
            goal=user_input,
            tape=tape,
            memory_store=self.memory_store if tape else None,
            runtime_job=runtime_job,
            recent_flow_ids=list(getattr(self, "_recent_flows_by_chat", {}).get(chat_key, [])),
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
        if (
            runtime_telemetry_store is not None
            and hasattr(runtime_telemetry_store, "recommend_reflection_guardrails")
        ):
            reflection_guardrails = runtime_telemetry_store.recommend_reflection_guardrails(
                chat_key=chat_key or None
            )
        if (
            self._reflection_enabled()
            and chat_key
            and runtime_telemetry_store is not None
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
                reflection_route_payload = runtime_telemetry_store.recommend_route_from_reflections(
                    chat_key=chat_key,
                    state_features=scheduling_features,
                    min_confidence=0.72,
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
            and runtime_telemetry_store is not None
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
        if (
            runtime_telemetry_store is not None
            and hasattr(runtime_telemetry_store, "recommend_router_timeout")
        ):
            recommendation = runtime_telemetry_store.recommend_router_timeout(
                base_timeout=routing_timeout,
                chat_key=chat_key or None,
                router_model=resolved_router_model,
            )
            routing_timeout = float(
                recommendation.get("timeout_seconds", routing_timeout) or routing_timeout
            )
        routing_started = time.perf_counter()
        if self._routing_enabled() and routing_decision is None:
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
                True if routing_decision is not None and routing_decision.need_clarification else None
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
        if (
            self._reflection_enabled()
            and getattr(self, "_policy_store", None) is not None
            and hasattr(self._policy_store, "list_reflection_hints")
        ):
            reflection_hints_payload = self._policy_store.list_reflection_hints(
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
            guardrail_softened_scheduling = softened_scheduling_policy is not scheduling_policy
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
            guardrail_softened_worker = softened_worker_policy is not worker_policy
            worker_policy = softened_worker_policy
        scheduling_base_action = str(scheduling_policy.get("action_name", "") or "")
        worker_base_action = str(worker_policy.get("action_name", "") or "")
        scheduling_policy = self._maybe_override_policy_from_reflection(
            scheduling_policy,
            preferred_action=self._select_reflection_override(
                (
                    reflection_hints_payload
                    if str(
                        reflection_guardrails.get("parallelism", {}).get("injection_level", "normal")
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
                        reflection_guardrails.get("worker", {}).get("injection_level", "normal")
                    )
                    != "reduced"
                    else []
                ),
                allowed_actions={"allow_worker", "deny_worker"},
            ),
            hint_prefix="历史反思建议 worker 动作：",
        )
        reflection_override_count = 0
        execution_style_reflection_count = 0
        parallelism_reflection_count = 0
        worker_reflection_count = 0
        execution_style_guardrail_reduce_count = 0
        parallelism_guardrail_soften_count = 0
        worker_guardrail_soften_count = 0
        if routing_decision is not None and routing_decision.decision_source == "reflection":
            execution_style_reflection_count = 1
        if (
            execution_style_guardrail_reduced
            and relaxed_reflection_route_payload is not None
            and routing_decision is None
        ):
            execution_style_guardrail_reduce_count = 1
        if str(scheduling_policy.get("action_name", "") or "") != scheduling_base_action:
            reflection_override_count += 1
            parallelism_reflection_count = 1
        if str(worker_policy.get("action_name", "") or "") != worker_base_action:
            reflection_override_count += 1
            worker_reflection_count = 1
        if guardrail_softened_scheduling:
            parallelism_guardrail_soften_count = 1
        if guardrail_softened_worker:
            worker_guardrail_soften_count = 1
        if (
            chat_key
            and runtime_telemetry_store is not None
            and hasattr(runtime_telemetry_store, "record_runtime_telemetry")
        ):
            runtime_telemetry_store.record_runtime_telemetry(
                flow_id=flow_id,
                chat_key=chat_key,
                route_mode=route_mode,
                router_model=resolved_router_model,
                router_latency_ms=routing_latency_ms,
                router_fallback=routing_decision is None,
                router_source=(
                    routing_decision.decision_source if routing_decision is not None else "fallback"
                ),
                execution_style=(
                    str(routing_decision.execution_style or "")
                    if routing_decision is not None
                    else ""
                ),
                intent_bucket=intent_bucket,
                reflection_hint_count=len(reflection_hints_payload),
                reflection_override_count=reflection_override_count,
                execution_style_reflection_count=execution_style_reflection_count,
                parallelism_reflection_count=parallelism_reflection_count,
                worker_reflection_count=worker_reflection_count,
                execution_style_guardrail_reduce_count=execution_style_guardrail_reduce_count,
                parallelism_guardrail_soften_count=parallelism_guardrail_soften_count,
                worker_guardrail_soften_count=worker_guardrail_soften_count,
            )
        if (
            routing_decision is not None
            and routing_decision.decision_source != "model"
            and self._routing_enabled()
            and self._routing_shadow_eval_enabled()
        ):
            self._spawn_background_task(
                self._shadow_evaluate_routing_async(
                    flow_id=flow_id,
                    routing_snapshot=routing_snapshot,
                    actual_decision=routing_decision,
                    model_name=configured_router_model,
                    timeout=min(routing_timeout, 1.0),
                ),
                label="routing-shadow-eval",
            )
        execution_plan = build_execution_plan(task_contract)
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
            )
        if chat_key and self._policy_learning_enabled():
            worker_action_name = str(worker_policy.get("action_name", "") or "").strip()
            if worker_action_name:
                context.emit(
                    "policy_decision",
                    decision_kind="worker",
                    action_name=worker_action_name,
                    state_features=scheduling_features,
                )

        logger.info("DynamicOrchestrator created, starting run flow_id=%s", flow_id)
        try:
            if heartbeat is not None:
                result = await heartbeat.watch(
                    orchestrator.run(goal=task_contract.goal, context=context),
                )
            else:
                result = await orchestrator.run(goal=task_contract.goal, context=context)
        except Exception as exc:
            self._persist_policy_events(
                flow_id=flow_id,
                chat_key=chat_key,
                events=list(context.events),
            )
            failed_outcome = self._policy_outcome_details(
                context.events,
                error=str(exc),
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
        )
        self._record_policy_outcome(
            PolicyOutcomeRecord(
                flow_id=flow_id,
                chat_key=chat_key,
                final_status="succeeded",
                reward=self._policy_reward(context.events, "succeeded"),
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
                    float(item[1].get("effective_samples", item[1].get("samples", 0.0)) or 0.0),
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
                    + f"feedback_score={float(payload.get('feedback_score', 0.0) or 0.0):.2f}"
                )
        telemetry_summary_fn = getattr(self._policy_store, "summarize_runtime_telemetry", None)
        if callable(telemetry_summary_fn):
            try:
                telemetry = telemetry_summary_fn(chat_key=chat_key or None)
            except TypeError:
                telemetry = telemetry_summary_fn()
            overall = telemetry.get("overall") if isinstance(telemetry, dict) else None
            by_route_mode = (
                telemetry.get("by_route_mode", {}) if isinstance(telemetry, dict) else {}
            )
            if isinstance(overall, dict) and int(overall.get("runs", 0) or 0) > 0:
                parts.append("[Routing Telemetry]")
                parts.append(
                    "- "
                    + f"runs={int(overall.get('runs', 0) or 0)} "
                    + f"avg_router_latency_ms={float(overall.get('avg_router_latency_ms', 0.0) or 0.0):.2f} "
                    + f"fallback_rate={float(overall.get('fallback_rate', 0.0) or 0.0):.2f} "
                    + f"rule_hit_rate={float(overall.get('rule_hit_rate', 0.0) or 0.0):.2f} "
                    + f"reflection_route_rate={float(overall.get('reflection_route_rate', 0.0) or 0.0):.2f} "
                    + f"reflection_match_rate={float(overall.get('reflection_match_rate', 0.0) or 0.0):.2f} "
                    + f"reflection_override_rate={float(overall.get('reflection_override_rate', 0.0) or 0.0):.2f} "
                    + f"execution_style_reflection_rate={float(overall.get('execution_style_reflection_rate', 0.0) or 0.0):.2f} "
                    + f"parallelism_reflection_rate={float(overall.get('parallelism_reflection_rate', 0.0) or 0.0):.2f} "
                    + f"worker_reflection_rate={float(overall.get('worker_reflection_rate', 0.0) or 0.0):.2f} "
                    + f"execution_style_guardrail_reduce_rate={float(overall.get('execution_style_guardrail_reduce_rate', 0.0) or 0.0):.2f} "
                    + f"parallelism_guardrail_soften_rate={float(overall.get('parallelism_guardrail_soften_rate', 0.0) or 0.0):.2f} "
                    + f"worker_guardrail_soften_rate={float(overall.get('worker_guardrail_soften_rate', 0.0) or 0.0):.2f} "
                    + f"shadow_routing_eval_rate={float(overall.get('shadow_routing_eval_rate', 0.0) or 0.0):.2f} "
                    + f"shadow_routing_agreement_rate={float(overall.get('shadow_routing_agreement_rate', 0.0) or 0.0):.2f} "
                    + f"mean_reward={float(overall.get('mean_reward', 0.0) or 0.0):.2f}"
                )
                if isinstance(by_route_mode, dict):
                    for route_mode, payload in sorted(by_route_mode.items()):
                        if not isinstance(payload, dict):
                            continue
                        parts.append(
                            "- "
                            + f"route_mode={route_mode} "
                            + f"runs={int(payload.get('runs', 0) or 0)} "
                            + f"avg_router_latency_ms={float(payload.get('avg_router_latency_ms', 0.0) or 0.0):.2f} "
                            + f"fallback_rate={float(payload.get('fallback_rate', 0.0) or 0.0):.2f} "
                            + f"rule_hit_rate={float(payload.get('rule_hit_rate', 0.0) or 0.0):.2f} "
                            + f"reflection_route_rate={float(payload.get('reflection_route_rate', 0.0) or 0.0):.2f} "
                            + f"reflection_match_rate={float(payload.get('reflection_match_rate', 0.0) or 0.0):.2f} "
                            + f"reflection_override_rate={float(payload.get('reflection_override_rate', 0.0) or 0.0):.2f} "
                            + f"execution_style_reflection_rate={float(payload.get('execution_style_reflection_rate', 0.0) or 0.0):.2f} "
                            + f"parallelism_reflection_rate={float(payload.get('parallelism_reflection_rate', 0.0) or 0.0):.2f} "
                            + f"worker_reflection_rate={float(payload.get('worker_reflection_rate', 0.0) or 0.0):.2f} "
                            + f"execution_style_guardrail_reduce_rate={float(payload.get('execution_style_guardrail_reduce_rate', 0.0) or 0.0):.2f} "
                            + f"parallelism_guardrail_soften_rate={float(payload.get('parallelism_guardrail_soften_rate', 0.0) or 0.0):.2f} "
                            + f"worker_guardrail_soften_rate={float(payload.get('worker_guardrail_soften_rate', 0.0) or 0.0):.2f} "
                            + f"shadow_routing_eval_rate={float(payload.get('shadow_routing_eval_rate', 0.0) or 0.0):.2f} "
                            + f"shadow_routing_agreement_rate={float(payload.get('shadow_routing_agreement_rate', 0.0) or 0.0):.2f} "
                            + f"mean_reward={float(payload.get('mean_reward', 0.0) or 0.0):.2f}"
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
            if runtime_event_callback is not None:
                maybe = runtime_event_callback(payload)
                if inspect.isawaitable(maybe):
                    await maybe

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
        asst_entry = tape.append(
            "message", {"role": "assistant", "content": text}
        )
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
                return await self._handle_interactive_session_command(
                    chat_key, control
                )
            if self._interactive_sessions.has_active_session(chat_key):
                reply = await self._handle_interactive_session_message(
                    chat_key,
                    user_input,
                    media_paths=media_paths,
                    heartbeat=heartbeat,
                    runtime_event_callback=runtime_event_callback,
                )
                if reply is not None:
                    return reply

        tape = self._prepare_tape(
            chat_key=chat_key,
            user_input=user_input,
            media_paths=media_paths,
        )
        runtime_job_store = getattr(self, "_runtime_job_store", None)
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
            if runtime_job_store is not None and runtime_job is not None:
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
            if runtime_job_store is not None and runtime_job is not None:
                runtime_job_store.transition(
                    runtime_job.job_id,
                    "completed",
                    progress_message="执行完成",
                    result_text=text,
                )

            return TaskResponse(text=text, media_paths=collected_media)
        except Exception as exc:
            if runtime_job_store is not None and runtime_job is not None:
                runtime_job_store.transition(
                    runtime_job.job_id,
                    "failed",
                    progress_message="执行失败",
                    error=str(exc),
                )
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
        store = getattr(self, "_runtime_job_store", None)
        if store is None:
            return None, "当前未启用作业存储。"
        normalized_target = str(target or "latest").strip() or "latest"
        job = (
            store.latest_for_chat(chat_key)
            if normalized_target == "latest"
            else store.get(normalized_target)
        )
        if job is None:
            return None, "未找到对应作业。"
        return job, ""

    def _runtime_maintenance_report(self) -> str:
        store = getattr(self, "_runtime_job_store", None)
        if store is None:
            return "当前未启用作业存储。"
        report = store.run_maintenance(retention_seconds=0)
        interactive_manager = getattr(self, "_interactive_sessions", None)
        stale_sessions = (
            int(interactive_manager.cleanup()) if interactive_manager is not None else 0
        )
        recent_flows = getattr(self, "_recent_flows_by_chat", {}) or {}
        known_flow_ids = store.known_flow_ids()
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
            lines.append(f"jobs={', '.join(str(item) for item in orphaned_job_ids[:10])}")
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
        if normalized_target and normalized_target not in {"latest"} and store is not None:
            target_job = store.get(normalized_target)
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
            return TaskResponse(text="支持的命令：@policy feedback <flow_id|latest> good|bad <reason> / @policy inspect [decision_kind|flow_id]")
        rating = str(control.get("rating", "") or "").strip().lower()
        reason = str(control.get("reason", "") or "").strip()
        if rating not in {"good", "bad"} or not reason:
            return TaskResponse(text="用法：@policy feedback <flow_id|latest> good|bad <reason>")
        if not chat_key:
            return TaskResponse(text="缺少 chat_key，无法记录策略反馈。")
        flow_id, error = self._resolve_policy_feedback_flow_id(
            chat_key=chat_key,
            target=str(control.get("target", "") or "").strip(),
        )
        if error:
            return TaskResponse(text=error)
        self._policy_store.record_feedback(
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
            return TaskResponse(text="支持的命令：@job status <job_id|latest> / @job resume <job_id|latest> / @job cleanup")
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
            store = getattr(self, "_runtime_job_store", None)
            if store is not None:
                store.transition(
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
            return TaskResponse(
                text=(
                    f"当前交互会话：{status.backend_name} "
                    f"(session_id={status.session_id})"
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
        if runtime_event_callback is not None:
            maybe = runtime_event_callback(
                {
                    "event": "running",
                    "task_id": "interactive_session",
                    "flow_id": f"interactive:{chat_key}",
                    "payload": {
                        "stage": "interactive_session",
                        "state": "running",
                        "message": "交互会话处理中",
                    },
                }
            )
            if inspect.isawaitable(maybe):
                await maybe
        if heartbeat is not None:
            async with heartbeat.keep_alive():
                reply = await self._interactive_sessions.send(chat_key, request)
        else:
            reply = await self._interactive_sessions.send(chat_key, request)
        if reply.expired:
            return None
        tape = self.tape_store.get_or_create(chat_key) if chat_key and self.tape_store else None
        if tape is not None:
            pending_entries = []
            if tape.last_anchor() is None:
                pending_entries.append(
                    tape.append("anchor", {"name": "session/start", "state": {}})
                )
            content_for_tape = user_input
            if media_paths:
                content_for_tape = f"{user_input}\n[附带 {len(media_paths)} 张图片]"
            pending_entries.append(
                tape.append("message", {"role": "user", "content": content_for_tape})
            )
            pending_entries.append(
                tape.append("message", {"role": "assistant", "content": reply.text})
            )
            self.tape_store.save_entries(chat_key, pending_entries)
            if hasattr(self, "memory_store") and self.memory_store is not None:
                self.memory_store.observe_user_message(chat_key, user_input)
        if runtime_event_callback is not None:
            maybe = runtime_event_callback(
                {
                    "event": "completed",
                    "task_id": "interactive_session",
                    "flow_id": f"interactive:{chat_key}",
                    "payload": {
                        "stage": "interactive_session",
                        "state": "completed",
                        "message": "交互会话完成",
                    },
                }
            )
            if inspect.isawaitable(maybe):
                await maybe
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
