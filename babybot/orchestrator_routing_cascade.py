"""Routing cascade for OrchestratorAgent.

Extracts the multi-stage routing decision logic (reflection → intent cache
→ rule-based → model router) from OrchestratorAgent into a testable,
standalone class.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .orchestration_router import (
    RoutingDecision,
    build_routing_intent_bucket,
    match_rule_based_routing,
    route_mode_to_contract_mode,
    route_mode_to_step_kind,
    route_task,
    should_call_model_router,
)
from .orchestration_routing_result import RoutingResult
from .orchestrator_policy_engine import PolicyEngine
from .task_contract import build_task_contract

if TYPE_CHECKING:
    from .config import Config
    from .model_gateway import OpenAICompatibleGateway
    from .orchestration_policy_store import OrchestrationPolicyStore

logger = logging.getLogger(__name__)


class RoutingCascade:
    """Multi-stage routing decision pipeline.

    Encapsulates the cascade: reflection → intent_cache → rule-based → model
    router, plus the policy selection / guardrail / override logic that
    surrounds it.

    All external dependencies are injected via the constructor so the class
    can be tested in isolation.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine | None,
        policy_store: OrchestrationPolicyStore | None,
        gateway: OpenAICompatibleGateway,
        config: Config,
    ) -> None:
        self._policy_engine = policy_engine
        self._policy_store = policy_store
        self._gateway = gateway
        self._config = config

    # ------------------------------------------------------------------
    # Thin wrappers around PolicyEngine — mirror OrchestratorAgent helpers
    # ------------------------------------------------------------------

    def _routing_enabled(self) -> bool:
        pe = self._policy_engine
        return pe.routing_enabled() if pe is not None else True

    def _reflection_enabled(self) -> bool:
        pe = self._policy_engine
        return pe.reflection_enabled() if pe is not None else False

    def _choose_scheduling_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        pe = self._policy_engine
        if pe is not None:
            return pe.choose_scheduling_policy(features=features)
        return {
            "action_name": "serial",
            "hint": "",
            "explain": "no_engine",
            "state_bucket": "default",
        }

    def _choose_worker_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        pe = self._policy_engine
        if pe is not None:
            return pe.choose_worker_policy(features=features)
        return {
            "action_name": "allow_worker",
            "hint": "",
            "explain": "no_engine",
            "state_bucket": "default",
        }

    def _build_scheduling_state_features(
        self,
        user_input: str,
        *,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        pe = self._policy_engine
        if pe is not None:
            return pe.build_scheduling_features(user_input, media_paths=media_paths)
        return PolicyEngine.build_state_features(user_input, media_paths=media_paths)

    # ------------------------------------------------------------------
    # Static / pure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def routing_policy_hints(decision: RoutingDecision | None) -> list[str]:
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
    def format_reflection_hint(payload: dict[str, Any]) -> str:
        return (
            "历史反思："
            f"曾出现 {str(payload.get('failure_pattern', '') or 'unknown')}，"
            f"下次优先考虑 {str(payload.get('recommended_action', '') or 'safe_action')} "
            f"(置信度 {float(payload.get('confidence', 0.0) or 0.0):.2f})。"
        )

    @staticmethod
    def select_reflection_override(
        hints: list[dict[str, Any]],
        *,
        allowed_actions: set[str],
    ) -> str:
        for payload in hints:
            action = str(payload.get("recommended_action", "") or "").strip()
            if action in allowed_actions:
                return action
        return ""

    def maybe_override_policy_from_reflection(
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
    def maybe_soften_policy_from_guardrail(
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
    def routing_decision_from_reflection(
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

    def routing_decision_from_intent_cache(
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

    # ------------------------------------------------------------------
    # Main routing cascade
    # ------------------------------------------------------------------

    async def resolve_routing(
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
        """Routing cascade: reflection → intent_cache → rule → model router.

        Returns a RoutingResult value object — all routing decisions and
        telemetry metadata consolidated in one place.
        """
        runtime_telemetry_store = self._policy_store
        intent_bucket = build_routing_intent_bucket(
            user_input,
            has_media=bool(media_paths),
        )
        configured_router_model = str(
            getattr(self._config.system, "routing_model_name", "") or ""
        ).strip()
        fallback_router_model = str(
            getattr(getattr(self._config, "model", None), "model_name", "") or ""
        ).strip()
        resolved_router_model = configured_router_model or fallback_router_model
        routing_timeout = float(
            getattr(self._config.system, "routing_timeout", 2.0) or 2.0
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
        routing_decision = None
        execution_style_guardrail_reduced = False
        relaxed_reflection_route_payload = None
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
            routing_decision = self.routing_decision_from_reflection(
                reflection_route_payload
            )
        if (
            routing_decision is None
            and chat_key
            and hasattr(runtime_telemetry_store, "recommend_route_from_intent_bucket")
            and intent_bucket.startswith("other|")
        ):
            routing_decision = self.routing_decision_from_intent_cache(
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
                self._gateway,
                routing_snapshot,
                heartbeat=heartbeat,
                model_name=configured_router_model,
                timeout=routing_timeout,
            )
        routing_latency_ms = (time.perf_counter() - routing_started) * 1000.0

        # Determine route_mode string from routing_decision
        task_contract_probe = build_task_contract(
            user_input=user_input,
            chat_key=chat_key,
            execution_constraints=execution_constraints,
            route_mode_override=(
                route_mode_to_contract_mode(routing_decision.route_mode)
                if routing_decision is not None
                else None
            ),
        )
        route_mode = (
            route_mode_to_step_kind(routing_decision.route_mode)
            if routing_decision is not None
            else ("debate" if task_contract_probe.mode == "debate" else "tool_workflow")
        )

        # Reflection hints
        reflection_hints_payload: list[dict[str, Any]] = []
        if self._reflection_enabled() and hasattr(
            runtime_telemetry_store, "list_reflection_hints"
        ):
            reflection_hints_payload = runtime_telemetry_store.list_reflection_hints(
                route_mode=route_mode,
                state_features=scheduling_features,
                limit=int(getattr(self._config.system, "reflection_max_hints", 3) or 3),
            )

        # Policy selection with guardrail + reflection overrides
        scheduling_policy = self._choose_scheduling_policy(features=scheduling_features)
        worker_policy = self._choose_worker_policy(features=scheduling_features)
        guardrail_softened_scheduling = False
        guardrail_softened_worker = False
        if int(scheduling_features.get("independent_subtasks", 1) or 1) >= 2:
            softened_scheduling_policy = self.maybe_soften_policy_from_guardrail(
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
            softened_worker_policy = self.maybe_soften_policy_from_guardrail(
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
        scheduling_policy = self.maybe_override_policy_from_reflection(
            scheduling_policy,
            preferred_action=self.select_reflection_override(
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
        worker_policy = self.maybe_override_policy_from_reflection(
            worker_policy,
            preferred_action=self.select_reflection_override(
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

        return RoutingResult(
            routing_decision=routing_decision,
            route_mode=route_mode,
            scheduling_policy=scheduling_policy,
            worker_policy=worker_policy,
            reflection_hints_payload=reflection_hints_payload,
            policy_hints=[],  # built by caller after decomposition_hint is known
            resolved_router_model=resolved_router_model,
            routing_latency_ms=routing_latency_ms,
            router_skip_reason=router_skip_reason,
            should_use_model_router=should_use_model_router,
            intent_bucket=intent_bucket,
            scheduling_overridden=scheduling_overridden,
            worker_overridden=worker_overridden,
            execution_style_guardrail_reduced=execution_style_guardrail_reduced,
            relaxed_reflection_route_payload=relaxed_reflection_route_payload,
            guardrail_softened_scheduling=guardrail_softened_scheduling,
            guardrail_softened_worker=guardrail_softened_worker,
            scheduling_features=scheduling_features,
        )
