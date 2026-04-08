"""PolicyEngine — strategy selection, recording, and evaluation.

Single Responsibility: All policy-learning concerns live here.
  - Feature extraction from user input
  - Action selection via ConservativePolicySelector
  - Recording decisions and outcomes to the policy store
  - Background reflection evaluation

Dependencies (injected):
  - config  → system.policy_learning_enabled, system.routing_enabled, etc.
  - policy_store → record_decision, record_outcome, record_feedback, etc.
  - inspect_service → record_policy_decisions (cache update after persist)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .execution_outcome import (
    build_execution_outcome,
    compute_policy_reward,
    summarize_task_results,
)
from .orchestration_policy import ConservativePolicySelector
from .orchestration_policy_types import PolicyDecisionRecord, PolicyOutcomeRecord
from .task_evaluator import TaskEvaluationInput, TaskEvaluator

if TYPE_CHECKING:
    from .orchestration_router import RoutingDecision
    from .orchestrator_inspect import InspectService


class _NullPolicyStore:
    """Fallback used when policy store is absent."""

    @staticmethod
    def summarize_action_stats(
        *,
        decision_kind: str,
        state_bucket: str | None = None,
    ) -> dict[str, dict[str, float | int]]:
        del decision_kind, state_bucket
        return {}


_NULL_POLICY_STORE = _NullPolicyStore()


class PolicyEngine:
    """Owns all policy-learning logic: selection, recording, reflection."""

    def __init__(
        self,
        *,
        config: Any,
        policy_store: Any,
        inspect_service: InspectService | None = None,
    ) -> None:
        self._config = config
        self._policy_store = policy_store
        self._inspect_service = inspect_service  # optional — for cache update

    # ------------------------------------------------------------------ #
    # Config flags                                                          #
    # ------------------------------------------------------------------ #

    def learning_enabled(self) -> bool:
        return bool(getattr(self._config.system, "policy_learning_enabled", False))

    def routing_enabled(self) -> bool:
        return bool(getattr(self._config.system, "routing_enabled", True))

    def reflection_enabled(self) -> bool:
        return bool(getattr(self._config.system, "reflection_enabled", False))

    # ------------------------------------------------------------------ #
    # Feature extraction (pure computation — no I/O)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_state_features(
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

    @staticmethod
    def estimate_independent_subtasks(user_input: str) -> int:
        text = str(user_input or "").strip()
        count = 1
        for token in ("同时", "分别", "并行", "并且"):
            count += text.count(token)
        return max(1, count)

    def build_scheduling_features(
        self,
        user_input: str,
        *,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        features = self.build_state_features(user_input, media_paths=media_paths)
        features["independent_subtasks"] = self.estimate_independent_subtasks(
            user_input
        )
        return features

    # ------------------------------------------------------------------ #
    # Selector factory                                                      #
    # ------------------------------------------------------------------ #

    def _make_selector(self) -> ConservativePolicySelector:
        store = self._policy_store
        if store is None or not hasattr(store, "summarize_action_stats"):
            store = _NULL_POLICY_STORE
        return ConservativePolicySelector(
            store,
            min_samples=int(
                getattr(self._config.system, "policy_learning_min_samples", 0) or 0
            ),
            explore_ratio=float(
                getattr(self._config.system, "policy_learning_explore_ratio", -1.0)
                or -1.0
            ),
        )

    # ------------------------------------------------------------------ #
    # Action selection                                                      #
    # ------------------------------------------------------------------ #

    def select_decomposition(
        self,
        *,
        user_input: str,
        media_paths: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        """Returns (action_name, features, hint)."""
        features = self.build_state_features(user_input, media_paths=media_paths)
        action = self._make_selector().choose_decomposition(features=features)
        return action.name, features, action.hint

    def choose_scheduling_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        selection = self._make_selector().select_scheduling(features=features)
        return {
            "action_name": selection.action.name,
            "hint": selection.action.hint,
            "explain": selection.explain,
            "state_bucket": selection.state_bucket,
        }

    def choose_worker_policy(self, *, features: dict[str, Any]) -> dict[str, Any]:
        if not self.learning_enabled():
            return {
                "action_name": "allow_worker",
                "hint": "",
                "explain": "policy_learning_disabled",
                "state_bucket": "disabled",
            }
        selection = self._make_selector().select_worker_gate(features=features)
        return {
            "action_name": selection.action.name,
            "hint": selection.action.hint,
            "explain": selection.explain,
            "state_bucket": selection.state_bucket,
        }

    # ------------------------------------------------------------------ #
    # Recording                                                             #
    # ------------------------------------------------------------------ #

    def record_decision(self, record: PolicyDecisionRecord) -> None:
        if not self.learning_enabled() or not record.chat_key:
            return
        if self._policy_store is None:
            return
        self._policy_store.record_decision(
            flow_id=record.flow_id,
            chat_key=record.chat_key,
            decision_kind=record.decision_kind,
            action_name=record.action_name,
            state_features=record.state_features,
        )

    def record_outcome(self, record: PolicyOutcomeRecord) -> None:
        if not self.learning_enabled() or not record.chat_key:
            return
        if self._policy_store is None:
            return
        self._policy_store.record_outcome(
            flow_id=record.flow_id,
            chat_key=record.chat_key,
            final_status=record.final_status,
            reward=record.reward,
            outcome=record.outcome,
        )

    def persist_policy_events(
        self,
        *,
        flow_id: str,
        chat_key: str,
        events: list[dict[str, Any]],
    ) -> None:
        """Persist policy_decision events and update inspection caches."""
        if self.learning_enabled() and chat_key:
            for event in events:
                if event.get("event") != "policy_decision":
                    continue
                self.record_decision(
                    PolicyDecisionRecord(
                        flow_id=flow_id,
                        chat_key=chat_key,
                        decision_kind=str(event.get("decision_kind", "") or "").strip(),
                        action_name=str(event.get("action_name", "") or "").strip(),
                        state_features=dict(event.get("state_features") or {}),
                    )
                )
        # Always update the inspection cache so @policy inspect works regardless
        # of whether learning is enabled.
        if self._inspect_service is not None:
            self._inspect_service.record_policy_decisions(flow_id, events)

    def record_feedback(
        self,
        *,
        flow_id: str,
        chat_key: str,
        rating: str,
        reason: str,
    ) -> None:
        if self._policy_store is None:
            return
        self._policy_store.record_feedback(
            flow_id=flow_id,
            chat_key=chat_key,
            rating=rating,
            reason=reason,
        )

    def record_routing_telemetry(
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
        store = self._policy_store
        if not chat_key or not hasattr(store, "record_runtime_telemetry"):
            return
        store.record_runtime_telemetry(
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
                    if self.routing_enabled() and not should_use_model_router
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

    # ------------------------------------------------------------------ #
    # Outcome helpers (pure computation)                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_reward(
        events: list[dict[str, Any]],
        final_status: str,
        *,
        result: Any | None = None,
    ) -> float:
        return compute_policy_reward(events, final_status=final_status, result=result)

    @staticmethod
    def build_outcome_details(
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

    @staticmethod
    def summarize_task_results(result: Any | None) -> dict[str, int]:
        return summarize_task_results(result)

    # ------------------------------------------------------------------ #
    # Reflection evaluation (async background task)                        #
    # ------------------------------------------------------------------ #

    async def evaluate_task_run_async(
        self,
        *,
        chat_key: str,
        route_mode: str,
        state_features: dict[str, Any],
        routing_decision: RoutingDecision | None,
        final_status: str,
        outcome: dict[str, Any],
    ) -> None:
        if not self.reflection_enabled():
            return
        store = self._policy_store
        if not hasattr(store, "record_reflection"):
            return
        TaskEvaluator(store).evaluate(
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
