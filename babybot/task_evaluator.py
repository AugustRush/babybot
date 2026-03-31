from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReflectionRecord:
    chat_key: str
    route_mode: str
    state_features: dict[str, Any]
    failure_pattern: str
    recommended_action: str
    confidence: float


@dataclass(frozen=True)
class TaskEvaluationInput:
    chat_key: str
    route_mode: str
    state_features: dict[str, Any] = field(default_factory=dict)
    execution_style: str = ""
    parallelism_hint: str = ""
    worker_hint: str = ""
    final_status: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)


class TaskEvaluator:
    def __init__(self, store: Any) -> None:
        self._store = store

    def evaluate(self, evaluation: TaskEvaluationInput) -> ReflectionRecord | None:
        reflection = self._build_reflection(evaluation)
        if reflection is None:
            return None
        self._record_reflection(reflection)
        for extra in self._build_additional_reflections(evaluation, primary=reflection):
            self._record_reflection(extra)
        return reflection

    def _record_reflection(self, reflection: ReflectionRecord) -> None:
        self._store.record_reflection(
            chat_key=reflection.chat_key,
            route_mode=reflection.route_mode,
            state_features=reflection.state_features,
            failure_pattern=reflection.failure_pattern,
            recommended_action=reflection.recommended_action,
            confidence=reflection.confidence,
        )

    @staticmethod
    def _build_additional_reflections(
        evaluation: TaskEvaluationInput,
        *,
        primary: ReflectionRecord,
    ) -> list[ReflectionRecord]:
        if primary.failure_pattern != "clean_success":
            return []
        payloads: list[ReflectionRecord] = []
        state_features = dict(evaluation.state_features or {})
        parallelism_action = str(evaluation.parallelism_hint or "").strip()
        if parallelism_action in {"serial", "bounded_parallel"}:
            payloads.append(
                ReflectionRecord(
                    chat_key=evaluation.chat_key,
                    route_mode=evaluation.route_mode,
                    state_features=state_features,
                    failure_pattern="clean_success",
                    recommended_action=parallelism_action,
                    confidence=0.58,
                )
            )
        worker_hint = str(evaluation.worker_hint or "").strip()
        worker_action = (
            "allow_worker"
            if worker_hint == "allow"
            else "deny_worker"
            if worker_hint == "deny"
            else ""
        )
        if worker_action:
            payloads.append(
                ReflectionRecord(
                    chat_key=evaluation.chat_key,
                    route_mode=evaluation.route_mode,
                    state_features=state_features,
                    failure_pattern="clean_success",
                    recommended_action=worker_action,
                    confidence=0.58,
                )
            )
        return payloads

    @staticmethod
    def _build_reflection(
        evaluation: TaskEvaluationInput,
    ) -> ReflectionRecord | None:
        outcome = dict(evaluation.outcome or {})
        retry_count = int(outcome.get("retry_count", 0) or 0)
        dead_letter_count = int(outcome.get("dead_letter_count", 0) or 0)
        stalled_count = int(outcome.get("stalled_count", 0) or 0)
        task_result_count = int(outcome.get("task_result_count", 0) or 0)
        tool_failure_count = int(outcome.get("tool_failure_count", 0) or 0)
        loop_guard_block_count = int(outcome.get("loop_guard_block_count", 0) or 0)
        max_step_exhausted_count = int(outcome.get("max_step_exhausted_count", 0) or 0)
        final_status = str(evaluation.final_status or "").strip().lower()

        if final_status and final_status != "succeeded":
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="failed_run",
                recommended_action="analyze_first",
                confidence=0.85,
            )
        if dead_letter_count > 0:
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="dead_lettered",
                recommended_action="serial",
                confidence=0.8,
            )
        if stalled_count > 0:
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="stalled",
                recommended_action="serial",
                confidence=0.75,
            )
        if max_step_exhausted_count > 0:
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="max_steps_exhausted",
                recommended_action="analyze_first",
                confidence=0.8,
            )
        if loop_guard_block_count > 0:
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="loop_guard_blocked",
                recommended_action="analyze_first",
                confidence=0.72,
            )
        if tool_failure_count >= 2:
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="tool_errors",
                recommended_action="analyze_first",
                confidence=0.7,
            )
        if retry_count >= 2:
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="retried_too_much",
                recommended_action="analyze_first",
                confidence=0.75,
            )
        if (
            final_status == "succeeded"
            and retry_count <= 0
            and dead_letter_count <= 0
            and stalled_count <= 0
            and task_result_count > 0
        ):
            recommended_action = str(evaluation.execution_style or "").strip()
            if recommended_action not in {
                "direct_execute",
                "analyze_first",
                "retrieve_first",
                "verify_first",
            }:
                recommended_action = "analyze_first"
            return ReflectionRecord(
                chat_key=evaluation.chat_key,
                route_mode=evaluation.route_mode,
                state_features=dict(evaluation.state_features or {}),
                failure_pattern="clean_success",
                recommended_action=recommended_action,
                confidence=0.6,
            )
        return None
