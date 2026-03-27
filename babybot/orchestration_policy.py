from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def build_policy_state_bucket(features: dict[str, Any]) -> str:
    task_shape = str(features.get("task_shape", "") or "unknown").strip() or "unknown"
    has_media = 1 if bool(features.get("has_media")) else 0
    raw_subtasks = int(features.get("independent_subtasks", 1) or 1)
    subtasks_bucket = "3plus" if raw_subtasks >= 3 else str(max(1, raw_subtasks))
    return (
        f"task_shape={task_shape}"
        f"|has_media={has_media}"
        f"|subtasks={subtasks_bucket}"
    )


@dataclass(frozen=True)
class PolicyAction:
    name: str
    hint: str


class ConservativePolicySelector:
    _DECOMPOSITION_ACTIONS: dict[str, PolicyAction] = {
        "direct_execute": PolicyAction(
            name="direct_execute",
            hint="如果任务非常直接且信息充分，可以直接执行，但仍要避免猜测。",
        ),
        "analyze_then_execute": PolicyAction(
            name="analyze_then_execute",
            hint="优先先分析任务边界、依赖和风险，再逐步执行。",
        ),
        "retrieve_then_execute": PolicyAction(
            name="retrieve_then_execute",
            hint="先补足上下文、文件或外部信息，再开始执行。",
        ),
        "verify_before_finish": PolicyAction(
            name="verify_before_finish",
            hint="在最终回复前，优先做局部或完整验证。",
        ),
    }

    _SCHEDULING_ACTIONS: dict[str, PolicyAction] = {
        "serial": PolicyAction(
            name="serial",
            hint="默认串行执行；只有独立子任务非常明确时才考虑并行。",
        ),
        "bounded_parallel": PolicyAction(
            name="bounded_parallel",
            hint="仅对明确独立的子任务做有限并行，并保留收敛检查。",
        ),
        "allow_worker": PolicyAction(
            name="allow_worker",
            hint="允许创建 worker，但仍应控制深度与数量。",
        ),
        "deny_worker": PolicyAction(
            name="deny_worker",
            hint="优先在当前编排内完成，不要轻易创建额外 worker。",
        ),
    }
    _ACTION_ALIASES: dict[str, dict[str, str]] = {
        "scheduling": {
            "serial_dispatch": "serial",
            "parallel_dispatch": "bounded_parallel",
        }
    }

    def __init__(
        self,
        store: Any,
        *,
        min_samples: int = 8,
        explore_ratio: float = 0.05,
    ) -> None:
        self._store = store
        self._min_samples = max(1, int(min_samples))
        self._explore_ratio = max(0.0, float(explore_ratio))

    def choose_decomposition(self, *, features: dict[str, Any]) -> PolicyAction:
        default = self._DECOMPOSITION_ACTIONS["analyze_then_execute"]
        stats = self._eligible_stats(decision_kind="decomposition", features=features)
        if not stats:
            return default
        return self._best_action(stats, default, self._DECOMPOSITION_ACTIONS)

    def choose_scheduling(self, *, features: dict[str, Any]) -> PolicyAction:
        default = self._SCHEDULING_ACTIONS["serial"]
        stats = self._eligible_stats(decision_kind="scheduling", features=features)
        if not stats:
            return default
        return self._best_action(stats, default, self._SCHEDULING_ACTIONS)

    def choose_worker_gate(self, *, features: dict[str, Any]) -> PolicyAction:
        default = self._SCHEDULING_ACTIONS["deny_worker"]
        stats = self._eligible_stats(decision_kind="worker", features=features)
        if not stats:
            return default
        return self._best_action(stats, default, self._SCHEDULING_ACTIONS)

    def _eligible_stats(
        self,
        *,
        decision_kind: str,
        features: dict[str, Any],
    ) -> dict[str, dict[str, float | int]]:
        bucket = build_policy_state_bucket(features)
        raw = self._store.summarize_action_stats(
            decision_kind=decision_kind,
            state_bucket=bucket,
        )
        raw = self._normalize_action_stats(decision_kind=decision_kind, raw=raw)
        eligible = {
            name: payload
            for name, payload in raw.items()
            if int(payload.get("samples", 0) or 0) >= self._min_samples
        }
        if eligible:
            return eligible
        raw = self._store.summarize_action_stats(decision_kind=decision_kind)
        raw = self._normalize_action_stats(decision_kind=decision_kind, raw=raw)
        return {
            name: payload
            for name, payload in raw.items()
            if int(payload.get("samples", 0) or 0) >= self._min_samples
        }

    @staticmethod
    def _normalize_action_stats(
        *,
        decision_kind: str,
        raw: dict[str, dict[str, float | int]],
    ) -> dict[str, dict[str, float | int]]:
        aliases = ConservativePolicySelector._ACTION_ALIASES.get(decision_kind, {})
        if not aliases:
            return dict(raw)
        normalized: dict[str, dict[str, float | int]] = {}
        for action_name, payload in raw.items():
            canonical_name = aliases.get(action_name, action_name)
            current = normalized.get(canonical_name)
            samples = int(payload.get("samples", 0) or 0)
            mean_reward = float(payload.get("mean_reward", 0.0) or 0.0)
            if current is None:
                normalized[canonical_name] = dict(payload)
                normalized[canonical_name]["samples"] = samples
                normalized[canonical_name]["mean_reward"] = mean_reward
                continue
            current_samples = int(current.get("samples", 0) or 0)
            total_samples = current_samples + samples
            weighted_reward = (float(current.get("mean_reward", 0.0) or 0.0) * current_samples) + (
                mean_reward * samples
            )
            current["samples"] = total_samples
            current["mean_reward"] = (
                weighted_reward / total_samples if total_samples > 0 else 0.0
            )
            for key, value in payload.items():
                if key not in {"samples", "mean_reward"} and key not in current:
                    current[key] = value
        return normalized

    @staticmethod
    def _score_payload(payload: dict[str, float | int]) -> float:
        mean_reward = float(payload.get("mean_reward", 0.0) or 0.0)
        failure_rate = float(payload.get("failure_rate", 0.0) or 0.0)
        retry_rate = float(payload.get("retry_rate", 0.0) or 0.0)
        dead_letter_rate = float(payload.get("dead_letter_rate", 0.0) or 0.0)
        stalled_rate = float(payload.get("stalled_rate", 0.0) or 0.0)
        feedback_score = float(payload.get("feedback_score", 0.0) or 0.0)
        return (
            mean_reward
            + (feedback_score * 0.4)
            - (failure_rate * 0.5)
            - (retry_rate * 0.2)
            - (dead_letter_rate * 0.35)
            - (stalled_rate * 0.25)
        )

    def _confidence_penalty(
        self,
        *,
        samples: int,
        total_samples: int,
    ) -> float:
        if samples <= 0 or total_samples <= 0:
            return float("inf")
        conservatism = max(0.0, 1.0 - self._explore_ratio)
        return conservatism * math.sqrt(math.log(total_samples + 1.0) / samples)

    def _best_action(
        self,
        stats: dict[str, dict[str, float | int]],
        default: PolicyAction,
        actions: dict[str, PolicyAction],
    ) -> PolicyAction:
        total_samples = sum(int(item.get("samples", 0) or 0) for item in stats.values())

        def _rank(item: tuple[str, dict[str, float | int]]) -> tuple[float, int]:
            payload = item[1]
            samples = int(payload.get("samples", 0) or 0)
            conservative_score = self._score_payload(payload) - self._confidence_penalty(
                samples=samples,
                total_samples=total_samples,
            )
            return (
                conservative_score,
                samples,
            )

        ranked = sorted(
            stats.items(),
            key=_rank,
            reverse=True,
        )
        for action_name, _payload in ranked:
            action = actions.get(action_name)
            if action is not None:
                return action
        return default
