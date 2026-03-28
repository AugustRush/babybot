from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _bucket_subtasks(value: Any) -> str:
    raw_subtasks = int(value or 1)
    return "3plus" if raw_subtasks >= 3 else str(max(1, raw_subtasks))


def build_policy_state_buckets(features: dict[str, Any]) -> tuple[str, ...]:
    task_shape = str(features.get("task_shape", "") or "unknown").strip() or "unknown"
    has_media = 1 if bool(features.get("has_media")) else 0
    subtasks_bucket = _bucket_subtasks(features.get("independent_subtasks", 1))
    ordered = (
        f"task_shape={task_shape}|has_media={has_media}|subtasks={subtasks_bucket}",
        f"task_shape={task_shape}|has_media={has_media}",
        f"task_shape={task_shape}|subtasks={subtasks_bucket}",
        f"task_shape={task_shape}",
    )
    deduped: list[str] = []
    for item in ordered:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def build_policy_state_bucket(features: dict[str, Any]) -> str:
    return build_policy_state_buckets(features)[0]


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
        self._min_samples_override = int(min_samples)
        self._explore_ratio_override = float(explore_ratio)

    def _effective_min_samples(self, *, decision_kind: str) -> int:
        if self._min_samples_override > 0:
            return self._min_samples_override
        defaults = {
            "decomposition": 6,
            "scheduling": 8,
            "worker": 10,
        }
        return defaults.get(decision_kind, 8)

    def _effective_explore_ratio(
        self,
        *,
        decision_kind: str,
        stats: dict[str, dict[str, float | int]],
    ) -> float:
        if self._explore_ratio_override >= 0.0:
            return min(0.25, max(0.0, self._explore_ratio_override))
        base = {
            "decomposition": 0.06,
            "scheduling": 0.04,
            "worker": 0.02,
        }.get(decision_kind, 0.04)
        if not stats:
            return 0.0
        total_samples = sum(int(item.get("samples", 0) or 0) for item in stats.values())
        avg_failure = sum(float(item.get("failure_rate", 0.0) or 0.0) for item in stats.values()) / max(1, len(stats))
        avg_stall = sum(float(item.get("stalled_rate", 0.0) or 0.0) for item in stats.values()) / max(1, len(stats))
        sample_factor = min(1.0, total_samples / float(self._effective_min_samples(decision_kind=decision_kind) * 4))
        risk_factor = max(0.2, 1.0 - avg_failure - avg_stall)
        return min(0.15, max(0.0, base * sample_factor * risk_factor))

    def choose_decomposition(self, *, features: dict[str, Any]) -> PolicyAction:
        default = self._DECOMPOSITION_ACTIONS["analyze_then_execute"]
        stats = self._eligible_stats(decision_kind="decomposition", features=features)
        if not stats:
            return default
        return self._best_action(
            decision_kind="decomposition",
            stats=stats,
            default=default,
            actions=self._DECOMPOSITION_ACTIONS,
        )

    def choose_scheduling(self, *, features: dict[str, Any]) -> PolicyAction:
        default = self._SCHEDULING_ACTIONS["serial"]
        stats = self._eligible_stats(decision_kind="scheduling", features=features)
        if not stats:
            return default
        return self._best_action(
            decision_kind="scheduling",
            stats=stats,
            default=default,
            actions=self._SCHEDULING_ACTIONS,
        )

    def choose_worker_gate(self, *, features: dict[str, Any]) -> PolicyAction:
        default = self._SCHEDULING_ACTIONS["deny_worker"]
        stats = self._eligible_stats(decision_kind="worker", features=features)
        if not stats:
            return default
        return self._best_action(
            decision_kind="worker",
            stats=stats,
            default=default,
            actions=self._SCHEDULING_ACTIONS,
        )

    def _eligible_stats(
        self,
        *,
        decision_kind: str,
        features: dict[str, Any],
    ) -> dict[str, dict[str, float | int]]:
        for bucket in build_policy_state_buckets(features):
            raw = self._store.summarize_action_stats(
                decision_kind=decision_kind,
                state_bucket=bucket,
            )
            raw = self._normalize_action_stats(decision_kind=decision_kind, raw=raw)
            eligible = {
                name: payload
                for name, payload in raw.items()
                if int(payload.get("samples", 0) or 0)
                >= self._effective_min_samples(decision_kind=decision_kind)
            }
            if eligible:
                return eligible
        raw = self._store.summarize_action_stats(decision_kind=decision_kind)
        raw = self._normalize_action_stats(decision_kind=decision_kind, raw=raw)
        return {
            name: payload
            for name, payload in raw.items()
            if int(payload.get("samples", 0) or 0)
            >= self._effective_min_samples(decision_kind=decision_kind)
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
        feedback_confidence = float(payload.get("feedback_confidence", 1.0) or 0.0)
        return (
            mean_reward
            + (feedback_score * feedback_confidence * 0.05)
            - (failure_rate * 0.5)
            - (retry_rate * 0.2)
            - (dead_letter_rate * 0.35)
            - (stalled_rate * 0.25)
        )

    def _confidence_penalty(
        self,
        *,
        decision_kind: str,
        samples: int,
        effective_samples: float,
        total_samples: int,
        total_effective_samples: float,
        stats: dict[str, dict[str, float | int]],
    ) -> float:
        effective = (
            effective_samples
            if effective_samples > 0
            else float(samples)
        )
        total_effective = (
            total_effective_samples
            if total_effective_samples > 0
            else float(total_samples)
        )
        if effective <= 0 or total_effective <= 0:
            return float("inf")
        explore_ratio = self._effective_explore_ratio(
            decision_kind=decision_kind,
            stats=stats,
        )
        conservatism = max(0.0, 1.0 - explore_ratio)
        return conservatism * math.sqrt(math.log(total_effective + 1.0) / effective)

    def _best_action(
        self,
        *,
        decision_kind: str,
        stats: dict[str, dict[str, float | int]],
        default: PolicyAction,
        actions: dict[str, PolicyAction],
    ) -> PolicyAction:
        total_samples = sum(int(item.get("samples", 0) or 0) for item in stats.values())
        total_effective_samples = sum(
            float(item.get("effective_samples", item.get("samples", 0.0)) or 0.0)
            for item in stats.values()
        )

        def _rank(item: tuple[str, dict[str, float | int]]) -> tuple[float, int]:
            payload = item[1]
            samples = int(payload.get("samples", 0) or 0)
            effective_samples = float(
                payload.get("effective_samples", payload.get("samples", 0.0)) or 0.0
            )
            conservative_score = self._score_payload(payload) - self._confidence_penalty(
                decision_kind=decision_kind,
                samples=samples,
                effective_samples=effective_samples,
                total_samples=total_samples,
                total_effective_samples=total_effective_samples,
                stats=stats,
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
