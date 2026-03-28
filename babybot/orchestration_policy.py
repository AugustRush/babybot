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


@dataclass(frozen=True)
class PolicySelection:
    action: PolicyAction
    decision_kind: str
    state_bucket: str
    score: float
    explain: str


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

    @staticmethod
    def _sample_mass(payload: dict[str, float | int]) -> float:
        effective = float(payload.get("effective_samples", 0.0) or 0.0)
        if effective > 0:
            return effective
        return float(payload.get("samples", 0) or 0)

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
        return self.select_decomposition(features=features).action

    def select_decomposition(self, *, features: dict[str, Any]) -> PolicySelection:
        default = self._DECOMPOSITION_ACTIONS["analyze_then_execute"]
        return self._select_action(
            decision_kind="decomposition",
            features=features,
            default=default,
            actions=self._DECOMPOSITION_ACTIONS,
        )

    def choose_scheduling(self, *, features: dict[str, Any]) -> PolicyAction:
        return self.select_scheduling(features=features).action

    def select_scheduling(self, *, features: dict[str, Any]) -> PolicySelection:
        default = self._SCHEDULING_ACTIONS["serial"]
        return self._select_action(
            decision_kind="scheduling",
            features=features,
            default=default,
            actions=self._SCHEDULING_ACTIONS,
        )

    def choose_worker_gate(self, *, features: dict[str, Any]) -> PolicyAction:
        return self.select_worker_gate(features=features).action

    def select_worker_gate(self, *, features: dict[str, Any]) -> PolicySelection:
        default = self._SCHEDULING_ACTIONS["deny_worker"]
        return self._select_action(
            decision_kind="worker",
            features=features,
            default=default,
            actions=self._SCHEDULING_ACTIONS,
        )

    def _select_action(
        self,
        *,
        decision_kind: str,
        features: dict[str, Any],
        default: PolicyAction,
        actions: dict[str, PolicyAction],
    ) -> PolicySelection:
        bucket, stats = self._eligible_stats(decision_kind=decision_kind, features=features)
        if not stats:
            return PolicySelection(
                action=default,
                decision_kind=decision_kind,
                state_bucket="global_default",
                score=0.0,
                explain="insufficient_samples; using safe default",
            )
        return self._best_action(
            decision_kind=decision_kind,
            state_bucket=bucket,
            stats=stats,
            default=default,
            actions=actions,
        )

    def _eligible_stats(
        self,
        *,
        decision_kind: str,
        features: dict[str, Any],
    ) -> tuple[str, dict[str, dict[str, float | int]]]:
        local_candidates: list[tuple[str, dict[str, dict[str, float | int]]]] = []
        min_samples = self._effective_min_samples(decision_kind=decision_kind)
        for bucket in build_policy_state_buckets(features):
            raw = self._store.summarize_action_stats(
                decision_kind=decision_kind,
                state_bucket=bucket,
            )
            raw = self._normalize_action_stats(decision_kind=decision_kind, raw=raw)
            eligible = {
                name: payload
                for name, payload in raw.items()
                if self._sample_mass(payload)
                >= min_samples
            }
            if eligible:
                local_candidates.append((bucket, eligible))
        if local_candidates:
            return max(
                local_candidates,
                key=lambda item: self._template_quality(item[1]),
            )
        raw = self._store.summarize_action_stats(decision_kind=decision_kind)
        raw = self._normalize_action_stats(decision_kind=decision_kind, raw=raw)
        global_stats = {
            name: payload
            for name, payload in raw.items()
            if self._sample_mass(payload)
            >= min_samples
        }
        if not global_stats:
            return "global_default", {}
        return "global", global_stats

    def _template_quality(
        self,
        stats: dict[str, dict[str, float | int]],
    ) -> tuple[float, float]:
        ranked_scores = sorted(
            (self._score_payload(payload) for payload in stats.values()),
            reverse=True,
        )
        top_score = ranked_scores[0] if ranked_scores else -1.0
        second_score = ranked_scores[1] if len(ranked_scores) > 1 else top_score - 0.05
        separation = top_score - second_score
        support = sum(self._sample_mass(payload) for payload in stats.values())
        return (
            separation,
            support,
        )

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
        drift_score = float(payload.get("drift_score", 0.0) or 0.0)
        return (
            mean_reward
            + (feedback_score * feedback_confidence * 0.05)
            - (failure_rate * 0.5)
            - (retry_rate * 0.2)
            - (dead_letter_rate * 0.35)
            - (stalled_rate * 0.25)
            - (drift_score * 0.35)
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
        state_bucket: str,
        stats: dict[str, dict[str, float | int]],
        default: PolicyAction,
        actions: dict[str, PolicyAction],
    ) -> PolicySelection:
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

        safe_items = [
            item
            for item in stats.items()
            if not self._is_recently_risky(item[1])
        ]
        ranked = sorted((safe_items or list(stats.items())), key=_rank, reverse=True)
        for action_name, payload in ranked:
            action = actions.get(action_name)
            if action is not None:
                score = _rank((action_name, payload))[0]
                return PolicySelection(
                    action=action,
                    decision_kind=decision_kind,
                    state_bucket=state_bucket,
                    score=score,
                    explain=self._build_explain(
                        action_name=action_name,
                        payload=payload,
                        state_bucket=state_bucket,
                        score=score,
                    ),
                )
        return PolicySelection(
            action=default,
            decision_kind=decision_kind,
            state_bucket=state_bucket,
            score=0.0,
            explain="no_action_match; using default",
        )

    @staticmethod
    def _build_explain(
        *,
        action_name: str,
        payload: dict[str, float | int],
        state_bucket: str,
        score: float,
    ) -> str:
        return (
            f"selected={action_name}; "
            f"bucket={state_bucket}; "
            f"score={score:.3f}; "
            f"mean_reward={float(payload.get('mean_reward', 0.0) or 0.0):.3f}; "
            f"recent_mean_reward={float(payload.get('recent_mean_reward', 0.0) or 0.0):.3f}; "
            f"effective_samples={float(payload.get('effective_samples', payload.get('samples', 0.0)) or 0.0):.3f}; "
            f"failure_rate={float(payload.get('failure_rate', 0.0) or 0.0):.3f}; "
            f"drift_score={float(payload.get('drift_score', 0.0) or 0.0):.3f}; "
            f"recent_failure_rate={float(payload.get('recent_failure_rate', 0.0) or 0.0):.3f}"
        )

    @staticmethod
    def _is_recently_risky(payload: dict[str, float | int]) -> bool:
        recent_guard_samples = float(payload.get("recent_guard_samples", 0.0) or 0.0)
        if recent_guard_samples >= 1.0:
            recent_failure_rate = float(payload.get("recent_failure_rate", 0.0) or 0.0)
            recent_bad_feedback_rate = float(
                payload.get("recent_bad_feedback_rate", 0.0) or 0.0
            )
            if recent_failure_rate >= 0.5 or recent_bad_feedback_rate >= 0.6:
                return True
        drift_score = float(payload.get("drift_score", 0.0) or 0.0)
        recent_mean_reward = float(payload.get("recent_mean_reward", 0.0) or 0.0)
        return drift_score >= 0.5 and recent_mean_reward <= 0.3
