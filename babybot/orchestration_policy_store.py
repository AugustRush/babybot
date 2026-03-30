from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .orchestration_policy import build_policy_state_buckets


class OrchestrationPolicyStore:
    _OUTCOME_HALF_LIFE_DAYS = 21.0
    _FEEDBACK_HALF_LIFE_DAYS = 30.0
    _REFLECTION_ROUTE_HALF_LIFE_DAYS = 14.0
    _RECENT_WINDOW_DAYS = 7.0

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 3000) -> None:
        self._db_path = Path(db_path)
        self._busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._db: sqlite3.Connection | None = None

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def pragma(self, name: str) -> str:
        row = self._ensure_db().execute(f"PRAGMA {name}").fetchone()
        if row is None:
            return ""
        return str(row[0])

    def record_decision(
        self,
        *,
        flow_id: str,
        chat_key: str,
        decision_kind: str,
        action_name: str,
        state_features: dict[str, Any],
        created_at: str | None = None,
    ) -> None:
        db = self._ensure_db()
        db.execute(
            """
            INSERT INTO policy_decisions (
                flow_id, chat_key, decision_kind, action_name, state_features_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                chat_key,
                decision_kind,
                action_name,
                json.dumps(state_features, ensure_ascii=False, sort_keys=True),
                created_at or self._utc_now(),
            ),
        )
        db.commit()

    def record_outcome(
        self,
        *,
        flow_id: str,
        chat_key: str,
        final_status: str,
        reward: float,
        outcome: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        db = self._ensure_db()
        db.execute(
            """
            INSERT INTO policy_outcomes (
                flow_id, chat_key, final_status, reward, outcome_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                chat_key,
                final_status,
                float(reward),
                json.dumps(outcome or {}, ensure_ascii=False, sort_keys=True),
                created_at or self._utc_now(),
            ),
        )
        db.commit()

    def record_feedback(
        self,
        *,
        flow_id: str,
        chat_key: str,
        rating: str,
        reason: str,
        created_at: str | None = None,
    ) -> None:
        db = self._ensure_db()
        db.execute(
            """
            INSERT INTO policy_feedback (
                flow_id, chat_key, rating, reason, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                chat_key,
                rating,
                reason,
                created_at or self._utc_now(),
            ),
        )
        db.commit()

    def record_reflection(
        self,
        *,
        chat_key: str,
        route_mode: str,
        state_features: dict[str, Any],
        failure_pattern: str,
        recommended_action: str,
        confidence: float,
        created_at: str | None = None,
    ) -> None:
        db = self._ensure_db()
        db.execute(
            """
            INSERT INTO policy_reflections (
                chat_key, route_mode, state_bucket, state_features_json,
                failure_pattern, recommended_action, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_key,
                route_mode,
                build_policy_state_buckets(state_features)[0],
                json.dumps(state_features, ensure_ascii=False, sort_keys=True),
                failure_pattern,
                recommended_action,
                max(0.0, min(1.0, float(confidence))),
                created_at or self._utc_now(),
            ),
        )
        db.commit()

    def list_reflection_hints(
        self,
        *,
        route_mode: str,
        state_features: dict[str, Any],
        limit: int = 3,
        chat_key: str | None = None,
    ) -> list[dict[str, Any]]:
        buckets = build_policy_state_buckets(state_features)
        rows = self._ensure_db().execute(
            """
            SELECT chat_key, route_mode, state_bucket, state_features_json,
                   failure_pattern, recommended_action, confidence, created_at
            FROM policy_reflections
            WHERE route_mode=?
            """,
            (route_mode,),
        ).fetchall()
        ranked: list[tuple[tuple[float, float, str], dict[str, Any]]] = []
        for row in rows:
            if chat_key and str(row["chat_key"] or "").strip() != str(chat_key).strip():
                continue
            state_bucket = str(row["state_bucket"] or "")
            if state_bucket not in buckets:
                continue
            payload = {
                "chat_key": str(row["chat_key"] or ""),
                "route_mode": str(row["route_mode"] or ""),
                "state_bucket": state_bucket,
                "state_features": self._decode_json_object(row["state_features_json"]),
                "failure_pattern": str(row["failure_pattern"] or ""),
                "recommended_action": str(row["recommended_action"] or ""),
                "confidence": float(row["confidence"] or 0.0),
                "created_at": str(row["created_at"] or ""),
            }
            ranked.append(
                (
                    (
                        float(len(buckets) - buckets.index(state_bucket)),
                        payload["confidence"],
                        payload["created_at"],
                    ),
                    payload,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in ranked[: max(0, int(limit))]]

    def recommend_route_from_reflections(
        self,
        *,
        state_features: dict[str, Any],
        chat_key: str | None = None,
        min_confidence: float = 0.55,
        min_samples: int = 2,
        limit: int = 12,
    ) -> dict[str, Any] | None:
        buckets = build_policy_state_buckets(state_features)
        rows = self._ensure_db().execute(
            """
            SELECT chat_key, route_mode, state_bucket, recommended_action,
                   confidence, created_at, failure_pattern
            FROM policy_reflections
            WHERE failure_pattern='clean_success'
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if chat_key and str(row["chat_key"] or "").strip() != str(chat_key).strip():
                continue
            state_bucket = str(row["state_bucket"] or "")
            if state_bucket not in buckets:
                continue
            confidence = max(0.0, float(row["confidence"] or 0.0))
            if confidence < float(min_confidence):
                continue
            created_at = self._parse_time(row["created_at"])
            weight = self._decay_weight(
                created_at,
                now=datetime.now(timezone.utc),
                half_life_days=self._REFLECTION_ROUTE_HALF_LIFE_DAYS,
            )
            route_mode = str(row["route_mode"] or "").strip()
            recommended_action = str(row["recommended_action"] or "").strip()
            if not route_mode:
                continue
            key = (route_mode, recommended_action)
            payload = grouped.setdefault(
                key,
                {
                    "route_mode": route_mode,
                    "recommended_action": recommended_action,
                    "samples": 0,
                    "effective_samples": 0.0,
                    "confidence_sum": 0.0,
                    "best_bucket_rank": 0.0,
                    "latest_created_at": "",
                },
            )
            payload["samples"] = int(payload["samples"]) + 1
            payload["effective_samples"] = float(payload["effective_samples"]) + weight
            payload["confidence_sum"] = float(payload["confidence_sum"]) + (confidence * weight)
            payload["best_bucket_rank"] = max(
                float(payload["best_bucket_rank"]),
                float(len(buckets) - buckets.index(state_bucket)),
            )
            payload["latest_created_at"] = max(
                str(payload["latest_created_at"] or ""),
                str(row["created_at"] or ""),
            )
        if not grouped:
            return None
        ranked = sorted(
            grouped.values(),
            key=lambda item: (
                float(item["effective_samples"]),
                float(item["confidence_sum"]) / max(1.0, float(item["effective_samples"])),
                float(item["best_bucket_rank"]),
                str(item["latest_created_at"]),
            ),
            reverse=True,
        )
        top = ranked[0]
        min_samples_required = max(1, int(min_samples))
        if int(top["samples"]) < min_samples_required:
            return None
        return {
            "route_mode": str(top["route_mode"]),
            "recommended_action": str(top["recommended_action"]),
            "samples": int(top["samples"]),
            "effective_samples": round(float(top["effective_samples"]), 6),
            "min_samples_required": min_samples_required,
            "confidence": round(
                float(top["confidence_sum"]) / max(1.0, float(top["effective_samples"])),
                6,
            ),
            "source": "reflection_success",
        }

    def recommend_reflection_guardrails(
        self,
        *,
        chat_key: str | None = None,
        limit: int = 24,
        min_runs: int = 6,
        low_rate: float = 0.15,
        high_rate: float = 0.6,
    ) -> dict[str, Any]:
        rows = self._ensure_db().execute(
            """
            SELECT execution_style_reflection_count,
                   parallelism_reflection_count,
                   worker_reflection_count
            FROM policy_runtime_telemetry
            WHERE (? IS NULL OR chat_key = ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_key, chat_key, max(1, int(limit))),
        ).fetchall()
        runs = len(rows)

        def _build_dimension(rate: float) -> dict[str, Any]:
            normalized = float(rate or 0.0)
            return {
                "hit_rate": round(normalized, 6),
                "injection_level": "reduced" if runs >= min_runs and normalized <= low_rate else "normal",
                "soften_default": bool(runs >= min_runs and normalized >= high_rate),
            }

        if not rows:
            return {
                "samples": 0,
                "execution_style": _build_dimension(0.0),
                "parallelism": _build_dimension(0.0),
                "worker": _build_dimension(0.0),
            }
        execution_style_rate = sum(
            1 for row in rows if int(row["execution_style_reflection_count"] or 0) > 0
        ) / runs
        parallelism_rate = sum(
            1 for row in rows if int(row["parallelism_reflection_count"] or 0) > 0
        ) / runs
        worker_rate = sum(
            1 for row in rows if int(row["worker_reflection_count"] or 0) > 0
        ) / runs
        return {
            "samples": runs,
            "execution_style": _build_dimension(execution_style_rate),
            "parallelism": _build_dimension(parallelism_rate),
            "worker": _build_dimension(worker_rate),
        }

    def recommend_route_from_intent_bucket(
        self,
        *,
        intent_bucket: str,
        chat_key: str | None = None,
        limit: int = 24,
        min_samples: int = 3,
        min_win_rate: float = 0.75,
    ) -> dict[str, Any] | None:
        bucket = str(intent_bucket or "").strip()
        if not bucket:
            return None
        rows = self._ensure_db().execute(
            """
            SELECT t.route_mode,
                   t.execution_style,
                   o.final_status
            FROM policy_runtime_telemetry AS t
            JOIN policy_outcomes AS o
              ON o.flow_id = t.flow_id
            WHERE t.intent_bucket = ?
              AND (? IS NULL OR t.chat_key = ?)
            ORDER BY t.id DESC
            LIMIT ?
            """,
            (bucket, chat_key, chat_key, max(1, int(limit))),
        ).fetchall()
        if len(rows) < max(1, int(min_samples)):
            return None
        grouped: dict[tuple[str, str], int] = {}
        for row in rows:
            route_mode = str(row["route_mode"] or "").strip()
            execution_style = str(row["execution_style"] or "").strip()
            if route_mode not in {"tool_workflow", "answer", "debate"}:
                continue
            if str(row["final_status"] or "").strip().lower() != "succeeded":
                continue
            key = (route_mode, execution_style)
            grouped[key] = grouped.get(key, 0) + 1
        if not grouped:
            return None
        ranked = sorted(
            grouped.items(),
            key=lambda item: (int(item[1]), item[0][0], item[0][1]),
            reverse=True,
        )
        (route_mode, execution_style), wins = ranked[0]
        total = len(rows)
        win_rate = wins / max(1, total)
        if wins < max(1, int(min_samples)) or win_rate < float(min_win_rate):
            return None
        return {
            "route_mode": route_mode,
            "execution_style": execution_style or "analyze_first",
            "samples": total,
            "wins": wins,
            "win_rate": round(win_rate, 6),
            "source": "intent_bucket_success",
        }

    def record_runtime_telemetry(
        self,
        *,
        flow_id: str,
        chat_key: str,
        route_mode: str,
        router_model: str,
        router_latency_ms: float,
        router_fallback: bool,
        router_source: str = "model",
        execution_style: str = "",
        intent_bucket: str = "",
        reflection_hint_count: int = 0,
        reflection_override_count: int = 0,
        execution_style_reflection_count: int = 0,
        parallelism_reflection_count: int = 0,
        worker_reflection_count: int = 0,
        execution_style_guardrail_reduce_count: int = 0,
        parallelism_guardrail_soften_count: int = 0,
        worker_guardrail_soften_count: int = 0,
        shadow_routing_eval_count: int = 0,
        shadow_routing_agree_count: int = 0,
        created_at: str | None = None,
    ) -> None:
        db = self._ensure_db()
        db.execute(
            """
            INSERT OR REPLACE INTO policy_runtime_telemetry (
                flow_id, chat_key, route_mode, router_model, router_latency_ms,
                router_fallback, router_source, execution_style, intent_bucket,
                reflection_hint_count,
                reflection_override_count, execution_style_reflection_count,
                parallelism_reflection_count, worker_reflection_count,
                execution_style_guardrail_reduce_count,
                parallelism_guardrail_soften_count,
                worker_guardrail_soften_count,
                shadow_routing_eval_count, shadow_routing_agree_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                chat_key,
                route_mode,
                router_model,
                max(0.0, float(router_latency_ms)),
                1 if router_fallback else 0,
                str(router_source or "model").strip() or "model",
                str(execution_style or "").strip(),
                str(intent_bucket or "").strip(),
                max(0, int(reflection_hint_count)),
                max(0, int(reflection_override_count)),
                max(0, int(execution_style_reflection_count)),
                max(0, int(parallelism_reflection_count)),
                max(0, int(worker_reflection_count)),
                max(0, int(execution_style_guardrail_reduce_count)),
                max(0, int(parallelism_guardrail_soften_count)),
                max(0, int(worker_guardrail_soften_count)),
                max(0, int(shadow_routing_eval_count)),
                max(0, int(shadow_routing_agree_count)),
                created_at or self._utc_now(),
            ),
        )
        db.commit()

    def record_shadow_routing_eval(
        self,
        *,
        flow_id: str,
        agreed: bool,
    ) -> None:
        db = self._ensure_db()
        db.execute(
            """
            UPDATE policy_runtime_telemetry
            SET shadow_routing_eval_count = 1,
                shadow_routing_agree_count = ?
            WHERE flow_id = ?
            """,
            (1 if agreed else 0, flow_id),
        )
        db.commit()

    def recommend_router_timeout(
        self,
        *,
        base_timeout: float,
        chat_key: str | None = None,
        router_model: str = "",
        limit: int = 24,
    ) -> dict[str, Any]:
        configured_base = max(0.5, float(base_timeout or 0.0))
        rows = self._ensure_db().execute(
            """
            SELECT router_latency_ms, router_fallback, router_source
            FROM policy_runtime_telemetry
            WHERE router_source='model'
              AND (? IS NULL OR chat_key = ?)
              AND (? = '' OR router_model = ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                chat_key,
                chat_key,
                str(router_model or "").strip(),
                str(router_model or "").strip(),
                max(1, int(limit)),
            ),
        ).fetchall()
        if not rows:
            return {
                "timeout_seconds": configured_base,
                "samples": 0,
                "avg_latency_ms": 0.0,
                "fallback_rate": 0.0,
                "router_source": "configured_base",
            }
        latencies = sorted(max(0.0, float(row["router_latency_ms"] or 0.0)) for row in rows)
        avg_latency_ms = sum(latencies) / len(latencies)
        fallback_rate = sum(int(row["router_fallback"] or 0) for row in rows) / len(rows)
        if len(latencies) < 4:
            timeout_seconds = configured_base
        else:
            p75_latency_ms = latencies[min(len(latencies) - 1, int(len(latencies) * 0.75))]
            target = max(
                0.8,
                avg_latency_ms * 1.8 / 1000.0,
                p75_latency_ms * 1.35 / 1000.0,
            )
            if fallback_rate >= 0.5:
                target = min(target, max(0.7, configured_base * 0.7))
            elif fallback_rate >= 0.25:
                target = min(target, max(0.8, configured_base * 0.85))
            timeout_seconds = min(configured_base, max(0.5, target))
        return {
            "timeout_seconds": round(timeout_seconds, 3),
            "samples": len(rows),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "fallback_rate": round(fallback_rate, 6),
            "router_source": "model_recent",
        }

    def summarize_runtime_telemetry(
        self,
        *,
        chat_key: str | None = None,
    ) -> dict[str, Any]:
        rows = self._ensure_db().execute(
            """
            SELECT t.flow_id,
                   t.chat_key,
                   t.route_mode,
                   t.router_model,
                   t.router_latency_ms,
                   t.router_fallback,
                   t.router_source,
                   t.execution_style,
                   t.intent_bucket,
                   t.reflection_hint_count,
                   t.reflection_override_count,
                   t.execution_style_reflection_count,
                   t.parallelism_reflection_count,
                   t.worker_reflection_count,
                   t.execution_style_guardrail_reduce_count,
                   t.parallelism_guardrail_soften_count,
                   t.worker_guardrail_soften_count,
                   t.shadow_routing_eval_count,
                   t.shadow_routing_agree_count,
                   o.reward
            FROM policy_runtime_telemetry AS t
            LEFT JOIN policy_outcomes AS o
              ON o.flow_id = t.flow_id
            WHERE (? IS NULL OR t.chat_key = ?)
            ORDER BY t.id DESC
            """,
            (chat_key, chat_key),
        ).fetchall()
        overall = self._summarize_runtime_telemetry_rows(rows)
        by_route_mode: dict[str, dict[str, float | int]] = {}
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["route_mode"] or "unknown"), []).append(row)
        for route_mode, route_rows in grouped.items():
            by_route_mode[route_mode] = self._summarize_runtime_telemetry_rows(route_rows)
        return {
            "overall": overall,
            "by_route_mode": by_route_mode,
        }

    def latest_feedback(self, flow_id: str) -> dict[str, Any] | None:
        row = self._ensure_db().execute(
            """
            SELECT flow_id, chat_key, rating, reason, created_at
            FROM policy_feedback
            WHERE flow_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (flow_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def summarize_action_stats(
        self,
        *,
        decision_kind: str | None = None,
        state_bucket: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, dict[str, float | int]]:
        db = self._ensure_db()
        current_time = now or datetime.now(timezone.utc)
        rows = db.execute(
            """
            SELECT d.flow_id AS flow_id,
                   d.action_name AS action_name,
                   d.state_features_json AS state_features_json,
                   d.created_at AS decision_created_at,
                   o.reward AS reward,
                   o.final_status AS final_status,
                   o.outcome_json AS outcome_json,
                   o.created_at AS outcome_created_at
            FROM policy_decisions AS d
            LEFT JOIN policy_outcomes AS o
              ON o.flow_id = d.flow_id
            WHERE (? IS NULL OR d.decision_kind = ?)
            """,
            (decision_kind, decision_kind),
        ).fetchall()
        summary: dict[str, dict[str, float | int]] = {}
        flow_actions: dict[str, set[str]] = {}
        for row in rows:
            flow_id = str(row["flow_id"] or "")
            action_name = str(row["action_name"] or "")
            state_features = self._decode_json_object(row["state_features_json"])
            if state_bucket is not None:
                if state_bucket not in build_policy_state_buckets(state_features):
                    continue
            flow_actions.setdefault(flow_id, set()).add(action_name)
            bucket = summary.setdefault(
                action_name,
                {
                    "samples": 0,
                    "effective_samples": 0.0,
                    "mean_reward": 0.0,
                    "recent_mean_reward": 0.0,
                    "drift_score": 0.0,
                    "failure_rate": 0.0,
                    "success_rate": 0.0,
                    "retry_rate": 0.0,
                    "dead_letter_rate": 0.0,
                    "stalled_rate": 0.0,
                    "feedback_good_count": 0,
                    "feedback_bad_count": 0,
                    "effective_feedback_samples": 0.0,
                    "feedback_confidence": 0.0,
                    "feedback_score": 0.0,
                    "recent_guard_samples": 0.0,
                    "recent_failure_rate": 0.0,
                    "recent_bad_feedback_rate": 0.0,
                    "last_updated_at": "",
                },
            )
            reward = row["reward"]
            if reward is None:
                continue
            reference_time = self._parse_time(
                row["outcome_created_at"] or row["decision_created_at"]
            )
            weight = self._decay_weight(reference_time, now=current_time)
            bucket["samples"] = int(bucket["samples"]) + 1
            bucket["effective_samples"] = float(bucket["effective_samples"]) + weight
            effective_samples = float(bucket["effective_samples"])
            bucket["mean_reward"] = (
                (
                    float(bucket["mean_reward"])
                    * max(0.0, effective_samples - weight)
                )
                + (float(reward) * weight)
            ) / effective_samples
            final_status = str(row["final_status"] or "").strip().lower()
            if final_status == "failed":
                bucket["failure_rate"] = float(bucket["failure_rate"]) + weight
            elif final_status == "succeeded":
                bucket["success_rate"] = float(bucket["success_rate"]) + weight
            if self._is_recent(reference_time, now=current_time):
                bucket["recent_guard_samples"] = float(bucket["recent_guard_samples"]) + 1.0
                recent_guard_samples = float(bucket["recent_guard_samples"])
                bucket["recent_mean_reward"] = (
                    (
                        float(bucket["recent_mean_reward"])
                        * max(0.0, recent_guard_samples - 1.0)
                    )
                    + float(reward)
                ) / recent_guard_samples
                if final_status == "failed":
                    bucket["recent_failure_rate"] = (
                        float(bucket["recent_failure_rate"]) + 1.0
                    )
            outcome = self._decode_json_object(row["outcome_json"])
            bucket["retry_rate"] = float(bucket["retry_rate"]) + float(
                outcome.get("retry_count", 0) or 0
            ) * weight
            bucket["dead_letter_rate"] = float(bucket["dead_letter_rate"]) + float(
                outcome.get("dead_letter_count", 0) or 0
            ) * weight
            bucket["stalled_rate"] = float(bucket["stalled_rate"]) + float(
                outcome.get("stalled_count", 0) or 0
            ) * weight
            bucket["last_updated_at"] = max(
                str(bucket.get("last_updated_at", "") or ""),
                reference_time.isoformat(timespec="seconds"),
            )
        for payload in summary.values():
            effective_samples = float(payload.get("effective_samples", 0.0) or 0.0)
            if effective_samples <= 0:
                continue
            payload["effective_samples"] = round(effective_samples, 6)
            payload["failure_rate"] = float(payload["failure_rate"]) / effective_samples
            payload["success_rate"] = float(payload["success_rate"]) / effective_samples
            payload["retry_rate"] = float(payload["retry_rate"]) / effective_samples
            payload["dead_letter_rate"] = float(payload["dead_letter_rate"]) / effective_samples
            payload["stalled_rate"] = float(payload["stalled_rate"]) / effective_samples
        if flow_actions:
            placeholders = ", ".join("?" for _ in flow_actions)
            feedback_rows = db.execute(
                f"""
                SELECT flow_id, rating, created_at
                FROM policy_feedback
                WHERE flow_id IN ({placeholders})
                """,
                tuple(flow_actions.keys()),
            ).fetchall()
            for row in feedback_rows:
                flow_id = str(row["flow_id"] or "")
                rating = str(row["rating"] or "").strip().lower()
                created_at = self._parse_time(row["created_at"])
                weight = self._decay_weight(
                    created_at,
                    now=current_time,
                    half_life_days=self._FEEDBACK_HALF_LIFE_DAYS,
                )
                signed = 1.0 if rating == "good" else -1.0 if rating == "bad" else 0.0
                for action_name in flow_actions.get(flow_id, ()):
                    payload = summary.get(action_name)
                    if payload is None:
                        continue
                    if rating == "good":
                        payload["feedback_good_count"] = (
                            int(payload["feedback_good_count"]) + 1
                        )
                    elif rating == "bad":
                        payload["feedback_bad_count"] = (
                            int(payload["feedback_bad_count"]) + 1
                        )
                    payload["effective_feedback_samples"] = (
                        float(payload["effective_feedback_samples"]) + weight
                    )
                    payload["feedback_score"] = float(payload["feedback_score"]) + (
                        signed * weight
                    )
                    if self._is_recent(created_at, now=current_time) and rating == "bad":
                        payload["recent_bad_feedback_rate"] = (
                            float(payload["recent_bad_feedback_rate"]) + 1.0
                        )
        for payload in summary.values():
            good_count = int(payload.get("feedback_good_count", 0) or 0)
            bad_count = int(payload.get("feedback_bad_count", 0) or 0)
            total_feedback = float(payload.get("effective_feedback_samples", 0.0) or 0.0)
            if total_feedback > 0:
                payload["feedback_score"] = float(payload["feedback_score"]) / total_feedback
                payload["feedback_confidence"] = min(1.0, total_feedback / 3.0)
                payload["effective_feedback_samples"] = round(total_feedback, 6)
            elif good_count + bad_count > 0:
                payload["feedback_confidence"] = 0.0
            recent_guard_samples = float(payload.get("recent_guard_samples", 0.0) or 0.0)
            if recent_guard_samples > 0:
                recent_mean_reward = float(payload.get("recent_mean_reward", 0.0) or 0.0)
                payload["recent_failure_rate"] = (
                    float(payload["recent_failure_rate"]) / recent_guard_samples
                )
                payload["recent_bad_feedback_rate"] = min(
                    1.0,
                    float(payload["recent_bad_feedback_rate"]) / recent_guard_samples,
                )
                payload["drift_score"] = max(
                    0.0,
                    float(payload.get("mean_reward", 0.0) or 0.0) - recent_mean_reward,
                )
        return summary

    def _ensure_db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self._db_path))
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_decisions (
                    id INTEGER PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    chat_key TEXT NOT NULL,
                    decision_kind TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    state_features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_outcomes (
                    id INTEGER PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    chat_key TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    reward REAL NOT NULL,
                    outcome_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_feedback (
                    id INTEGER PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    chat_key TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_reflections (
                    id INTEGER PRIMARY KEY,
                    chat_key TEXT NOT NULL,
                    route_mode TEXT NOT NULL,
                    state_bucket TEXT NOT NULL,
                    state_features_json TEXT NOT NULL,
                    failure_pattern TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_runtime_telemetry (
                    id INTEGER PRIMARY KEY,
                    flow_id TEXT NOT NULL UNIQUE,
                    chat_key TEXT NOT NULL,
                    route_mode TEXT NOT NULL,
                    router_model TEXT NOT NULL,
                    router_latency_ms REAL NOT NULL,
                    router_fallback INTEGER NOT NULL DEFAULT 0,
                    router_source TEXT NOT NULL DEFAULT 'model',
                    execution_style TEXT NOT NULL DEFAULT '',
                    intent_bucket TEXT NOT NULL DEFAULT '',
                    reflection_hint_count INTEGER NOT NULL DEFAULT 0,
                    reflection_override_count INTEGER NOT NULL DEFAULT 0,
                    execution_style_reflection_count INTEGER NOT NULL DEFAULT 0,
                    parallelism_reflection_count INTEGER NOT NULL DEFAULT 0,
                    worker_reflection_count INTEGER NOT NULL DEFAULT 0,
                    execution_style_guardrail_reduce_count INTEGER NOT NULL DEFAULT 0,
                    parallelism_guardrail_soften_count INTEGER NOT NULL DEFAULT 0,
                    worker_guardrail_soften_count INTEGER NOT NULL DEFAULT 0,
                    shadow_routing_eval_count INTEGER NOT NULL DEFAULT 0,
                    shadow_routing_agree_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"] or "")
                for row in db.execute("PRAGMA table_info(policy_runtime_telemetry)").fetchall()
            }
            if "router_source" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN router_source TEXT NOT NULL DEFAULT 'model'
                    """
                )
            if "execution_style" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN execution_style TEXT NOT NULL DEFAULT ''
                    """
                )
            if "intent_bucket" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN intent_bucket TEXT NOT NULL DEFAULT ''
                    """
                )
            if "execution_style_reflection_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN execution_style_reflection_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "parallelism_reflection_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN parallelism_reflection_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "worker_reflection_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN worker_reflection_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "execution_style_guardrail_reduce_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN execution_style_guardrail_reduce_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "parallelism_guardrail_soften_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN parallelism_guardrail_soften_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "worker_guardrail_soften_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN worker_guardrail_soften_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "shadow_routing_eval_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN shadow_routing_eval_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "shadow_routing_agree_count" not in columns:
                db.execute(
                    """
                    ALTER TABLE policy_runtime_telemetry
                    ADD COLUMN shadow_routing_agree_count INTEGER NOT NULL DEFAULT 0
                    """
                )
            db.commit()
            self._db = db
        return self._db

    @staticmethod
    def _summarize_runtime_telemetry_rows(
        rows: list[sqlite3.Row],
    ) -> dict[str, float | int]:
        if not rows:
            return {
                "runs": 0,
                "avg_router_latency_ms": 0.0,
                "fallback_rate": 0.0,
                "rule_hit_rate": 0.0,
                "reflection_route_rate": 0.0,
                "reflection_match_rate": 0.0,
                "reflection_override_rate": 0.0,
                "execution_style_reflection_rate": 0.0,
                "parallelism_reflection_rate": 0.0,
                "worker_reflection_rate": 0.0,
                "execution_style_guardrail_reduce_rate": 0.0,
                "parallelism_guardrail_soften_rate": 0.0,
                "worker_guardrail_soften_rate": 0.0,
                "shadow_routing_eval_rate": 0.0,
                "shadow_routing_agreement_rate": 0.0,
                "mean_reward": 0.0,
            }
        runs = len(rows)
        latency_total = sum(float(row["router_latency_ms"] or 0.0) for row in rows)
        fallback_total = sum(int(row["router_fallback"] or 0) for row in rows)
        rule_hit_total = sum(
            1 for row in rows if str(row["router_source"] or "").strip() == "rule"
        )
        reflection_route_total = sum(
            1 for row in rows if str(row["router_source"] or "").strip() == "reflection"
        )
        reflection_match_total = sum(
            1 for row in rows if int(row["reflection_hint_count"] or 0) > 0
        )
        reflection_override_total = sum(
            1 for row in rows if int(row["reflection_override_count"] or 0) > 0
        )
        execution_style_reflection_total = sum(
            1 for row in rows if int(row["execution_style_reflection_count"] or 0) > 0
        )
        parallelism_reflection_total = sum(
            1 for row in rows if int(row["parallelism_reflection_count"] or 0) > 0
        )
        worker_reflection_total = sum(
            1 for row in rows if int(row["worker_reflection_count"] or 0) > 0
        )
        execution_style_guardrail_reduce_total = sum(
            1 for row in rows if int(row["execution_style_guardrail_reduce_count"] or 0) > 0
        )
        parallelism_guardrail_soften_total = sum(
            1 for row in rows if int(row["parallelism_guardrail_soften_count"] or 0) > 0
        )
        worker_guardrail_soften_total = sum(
            1 for row in rows if int(row["worker_guardrail_soften_count"] or 0) > 0
        )
        shadow_routing_eval_total = sum(
            1 for row in rows if int(row["shadow_routing_eval_count"] or 0) > 0
        )
        shadow_routing_agree_total = sum(
            1 for row in rows if int(row["shadow_routing_agree_count"] or 0) > 0
        )
        reward_values = [float(row["reward"]) for row in rows if row["reward"] is not None]
        return {
            "runs": runs,
            "avg_router_latency_ms": round(latency_total / runs, 2),
            "fallback_rate": round(fallback_total / runs, 6),
            "rule_hit_rate": round(rule_hit_total / runs, 6),
            "reflection_route_rate": round(reflection_route_total / runs, 6),
            "reflection_match_rate": round(reflection_match_total / runs, 6),
            "reflection_override_rate": round(reflection_override_total / runs, 6),
            "execution_style_reflection_rate": round(
                execution_style_reflection_total / runs, 6
            ),
            "parallelism_reflection_rate": round(
                parallelism_reflection_total / runs, 6
            ),
            "worker_reflection_rate": round(worker_reflection_total / runs, 6),
            "execution_style_guardrail_reduce_rate": round(
                execution_style_guardrail_reduce_total / runs, 6
            ),
            "parallelism_guardrail_soften_rate": round(
                parallelism_guardrail_soften_total / runs, 6
            ),
            "worker_guardrail_soften_rate": round(
                worker_guardrail_soften_total / runs, 6
            ),
            "shadow_routing_eval_rate": round(
                shadow_routing_eval_total / runs, 6
            ),
            "shadow_routing_agreement_rate": round(
                shadow_routing_agree_total / max(1, shadow_routing_eval_total), 6
            ),
            "mean_reward": round(
                sum(reward_values) / len(reward_values), 6
            ) if reward_values else 0.0,
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @classmethod
    def _decay_weight(
        cls,
        created_at: datetime,
        *,
        now: datetime,
        half_life_days: float | None = None,
    ) -> float:
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        half_life = half_life_days or cls._OUTCOME_HALF_LIFE_DAYS
        return 0.5 ** (age_days / half_life)

    @staticmethod
    def _parse_time(raw: Any) -> datetime:
        if isinstance(raw, datetime):
            value = raw
        else:
            value = datetime.fromisoformat(str(raw))
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _is_recent(cls, created_at: datetime, *, now: datetime) -> bool:
        return (now - created_at).total_seconds() <= cls._RECENT_WINDOW_DAYS * 86400.0

    @staticmethod
    def _decode_json_object(raw: Any) -> dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        try:
            decoded = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
