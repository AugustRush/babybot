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
            db.commit()
            self._db = db
        return self._db

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
