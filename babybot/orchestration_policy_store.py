from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .orchestration_policy import build_policy_state_bucket


class OrchestrationPolicyStore:
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
                self._utc_now(),
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
                self._utc_now(),
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
                self._utc_now(),
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
    ) -> dict[str, dict[str, float | int]]:
        db = self._ensure_db()
        rows = db.execute(
            """
            SELECT d.flow_id AS flow_id,
                   d.action_name AS action_name,
                   d.state_features_json AS state_features_json,
                   o.reward AS reward,
                   o.final_status AS final_status,
                   o.outcome_json AS outcome_json
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
                if build_policy_state_bucket(state_features) != state_bucket:
                    continue
            flow_actions.setdefault(flow_id, set()).add(action_name)
            bucket = summary.setdefault(
                action_name,
                {
                    "samples": 0,
                    "mean_reward": 0.0,
                    "failure_rate": 0.0,
                    "success_rate": 0.0,
                    "retry_rate": 0.0,
                    "dead_letter_rate": 0.0,
                    "stalled_rate": 0.0,
                    "feedback_good_count": 0,
                    "feedback_bad_count": 0,
                    "feedback_score": 0.0,
                },
            )
            reward = row["reward"]
            if reward is None:
                continue
            bucket["samples"] = int(bucket["samples"]) + 1
            sample_count = int(bucket["samples"])
            bucket["mean_reward"] = (
                (
                    float(bucket["mean_reward"])
                    * max(0, sample_count - 1)
                )
                + float(reward)
            ) / sample_count
            final_status = str(row["final_status"] or "").strip().lower()
            if final_status == "failed":
                bucket["failure_rate"] = float(bucket["failure_rate"]) + 1.0
            elif final_status == "succeeded":
                bucket["success_rate"] = float(bucket["success_rate"]) + 1.0
            outcome = self._decode_json_object(row["outcome_json"])
            bucket["retry_rate"] = float(bucket["retry_rate"]) + float(
                outcome.get("retry_count", 0) or 0
            )
            bucket["dead_letter_rate"] = float(bucket["dead_letter_rate"]) + float(
                outcome.get("dead_letter_count", 0) or 0
            )
            bucket["stalled_rate"] = float(bucket["stalled_rate"]) + float(
                outcome.get("stalled_count", 0) or 0
            )
        for payload in summary.values():
            samples = int(payload.get("samples", 0) or 0)
            if samples <= 0:
                continue
            payload["failure_rate"] = float(payload["failure_rate"]) / samples
            payload["success_rate"] = float(payload["success_rate"]) / samples
            payload["retry_rate"] = float(payload["retry_rate"]) / samples
            payload["dead_letter_rate"] = float(payload["dead_letter_rate"]) / samples
            payload["stalled_rate"] = float(payload["stalled_rate"]) / samples
        if flow_actions:
            placeholders = ", ".join("?" for _ in flow_actions)
            feedback_rows = db.execute(
                f"""
                SELECT flow_id, rating
                FROM policy_feedback
                WHERE flow_id IN ({placeholders})
                """,
                tuple(flow_actions.keys()),
            ).fetchall()
            for row in feedback_rows:
                flow_id = str(row["flow_id"] or "")
                rating = str(row["rating"] or "").strip().lower()
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
        for payload in summary.values():
            good_count = int(payload.get("feedback_good_count", 0) or 0)
            bad_count = int(payload.get("feedback_bad_count", 0) or 0)
            total_feedback = good_count + bad_count
            if total_feedback > 0:
                payload["feedback_score"] = (good_count - bad_count) / total_feedback
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
