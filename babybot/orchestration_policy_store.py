from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    ) -> dict[str, dict[str, float | int]]:
        db = self._ensure_db()
        query = """
            SELECT d.action_name AS action_name,
                   COUNT(o.id) AS samples,
                   AVG(o.reward) AS mean_reward
            FROM policy_decisions AS d
            LEFT JOIN policy_outcomes AS o
              ON o.flow_id = d.flow_id
            WHERE (? IS NULL OR d.decision_kind = ?)
            GROUP BY d.action_name
        """
        rows = db.execute(query, (decision_kind, decision_kind)).fetchall()
        summary: dict[str, dict[str, float | int]] = {}
        for row in rows:
            summary[str(row["action_name"])] = {
                "samples": int(row["samples"] or 0),
                "mean_reward": float(row["mean_reward"] or 0.0),
            }
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
