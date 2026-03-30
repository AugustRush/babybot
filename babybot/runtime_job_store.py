"""SQLite-backed store for resumable runtime jobs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime_jobs import JOB_STATES, RuntimeJob


class RuntimeJobStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db: sqlite3.Connection | None = None

    def create(self, *, chat_key: str, goal: str, plan_id: str = "") -> RuntimeJob:
        job = RuntimeJob(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            chat_key=str(chat_key or ""),
            goal=str(goal or ""),
            plan_id=str(plan_id or ""),
            state="queued",
            created_at=self._utc_now(),
            updated_at=self._utc_now(),
        )
        db = self._ensure_db()
        db.execute(
            """
            INSERT INTO runtime_jobs (
                job_id, chat_key, goal, plan_id, state, progress_message,
                result_text, error, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                job.chat_key,
                job.goal,
                job.plan_id,
                job.state,
                job.progress_message,
                job.result_text,
                job.error,
                json.dumps(job.metadata, ensure_ascii=False, sort_keys=True),
                job.created_at,
                job.updated_at,
            ),
        )
        db.commit()
        return job

    def transition(self, job_id: str, state: str, **fields: Any) -> RuntimeJob:
        normalized_state = str(state or "").strip()
        if normalized_state not in JOB_STATES:
            raise ValueError(f"invalid job state: {state}")
        current = self.get(job_id)
        if current is None:
            raise ValueError(f"unknown job_id: {job_id}")
        merged_metadata = dict(current.metadata)
        raw_metadata = fields.pop("metadata", None)
        if isinstance(raw_metadata, dict):
            merged_metadata.update(raw_metadata)
        next_job = RuntimeJob(
            job_id=current.job_id,
            chat_key=current.chat_key,
            goal=current.goal,
            plan_id=str(fields.get("plan_id", current.plan_id) or ""),
            state=normalized_state,  # type: ignore[arg-type]
            progress_message=str(
                fields.get("progress_message", current.progress_message) or ""
            ),
            result_text=str(fields.get("result_text", current.result_text) or ""),
            error=str(fields.get("error", current.error) or ""),
            created_at=current.created_at,
            updated_at=self._utc_now(),
            metadata=merged_metadata,
        )
        db = self._ensure_db()
        db.execute(
            """
            UPDATE runtime_jobs
            SET plan_id=?, state=?, progress_message=?, result_text=?, error=?,
                metadata_json=?, updated_at=?
            WHERE job_id=?
            """,
            (
                next_job.plan_id,
                next_job.state,
                next_job.progress_message,
                next_job.result_text,
                next_job.error,
                json.dumps(next_job.metadata, ensure_ascii=False, sort_keys=True),
                next_job.updated_at,
                next_job.job_id,
            ),
        )
        db.commit()
        return next_job

    def get(self, job_id: str) -> RuntimeJob | None:
        row = self._ensure_db().execute(
            """
            SELECT job_id, chat_key, goal, plan_id, state, progress_message,
                   result_text, error, metadata_json, created_at, updated_at
            FROM runtime_jobs
            WHERE job_id=?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def latest_for_chat(self, chat_key: str) -> RuntimeJob | None:
        row = self._ensure_db().execute(
            """
            SELECT job_id, chat_key, goal, plan_id, state, progress_message,
                   result_text, error, metadata_json, created_at, updated_at
            FROM runtime_jobs
            WHERE chat_key=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_key,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def _ensure_db(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self._db_path))
        db.row_factory = sqlite3.Row
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                chat_key TEXT NOT NULL,
                goal TEXT NOT NULL,
                plan_id TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL,
                progress_message TEXT NOT NULL DEFAULT '',
                result_text TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        db.commit()
        self._db = db
        return db

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> RuntimeJob:
        metadata: dict[str, Any] = {}
        try:
            loaded = json.loads(str(row["metadata_json"] or "{}"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        return RuntimeJob(
            job_id=str(row["job_id"] or ""),
            chat_key=str(row["chat_key"] or ""),
            goal=str(row["goal"] or ""),
            plan_id=str(row["plan_id"] or ""),
            state=str(row["state"] or "queued"),  # type: ignore[arg-type]
            progress_message=str(row["progress_message"] or ""),
            result_text=str(row["result_text"] or ""),
            error=str(row["error"] or ""),
            created_at=str(row["created_at"] or ""),
            updated_at=str(row["updated_at"] or ""),
            metadata=metadata,
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
