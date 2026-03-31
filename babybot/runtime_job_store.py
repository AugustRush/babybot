"""SQLite-backed store for resumable runtime jobs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .runtime_jobs import ACTIVE_JOB_STATES, JOB_STATES, RuntimeJob
from .sqlite_utils import connect_sqlite


class RuntimeJobStore:
    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 3000) -> None:
        self._db_path = Path(db_path)
        self._busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._db: sqlite3.Connection | None = None

    def create(
        self,
        *,
        chat_key: str,
        goal: str,
        plan_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeJob:
        normalized_metadata = dict(metadata or {})
        job = RuntimeJob(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            chat_key=str(chat_key or ""),
            goal=str(goal or ""),
            plan_id=str(plan_id or ""),
            state="queued",
            created_at=self._utc_now(),
            updated_at=self._utc_now(),
            metadata=normalized_metadata,
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
                json.dumps(normalized_metadata, ensure_ascii=False, sort_keys=True),
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

    def find_by_flow_id(self, flow_id: str) -> RuntimeJob | None:
        normalized_flow_id = str(flow_id or "").strip()
        if not normalized_flow_id:
            return None
        for job in self.list_jobs():
            if str(job.metadata.get("flow_id", "") or "").strip() == normalized_flow_id:
                return job
        return None

    def known_flow_ids(self) -> set[str]:
        return {
            flow_id
            for job in self.list_jobs()
            if (flow_id := str(job.metadata.get("flow_id", "") or "").strip())
        }

    def list_jobs(self, *, chat_key: str | None = None) -> list[RuntimeJob]:
        sql = """
            SELECT job_id, chat_key, goal, plan_id, state, progress_message,
                   result_text, error, metadata_json, created_at, updated_at
            FROM runtime_jobs
        """
        params: tuple[Any, ...] = ()
        if chat_key is not None:
            sql += " WHERE chat_key=?"
            params = (chat_key,)
        sql += " ORDER BY id DESC"
        rows = self._ensure_db().execute(sql, params).fetchall()
        return [self._row_to_job(row) for row in rows]

    def run_maintenance(
        self,
        *,
        now: datetime | None = None,
        retention_seconds: int = 3600,
    ) -> dict[str, Any]:
        current_time = now.astimezone(timezone.utc) if now is not None else datetime.now(
            timezone.utc
        )
        cutoff = current_time - timedelta(seconds=max(0, int(retention_seconds)))
        orphaned_job_ids: list[str] = []
        for job in self.list_jobs():
            flow_id = str(job.metadata.get("flow_id", "") or "").strip()
            updated_at = self._parse_timestamp(job.updated_at)
            if (
                job.state in ACTIVE_JOB_STATES
                and updated_at <= cutoff
                and not flow_id
            ):
                orphaned_job_ids.append(job.job_id)

        if orphaned_job_ids:
            placeholders = ",".join("?" for _ in orphaned_job_ids)
            self._ensure_db().execute(
                f"DELETE FROM runtime_jobs WHERE job_id IN ({placeholders})",
                tuple(orphaned_job_ids),
            )
            self._ensure_db().commit()
        return {
            "orphaned_jobs_pruned": len(orphaned_job_ids),
            "orphaned_job_ids": orphaned_job_ids,
        }

    def _ensure_db(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = connect_sqlite(
            self._db_path,
            row_factory=sqlite3.Row,
            busy_timeout_ms=self._busy_timeout_ms,
        )
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

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        text = str(value or "").strip()
        if not text:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
