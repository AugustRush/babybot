"""Durable SQLite-backed storage for plan notebooks."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..sqlite_utils import connect_sqlite
from .plan_notebook import PlanNotebook


def _json_safe(payload: Any) -> Any:
    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in payload]
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump())
        except Exception:
            return str(payload)
    if is_dataclass(payload) and not isinstance(payload, type):
        try:
            return _json_safe(asdict(payload))
        except Exception:
            return str(payload)
    return str(payload)


class PlanNotebookStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        prefer_fts: bool = True,
        busy_timeout_ms: int = 3000,
    ) -> None:
        self._db_path = Path(db_path)
        self._prefer_fts = bool(prefer_fts)
        self._busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._db: sqlite3.Connection | None = None
        self._fts_enabled = False

    def save_notebook(self, notebook: PlanNotebook, *, chat_key: str = "") -> None:
        db = self._ensure_db()
        payload = _json_safe(notebook.to_dict())
        notebook_metadata = _json_safe(notebook.metadata)
        db.execute(
            """
            INSERT INTO plan_notebooks (
                notebook_id, chat_key, goal, flow_id, plan_id,
                snapshot_json, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(notebook_id) DO UPDATE SET
                chat_key=excluded.chat_key,
                goal=excluded.goal,
                flow_id=excluded.flow_id,
                plan_id=excluded.plan_id,
                snapshot_json=excluded.snapshot_json,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                notebook.notebook_id,
                str(chat_key or notebook.metadata.get("chat_key", "") or ""),
                notebook.goal,
                notebook.flow_id,
                notebook.plan_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                json.dumps(notebook_metadata, ensure_ascii=False, sort_keys=True),
                notebook.created_at,
                notebook.updated_at,
            ),
        )
        db.execute("DELETE FROM notebook_events WHERE notebook_id=?", (notebook.notebook_id,))
        if self._fts_enabled:
            db.execute("DELETE FROM notebook_events_fts WHERE notebook_id=?", (notebook.notebook_id,))
        for node in notebook.nodes.values():
            for event in node.events:
                self._save_event(
                    notebook=notebook,
                    chat_key=str(chat_key or notebook.metadata.get("chat_key", "") or ""),
                    event_payload={
                        "node_id": node.node_id,
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "summary": event.summary,
                        "detail": event.detail,
                        "created_at": event.created_at,
                        "metadata": dict(event.metadata),
                    },
                )
        self._save_completion_summary(
            notebook.notebook_id,
            chat_key=str(chat_key or notebook.metadata.get("chat_key", "") or ""),
            flow_id=notebook.flow_id,
            completion_summary=dict(notebook.completion_summary),
        )
        db.commit()

    def load_notebook(self, notebook_id: str) -> PlanNotebook | None:
        row = self._ensure_db().execute(
            """
            SELECT snapshot_json
            FROM plan_notebooks
            WHERE notebook_id=?
            """,
            (notebook_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["snapshot_json"] or "{}"))
        notebook = PlanNotebook.from_dict(payload)
        completion_row = self._ensure_db().execute(
            """
            SELECT summary_json
            FROM notebook_completion
            WHERE notebook_id=?
            """,
            (notebook_id,),
        ).fetchone()
        if completion_row is not None:
            notebook.completion_summary = dict(
                json.loads(str(completion_row["summary_json"] or "{}")) or {}
            )
        return notebook

    def search_raw_text(
        self,
        query: str,
        *,
        notebook_id: str = "",
        chat_key: str = "",
        flow_id: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        db = self._ensure_db()
        params: list[Any] = []
        where_clauses: list[str] = []
        if notebook_id:
            where_clauses.append("notebook_id=?")
            params.append(notebook_id)
        if chat_key:
            where_clauses.append("chat_key=?")
            params.append(chat_key)
        if flow_id:
            where_clauses.append("flow_id=?")
            params.append(flow_id)
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        if self._fts_enabled and self._prefer_fts:
            fts_where_clauses = [clause.replace("notebook_id", "e.notebook_id") for clause in where_clauses]
            fts_where_clauses = [clause.replace("chat_key", "e.chat_key") for clause in fts_where_clauses]
            fts_where_clauses = [clause.replace("flow_id", "e.flow_id") for clause in fts_where_clauses]
            fts_where_sql = (
                "WHERE " + " AND ".join(fts_where_clauses)
                if fts_where_clauses
                else ""
            )
            sql = f"""
                SELECT
                    e.notebook_id,
                    e.node_id,
                    e.event_id,
                    e.kind,
                    e.summary,
                    e.detail,
                    e.created_at
                FROM notebook_events_fts f
                JOIN notebook_events e ON e.event_id = f.event_id
                {fts_where_sql}
                  {"AND" if fts_where_clauses else "WHERE"} notebook_events_fts MATCH ?
                ORDER BY e.created_at DESC
                LIMIT ?
            """
            rows = db.execute(sql, (*params, normalized_query, int(limit))).fetchall()
            if rows:
                return [
                    {
                        "notebook_id": str(row["notebook_id"] or ""),
                        "node_id": str(row["node_id"] or ""),
                        "event_id": str(row["event_id"] or ""),
                        "kind": str(row["kind"] or ""),
                        "summary": str(row["summary"] or ""),
                        "detail": str(row["detail"] or ""),
                        "created_at": float(row["created_at"] or 0.0),
                    }
                    for row in rows
                ]

        sql = f"""
            SELECT notebook_id, node_id, event_id, kind, summary, detail, created_at
            FROM notebook_events
            {where_sql}
              {"AND" if where_clauses else "WHERE"} search_text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """
        rows = db.execute(
            sql,
            (*params, f"%{normalized_query}%", int(limit)),
        ).fetchall()
        return [
            {
                "notebook_id": str(row["notebook_id"] or ""),
                "node_id": str(row["node_id"] or ""),
                "event_id": str(row["event_id"] or ""),
                "kind": str(row["kind"] or ""),
                "summary": str(row["summary"] or ""),
                "detail": str(row["detail"] or ""),
                "created_at": float(row["created_at"] or 0.0),
            }
            for row in rows
        ]

    def _save_event(
        self,
        *,
        notebook: PlanNotebook,
        chat_key: str,
        event_payload: dict[str, Any],
    ) -> None:
        db = self._ensure_db()
        search_text = " ".join(
            part
            for part in (
                event_payload.get("kind", ""),
                event_payload.get("summary", ""),
                event_payload.get("detail", ""),
            )
            if str(part).strip()
        )
        db.execute(
            """
            INSERT OR REPLACE INTO notebook_events (
                notebook_id, chat_key, flow_id, node_id, event_id, kind,
                summary, detail, search_text, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notebook.notebook_id,
                chat_key,
                notebook.flow_id,
                str(event_payload.get("node_id", "") or ""),
                str(event_payload.get("event_id", "") or ""),
                str(event_payload.get("kind", "") or ""),
                str(event_payload.get("summary", "") or ""),
                str(event_payload.get("detail", "") or ""),
                search_text,
                float(event_payload.get("created_at", 0.0) or 0.0),
                json.dumps(
                    _json_safe(event_payload.get("metadata") or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        if self._fts_enabled:
            db.execute(
                """
                INSERT INTO notebook_events_fts (
                    notebook_id, chat_key, flow_id, node_id, event_id,
                    summary, detail, search_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notebook.notebook_id,
                    chat_key,
                    notebook.flow_id,
                    str(event_payload.get("node_id", "") or ""),
                    str(event_payload.get("event_id", "") or ""),
                    str(event_payload.get("summary", "") or ""),
                    str(event_payload.get("detail", "") or ""),
                    search_text,
                    float(event_payload.get("created_at", 0.0) or 0.0),
                ),
            )

    def _save_completion_summary(
        self,
        notebook_id: str,
        *,
        chat_key: str,
        flow_id: str,
        completion_summary: dict[str, Any],
    ) -> None:
        db = self._ensure_db()
        final_summary = str(completion_summary.get("final_summary", "") or "")
        search_terms = completion_summary.get("search_terms") or []
        node_summaries = completion_summary.get("node_summaries") or []
        search_text = " ".join(
            [
                final_summary,
                " ".join(str(item) for item in search_terms),
                " ".join(str(item) for item in node_summaries),
            ]
        ).strip()
        db.execute(
            """
            INSERT OR REPLACE INTO notebook_completion (
                notebook_id, chat_key, flow_id, final_summary, summary_json, search_text, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notebook_id,
                chat_key,
                flow_id,
                final_summary,
                json.dumps(
                    _json_safe(completion_summary),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                search_text,
                completion_summary.get("updated_at") or 0.0,
            ),
        )

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
            CREATE TABLE IF NOT EXISTS plan_notebooks (
                notebook_id TEXT PRIMARY KEY,
                chat_key TEXT NOT NULL DEFAULT '',
                goal TEXT NOT NULL DEFAULT '',
                flow_id TEXT NOT NULL DEFAULT '',
                plan_id TEXT NOT NULL DEFAULT '',
                snapshot_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS notebook_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id TEXT NOT NULL,
                chat_key TEXT NOT NULL DEFAULT '',
                flow_id TEXT NOT NULL DEFAULT '',
                node_id TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_notebook_events_lookup
            ON notebook_events(notebook_id, chat_key, flow_id, created_at DESC)
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS notebook_completion (
                notebook_id TEXT PRIMARY KEY,
                chat_key TEXT NOT NULL DEFAULT '',
                flow_id TEXT NOT NULL DEFAULT '',
                final_summary TEXT NOT NULL DEFAULT '',
                summary_json TEXT NOT NULL DEFAULT '{}',
                search_text TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        self._fts_enabled = False
        if self._prefer_fts:
            try:
                db.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS notebook_events_fts
                    USING fts5(
                        notebook_id UNINDEXED,
                        chat_key UNINDEXED,
                        flow_id UNINDEXED,
                        node_id UNINDEXED,
                        event_id UNINDEXED,
                        summary,
                        detail,
                        search_text,
                        created_at UNINDEXED
                    )
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
        db.commit()
        self._db = db
        return db
