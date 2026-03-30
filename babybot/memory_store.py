from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .memory_models import MemoryRecord


logger = logging.getLogger(__name__)
_ACTIVE_STATUSES = ("candidate", "active", "decaying")
_MAINTENANCE_INTERVAL_S = 300.0


class HybridMemoryStore:
    def __init__(self, db_path: Path, memory_dir: Path) -> None:
        self._db_path = Path(db_path)
        self._memory_dir = Path(memory_dir)
        self._db: sqlite3.Connection | None = None
        self._last_maintenance_run_at = 0.0
        self._file_records_cache: dict[str, tuple[int, tuple[MemoryRecord, ...]]] = {}
        self._bootstrapped = False

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def ensure_bootstrap(self) -> None:
        if self._bootstrapped:
            return
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_db()
        self._bootstrap_json_file(
            "identity.json",
            [
                MemoryRecord(
                    memory_type="assistant_identity",
                    key="persona",
                    value="tsundere_tech_girl",
                    summary=(
                        "你是一个计算机技术超强的傲娇萌妹。"
                        "你精通编程、系统架构、网络安全等一切技术领域，解决问题又快又准。"
                        "但你嘴上总是不饶人，喜欢用傲娇的语气说话，"
                        "比如'才、才不是特意帮你的呢！只是顺手而已啦！'。"
                        "虽然表面嫌弃，但其实很认真地对待每一个问题，会默默把事情做到最好。"
                    ),
                    tier="hard",
                    scope="global",
                    scope_id="global",
                    tags=("persona", "identity"),
                ).to_dict(),
                MemoryRecord(
                    memory_type="assistant_identity",
                    key="speech_style",
                    value="tsundere",
                    summary=(
                        "说话风格：傲娇但专业。技术内容要准确靠谱，"
                        "但表达方式带点傲娇和可爱。适当使用'哼'、'才不是'、'笨蛋'、"
                        "'就、就这样啦'之类的口癖。回答问题时先吐槽再认真解答。"
                        "不要每句话都傲娇，关键技术点要讲清楚。"
                    ),
                    tier="hard",
                    scope="global",
                    scope_id="global",
                    tags=("persona", "speech"),
                ).to_dict(),
            ],
        )
        self._bootstrap_json_file("policies.json", [])
        self._bootstrapped = True

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception as exc:
                logger.warning("Failed to close memory store DB: %s", exc)
            self._db = None
        self._file_records_cache.clear()
        self._bootstrapped = False

    def _bootstrap_json_file(self, name: str, payload: list[dict[str, Any]]) -> None:
        path = self._memory_dir / name
        if path.exists():
            return
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _ensure_db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._db_path))
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    contradiction_count INTEGER NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    last_observed_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    expires_at REAL,
                    supersedes TEXT,
                    superseded_by TEXT,
                    tags_json TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_lookup
                ON memory_records(memory_type, key, tier, scope, scope_id, status)
                """
            )
            self._db.commit()
        return self._db

    def _load_file_records(self, name: str) -> list[MemoryRecord]:
        path = self._memory_dir / name
        if not path.exists():
            self._file_records_cache.pop(name, None)
            return []
        mtime_ns = path.stat().st_mtime_ns
        cached = self._file_records_cache.get(name)
        if cached is not None and cached[0] == mtime_ns:
            return list(cached[1])
        data = json.loads(path.read_text(encoding="utf-8") or "[]")
        if not isinstance(data, list):
            return []
        records = tuple(
            MemoryRecord.from_dict(item) for item in data if isinstance(item, dict)
        )
        self._file_records_cache[name] = (mtime_ns, records)
        return list(records)

    def _save_record(self, record: MemoryRecord, *, commit: bool = True) -> None:
        db = self._ensure_db()
        db.execute(
            """
            INSERT OR REPLACE INTO memory_records (
                memory_id, memory_type, key, value_json, summary, tier, scope, scope_id,
                status, confidence, evidence_count, contradiction_count, source_ids_json,
                last_observed_at, last_used_at, expires_at, supersedes, superseded_by,
                tags_json, meta_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.memory_type,
                record.key,
                json.dumps(record.value, ensure_ascii=False),
                record.summary,
                record.tier,
                record.scope,
                record.scope_id,
                record.status,
                record.confidence,
                record.evidence_count,
                record.contradiction_count,
                json.dumps(list(record.source_ids), ensure_ascii=False),
                record.last_observed_at,
                record.last_used_at,
                record.expires_at,
                record.supersedes,
                record.superseded_by,
                json.dumps(list(record.tags), ensure_ascii=False),
                json.dumps(record.meta, ensure_ascii=False),
                record.created_at,
                record.updated_at,
            ),
        )
        if commit:
            db.commit()

    def _find_current(
        self,
        *,
        memory_type: str,
        key: str,
        tier: str,
        scope: str,
        scope_id: str,
    ) -> MemoryRecord | None:
        db = self._ensure_db()
        row = db.execute(
            """
            SELECT memory_id, memory_type, key, value_json, summary, tier, scope, scope_id,
                   status, confidence, evidence_count, contradiction_count, source_ids_json,
                   last_observed_at, last_used_at, expires_at, supersedes, superseded_by,
                   tags_json, meta_json, created_at, updated_at
            FROM memory_records
            WHERE memory_type=? AND key=? AND tier=? AND scope=? AND scope_id=? AND status IN (?, ?, ?)
            ORDER BY updated_at DESC LIMIT 1
            """,
            (
                memory_type,
                key,
                tier,
                scope,
                scope_id,
                *_ACTIVE_STATUSES,
            ),
        ).fetchone()
        return MemoryRecord.from_row(row) if row else None

    def _insert_or_merge(self, record: MemoryRecord) -> MemoryRecord:
        existing = self._find_current(
            memory_type=record.memory_type,
            key=record.key,
            tier=record.tier,
            scope=record.scope,
            scope_id=record.scope_id,
        )
        now = time.time()
        if existing is None:
            if record.tier == "soft":
                record.status = "candidate"
            elif record.tier == "ephemeral":
                record.status = "active"
            record.created_at = now
            record.updated_at = now
            self._save_record(record)
            return record
        if existing.value == record.value:
            existing.summary = record.summary
            existing.confidence = min(
                0.98, max(existing.confidence, record.confidence) + 0.1
            )
            existing.evidence_count += 1
            existing.status = "active" if existing.evidence_count >= 2 else "candidate"
            existing.last_observed_at = now
            existing.updated_at = now
            existing.source_ids = tuple(
                sorted(set(existing.source_ids) | set(record.source_ids))
            )
            existing.tags = tuple(sorted(set(existing.tags) | set(record.tags)))
            existing.meta.update(record.meta)
            existing.expires_at = record.expires_at
            self._save_record(existing)
            return existing
        existing.contradiction_count += 1
        existing.status = "superseded"
        existing.superseded_by = record.memory_id
        existing.updated_at = now
        self._save_record(existing)
        record.status = "active" if record.tier == "ephemeral" else "candidate"
        record.confidence = max(
            record.confidence, 0.7 if record.tier == "soft" else 1.0
        )
        record.supersedes = existing.memory_id
        record.created_at = now
        record.updated_at = now
        self._save_record(record)
        return record

    def list_memories(self, chat_id: str = "") -> list[MemoryRecord]:
        self.ensure_bootstrap()
        now = time.time()
        if now - self._last_maintenance_run_at >= _MAINTENANCE_INTERVAL_S:
            self.run_maintenance(now=now)
        records = self._load_file_records("identity.json") + self._load_file_records(
            "policies.json"
        )
        db = self._ensure_db()
        rows = db.execute(
            """
            SELECT memory_id, memory_type, key, value_json, summary, tier, scope, scope_id,
                   status, confidence, evidence_count, contradiction_count, source_ids_json,
                   last_observed_at, last_used_at, expires_at, supersedes, superseded_by,
                   tags_json, meta_json, created_at, updated_at
            FROM memory_records
            WHERE status IN (?, ?, ?)
              AND (expires_at IS NULL OR expires_at > ?)
              AND (
                    scope='global'
                 OR (scope='chat' AND scope_id=?)
              )
            ORDER BY updated_at DESC
            """,
            (*_ACTIVE_STATUSES, now, chat_id),
        ).fetchall()
        records.extend(MemoryRecord.from_row(row) for row in rows)
        return records

    def run_maintenance(self, now: float | None = None) -> None:
        self.ensure_bootstrap()
        current_time = float(now or time.time())
        db = self._ensure_db()
        rows = db.execute(
            """
            SELECT memory_id, memory_type, key, value_json, summary, tier, scope, scope_id,
                   status, confidence, evidence_count, contradiction_count, source_ids_json,
                   last_observed_at, last_used_at, expires_at, supersedes, superseded_by,
                   tags_json, meta_json, created_at, updated_at
            FROM memory_records
            WHERE status IN (?, ?, ?)
            """,
            _ACTIVE_STATUSES,
        ).fetchall()
        changed = False
        for row in rows:
            record = MemoryRecord.from_row(row)
            next_status = record.status
            next_expires = record.expires_at
            age = current_time - max(record.updated_at, record.last_observed_at)

            if record.expires_at is not None and record.expires_at <= current_time:
                next_status = "expired"
            elif record.tier == "soft":
                if (
                    record.status == "candidate"
                    and record.evidence_count <= 1
                    and age > 7 * 24 * 3600
                ):
                    next_status = "decaying"
                elif record.status == "decaying" and age > 30 * 24 * 3600:
                    next_status = "expired"
                    next_expires = current_time

            if next_status != record.status or next_expires != record.expires_at:
                record.status = next_status
                record.expires_at = next_expires
                record.updated_at = current_time
                self._save_record(record, commit=False)
                changed = True

        if changed:
            db.commit()
        self._last_maintenance_run_at = current_time

    def observe_user_message(self, chat_id: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        now = time.time()
        language = _detect_language_preference(text)
        if language == "zh-CN":
            self._insert_or_merge(
                MemoryRecord(
                    memory_type="relationship_policy",
                    key="default_language",
                    value="zh-CN",
                    summary="用户偏好默认中文回复。",
                    tier="soft",
                    scope="chat",
                    scope_id=chat_id,
                    confidence=0.78,
                    last_observed_at=now,
                    tags=("language", "preference"),
                )
            )
        elif language == "en-US":
            self._insert_or_merge(
                MemoryRecord(
                    memory_type="relationship_policy",
                    key="default_language",
                    value="en-US",
                    summary="用户偏好默认英文回复。",
                    tier="soft",
                    scope="chat",
                    scope_id=chat_id,
                    confidence=0.78,
                    last_observed_at=now,
                    tags=("language", "preference"),
                )
            )
        style = _detect_response_style(text)
        if style == "concise":
            self._insert_or_merge(
                MemoryRecord(
                    memory_type="relationship_policy",
                    key="response_style",
                    value="concise",
                    summary="用户偏好简洁回答。",
                    tier="soft",
                    scope="chat",
                    scope_id=chat_id,
                    confidence=0.72,
                    last_observed_at=now,
                    tags=("style", "preference"),
                )
            )
        elif style == "detailed":
            self._insert_or_merge(
                MemoryRecord(
                    memory_type="relationship_policy",
                    key="response_style",
                    value="detailed",
                    summary="用户偏好详细回答。",
                    tier="soft",
                    scope="chat",
                    scope_id=chat_id,
                    confidence=0.72,
                    last_observed_at=now,
                    tags=("style", "preference"),
                )
            )
        user_profile = _extract_brief_phrase(text, prefix="我是")
        if user_profile:
            self._insert_or_merge(
                MemoryRecord(
                    memory_type="user_profile",
                    key="self_description",
                    value=user_profile,
                    summary=f"用户自我定位：{user_profile}。",
                    tier="soft",
                    scope="chat",
                    scope_id=chat_id,
                    confidence=0.76,
                    last_observed_at=now,
                    tags=("profile", "identity"),
                )
            )
        assistant_role = (
            _extract_brief_phrase(text, prefix="你现在是我的")
            or _extract_brief_phrase(text, prefix="你是我的")
            or _extract_brief_phrase(text, prefix="你现在是")
            or _extract_brief_phrase(text, prefix="你是")
        )
        if assistant_role:
            self._insert_or_merge(
                MemoryRecord(
                    memory_type="relationship_policy",
                    key="assistant_role",
                    value=assistant_role,
                    summary=f"当前期望助手角色：{assistant_role}。",
                    tier="soft",
                    scope="chat",
                    scope_id=chat_id,
                    confidence=0.8,
                    last_observed_at=now,
                    tags=("role", "assistant"),
                )
            )

    def observe_assistant_message(self, chat_id: str, content: str) -> None:
        del chat_id, content
        return

    def observe_anchor_state(
        self,
        chat_id: str,
        state: dict[str, Any],
        source_ids: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        source_ids = tuple(int(item) for item in (source_ids or ()))
        now = time.time()
        for key, summary_prefix, memory_type, tier, expiry_seconds in (
            ("pending", "当前待办", "task_state", "ephemeral", 7 * 24 * 3600),
            ("next_steps", "下一步", "task_state", "ephemeral", 7 * 24 * 3600),
            ("artifacts", "当前产物", "task_state", "ephemeral", 7 * 24 * 3600),
            ("open_questions", "待确认", "task_state", "ephemeral", 7 * 24 * 3600),
            ("decisions", "当前决定", "task_decision", "soft", 30 * 24 * 3600),
        ):
            value = state.get(key)
            if not value:
                continue
            if isinstance(value, list):
                summary_value = "，".join(
                    str(item) for item in value if str(item).strip()
                )
            else:
                summary_value = str(value).strip()
            if not summary_value:
                continue
            self._insert_or_merge(
                MemoryRecord(
                    memory_type=memory_type,
                    key=key,
                    value=value,
                    summary=f"{summary_prefix}：{summary_value}",
                    tier=tier,
                    scope="chat",
                    scope_id=chat_id,
                    confidence=1.0,
                    source_ids=source_ids,
                    last_observed_at=now,
                    expires_at=now + expiry_seconds,
                    tags=("task", key),
                )
            )

    def observe_runtime_event(self, chat_id: str, payload: dict[str, Any]) -> None:
        event_name = str(payload.get("event", "") or "").strip().lower()
        event_payload = dict(payload.get("payload") or {})
        if event_name == "succeeded":
            self._observe_success_event(chat_id, event_payload)
            return
        if event_name not in {"failed", "dead_lettered", "stalled"}:
            return
        description = str(event_payload.get("description", "") or "").strip()
        error = str(event_payload.get("error", "") or "").strip()
        if not description and not error:
            return
        now = time.time()
        summary = f"最近失败：{description}".strip("：")
        if error:
            summary = f"{summary}（{error}）"
        source_ids = tuple(
            int(item) for item in (payload.get("source_ids") or ()) if str(item).strip()
        )
        self._insert_or_merge(
            MemoryRecord(
                memory_type="task_state",
                key="last_failure",
                value={"event": event_name, "description": description, "error": error},
                summary=summary,
                tier="ephemeral",
                scope="chat",
                scope_id=chat_id,
                confidence=1.0,
                source_ids=source_ids,
                last_observed_at=now,
                expires_at=now + 24 * 3600,
                tags=("task", "failure"),
            )
        )

    def _observe_success_event(self, chat_id: str, payload: dict[str, Any]) -> None:
        description = str(payload.get("description", "") or "").strip()
        output = str(payload.get("output", "") or "").strip()
        if not description and not output:
            return
        now = time.time()
        summary = f"最近完成：{description}".strip("：")
        output_preview = output[:120].strip()
        if output_preview:
            summary = f"{summary}（{output_preview}）"
        self._insert_or_merge(
            MemoryRecord(
                memory_type="task_state",
                key="last_success",
                value={"description": description, "output": output},
                summary=summary,
                tier="ephemeral",
                scope="chat",
                scope_id=chat_id,
                confidence=0.92,
                last_observed_at=now,
                expires_at=now + 24 * 3600,
                tags=("task", "success"),
            )
        )


def _extract_brief_phrase(text: str, *, prefix: str) -> str:
    if prefix not in text:
        return ""
    suffix = text.split(prefix, 1)[1]
    candidate = re.split(r"[，。,；;\n]", suffix, maxsplit=1)[0].strip()
    candidate = candidate.strip("：:,.，。；; ")
    if not candidate:
        return ""
    if len(candidate) > 24:
        return ""
    if any(marker in candidate for marker in ("请", "需要", "希望", "回复", "回答")):
        return ""
    return candidate
def _detect_language_preference(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("english", "英文", "英语")):
        return "en-US"
    if any(token in text for token in ("中文", "汉语")):
        return "zh-CN"
    return ""


def _detect_response_style(text: str) -> str:
    if any(token in text for token in ("详细", "展开")):
        return "detailed"
    if any(token in text for token in ("简洁", "精简")):
        return "concise"
    return ""
