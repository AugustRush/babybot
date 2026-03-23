"""Tape context: Entry + Anchor + Tape + TapeStore (SQLite persistence)."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NO_ANCHOR_RECENT_LIMIT = 200

# CJK Unicode ranges for keyword extraction
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_LATIN_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")

# High-frequency CJK bigrams that carry little semantic value
_CJK_STOPWORDS: set[str] = {
    "的是",
    "是的",
    "了吗",
    "吗？",
    "怎么",
    "什么",
    "这个",
    "那个",
    "一个",
    "不是",
    "没有",
    "可以",
    "我们",
    "他们",
    "她们",
    "你们",
    "已经",
    "就是",
    "还是",
    "但是",
    "因为",
    "所以",
    "如果",
    "虽然",
    "而且",
    "或者",
    "以及",
    "不过",
    "然后",
    "这样",
    "那样",
    "现在",
    "知道",
    "觉得",
    "应该",
    "需要",
    "能够",
    "可能",
    "时候",
    "地方",
}
_LATIN_STOPWORDS: set[str] = {
    "the",
    "is",
    "at",
    "in",
    "on",
    "to",
    "of",
    "and",
    "or",
    "for",
    "it",
    "be",
    "as",
    "by",
    "an",
    "no",
    "do",
    "if",
    "so",
}


def _estimate_token_count(text: str) -> int:
    cjk_chars = sum(1 for ch in text if _CJK_RE.match(ch))
    other_chars = max(0, len(text) - cjk_chars)
    return max(1, math.ceil(cjk_chars / 1.8 + other_chars / 4.0))


def _payload_search_text(kind: str, payload: dict[str, Any]) -> str:
    if kind == "message":
        return str(payload.get("content", "") or "")
    if kind == "anchor":
        state = payload.get("state") or {}
        if not isinstance(state, dict):
            return ""
        parts: list[str] = []
        for key in (
            "summary",
            "user_intent",
            "pending",
            "next_steps",
            "artifacts",
            "open_questions",
            "decisions",
            "entities",
        ):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, list):
                parts.extend(str(item).strip() for item in value if str(item).strip())
        return "\n".join(parts)
    if kind == "tool_result":
        parts = [
            str(payload.get("name", "") or ""),
            str(payload.get("content_preview", "") or ""),
        ]
        artifacts = payload.get("artifacts") or []
        if isinstance(artifacts, list):
            parts.extend(str(item).strip() for item in artifacts if str(item).strip())
        return "\n".join(part for part in parts if part)
    if kind == "tool_call":
        return "\n".join(
            part
            for part in (
                str(payload.get("name", "") or ""),
                json.dumps(payload.get("arguments", {}), ensure_ascii=False),
            )
            if part
        )
    if kind == "event":
        return "\n".join(
            part
            for part in (
                str(payload.get("event", "") or ""),
                json.dumps(payload.get("payload", {}), ensure_ascii=False),
            )
            if part
        )
    return ""


def _extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract search keywords from text.

    For CJK: sliding 2-gram windows (bigrams) over contiguous CJK runs.
    For Latin: whole words (2+ chars).
    Returns deduplicated keywords, most useful first.
    """
    keywords: list[str] = []
    seen: set[str] = set()

    # CJK bigrams
    for match in _CJK_RE.finditer(text):
        segment = match.group()
        for i in range(len(segment) - 1):
            bigram = segment[i : i + 2]
            if bigram not in seen and bigram not in _CJK_STOPWORDS:
                seen.add(bigram)
                keywords.append(bigram)

    # Latin words
    for match in _LATIN_WORD_RE.finditer(text):
        word = match.group().lower()
        if word not in seen and word not in _LATIN_STOPWORDS:
            seen.add(word)
            keywords.append(word)

    return keywords[:max_keywords]


@dataclass(frozen=True)
class Entry:
    """Immutable fact record on the tape."""

    entry_id: int
    kind: str  # "message" | "tool_call" | "tool_result" | "anchor" | "event"
    payload: dict[str, Any]
    meta: dict[str, Any]
    timestamp: float
    _token_est: int = field(default=-1, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token_est < 0:
            est = _estimate_token_count(json.dumps(self.payload, ensure_ascii=False))
            object.__setattr__(self, "_token_est", est)

    @property
    def token_estimate(self) -> int:
        return self._token_est

    @staticmethod
    def message(entry_id: int, role: str, content: str, **meta: Any) -> Entry:
        return Entry(
            entry_id,
            "message",
            {"role": role, "content": content},
            dict(meta),
            time.time(),
        )

    @staticmethod
    def anchor(
        entry_id: int, name: str, state: dict[str, Any] | None = None, **meta: Any
    ) -> Entry:
        return Entry(
            entry_id,
            "anchor",
            {"name": name, "state": state or {}},
            dict(meta),
            time.time(),
        )

    @staticmethod
    def tool_call(
        entry_id: int,
        name: str,
        arguments: dict[str, Any],
        **meta: Any,
    ) -> Entry:
        return Entry(
            entry_id,
            "tool_call",
            {"name": name, "arguments": dict(arguments)},
            dict(meta),
            time.time(),
        )

    @staticmethod
    def tool_result(
        entry_id: int,
        name: str,
        ok: bool,
        content_preview: str = "",
        artifacts: list[str] | None = None,
        **meta: Any,
    ) -> Entry:
        return Entry(
            entry_id,
            "tool_result",
            {
                "name": name,
                "ok": bool(ok),
                "content_preview": content_preview,
                "artifacts": list(artifacts or []),
            },
            dict(meta),
            time.time(),
        )

    @staticmethod
    def event(
        entry_id: int, event: str, payload: dict[str, Any] | None = None, **meta: Any
    ) -> Entry:
        return Entry(
            entry_id,
            "event",
            {"event": event, "payload": dict(payload or {})},
            dict(meta),
            time.time(),
        )


class Tape:
    """Append-only timeline for a single conversation."""

    def __init__(self, chat_id: str, entries: list[Entry] | None = None):
        self.chat_id = chat_id
        self.entries: list[Entry] = list(entries) if entries else []
        self._next_id: int = (self.entries[-1].entry_id + 1) if self.entries else 1

    def append(
        self, kind: str, payload: dict[str, Any], meta: dict[str, Any] | None = None
    ) -> Entry:
        entry = Entry(self._next_id, kind, payload, meta or {}, time.time())
        self.entries.append(entry)
        self._next_id += 1
        return entry

    def last_anchor(self) -> Entry | None:
        for entry in reversed(self.entries):
            if entry.kind == "anchor":
                return entry
        return None

    def entries_since_anchor(self) -> list[Entry]:
        """Return all entries after the most recent anchor."""
        for i in range(len(self.entries) - 1, -1, -1):
            if self.entries[i].kind == "anchor":
                return self.entries[i + 1 :]
        return list(self.entries)

    def total_tokens_since_anchor(self) -> int:
        return sum(e.token_estimate for e in self.entries_since_anchor())

    def turn_count(self) -> int:
        return sum(
            1
            for e in self.entries
            if e.kind == "message" and e.payload.get("role") == "user"
        )

    def compact_entries(self) -> None:
        """Keep only the latest anchor and entries after it in memory."""
        for i in range(len(self.entries) - 1, -1, -1):
            if self.entries[i].kind == "anchor":
                self.entries = self.entries[i:]
                return


class TapeStore:
    """In-memory LRU cache + SQLite persistence for tapes."""

    def __init__(self, db_path: Path, max_chats: int = 500):
        self._db_path = db_path
        self._max_chats = max_chats
        self._cache: OrderedDict[str, Tape] = OrderedDict()
        self._db: sqlite3.Connection | None = None

    def _ensure_db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._db_path))
            self._db.execute("PRAGMA journal_mode=WAL")
            self._create_tables()
        return self._db

    def _create_tables(self) -> None:
        db = self._db
        assert db is not None
        db.executescript("""
            CREATE TABLE IF NOT EXISTS tapes (
                chat_id    TEXT PRIMARY KEY,
                created_at REAL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS entries (
                entry_id  INTEGER NOT NULL,
                chat_id   TEXT NOT NULL,
                kind      TEXT NOT NULL,
                payload   TEXT NOT NULL,
                meta      TEXT DEFAULT '{}',
                content   TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                PRIMARY KEY (chat_id, entry_id)
            );
            CREATE INDEX IF NOT EXISTS idx_entries_kind
                ON entries(chat_id, kind);
        """)
        # Migration: add content column if missing (existing DBs)
        try:
            db.execute("SELECT content FROM entries LIMIT 0")
        except sqlite3.OperationalError:
            db.execute("ALTER TABLE entries ADD COLUMN content TEXT DEFAULT ''")
        # Backfill empty content in batches to avoid OOM on large DBs.
        _BATCH = 500
        while True:
            rows = db.execute(
                "SELECT entry_id, chat_id, kind, payload FROM entries "
                "WHERE content IS NULL OR content = '' "
                "LIMIT ?",
                (_BATCH,),
            ).fetchall()
            if not rows:
                break
            db.executemany(
                "UPDATE entries SET content = ? WHERE chat_id = ? AND entry_id = ?",
                [
                    (
                        _payload_search_text(kind, json.loads(payload)),
                        chat_id,
                        entry_id,
                    )
                    for entry_id, chat_id, kind, payload in rows
                ],
            )
            db.commit()

    def get_or_create(self, chat_id: str) -> Tape:
        if chat_id in self._cache:
            self._touch(chat_id)
            return self._cache[chat_id]
        tape = self._load_from_db(chat_id)
        self._cache[chat_id] = tape
        self._touch(chat_id)
        self._evict_if_needed()
        return tape

    def save_entry(self, chat_id: str, entry: Entry) -> None:
        self.save_entries(chat_id, [entry])

    def save_entries(self, chat_id: str, entries: list[Entry]) -> None:
        if not entries:
            return
        db = self._ensure_db()
        now = time.time()
        db.execute(
            "INSERT OR REPLACE INTO tapes (chat_id, created_at, updated_at) "
            "VALUES (?, COALESCE((SELECT created_at FROM tapes WHERE chat_id=?), ?), ?)",
            (chat_id, chat_id, now, now),
        )
        db.executemany(
            "INSERT INTO entries (entry_id, chat_id, kind, payload, meta, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    entry.entry_id,
                    chat_id,
                    entry.kind,
                    json.dumps(entry.payload, ensure_ascii=False),
                    json.dumps(entry.meta, ensure_ascii=False),
                    _payload_search_text(entry.kind, entry.payload),
                    entry.timestamp,
                )
                for entry in entries
            ],
        )
        db.commit()

    def search_relevant(
        self,
        chat_id: str,
        query: str,
        limit: int = 5,
        exclude_ids: set[int] | None = None,
    ) -> list[Entry]:
        """Search across searchable entries for this chat (cross-anchor recall).

        Uses keyword extraction + LIKE candidate fetching + BM25 ranking.
        Supports both CJK and Latin text without external dependencies.
        """
        query = query.strip()
        if not query:
            return []
        db = self._ensure_db()
        exclude = exclude_ids or set()

        keywords = _extract_keywords(query)
        if not keywords:
            return []

        # Fetch candidates via LIKE on content column (any keyword match)
        conditions = []
        searchable_kinds = ("message", "anchor", "tool_result", "event")
        params: list[str | int] = [chat_id, *searchable_kinds]
        for kw in keywords:
            conditions.append("content LIKE ? ESCAPE '\\'")
            escaped_kw = (
                kw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            params.append(f"%{escaped_kw}%")

        where_clause = " OR ".join(conditions)
        rows = db.execute(
            f"SELECT entry_id, kind, payload, meta, content, timestamp FROM entries "
            f"WHERE chat_id=? AND kind IN (?, ?, ?, ?) AND ({where_clause}) "
            f"ORDER BY entry_id DESC LIMIT ?",
            [*params, limit * 3],
        ).fetchall()

        if not rows:
            return []

        stats_select = ["COUNT(*)", "AVG(LENGTH(content))"]
        stats_params: list[str] = []
        for kw in keywords:
            escaped_kw = (
                kw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            stats_select.append(
                "SUM(CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"
            )
            stats_params.append(f"%{escaped_kw}%")
        stats = db.execute(
            "SELECT " + ", ".join(stats_select) + " FROM entries "
            "WHERE chat_id=? AND kind IN ('message', 'anchor', 'tool_result', 'event')",
            (*stats_params, chat_id),
        ).fetchone()
        total_docs = max(1, stats[0])
        avg_dl = max(1.0, float(stats[1] or 100))

        doc_freq = {kw: int(stats[idx + 2] or 0) for idx, kw in enumerate(keywords)}

        # BM25 scoring (k1=1.5, b=0.75)
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, Entry]] = []
        for r in rows:
            eid = r[0]
            if eid in exclude:
                continue
            entry = Entry(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[5])
            content = r[4] or ""
            dl = len(content)
            score = 0.0
            for kw in keywords:
                tf = content.count(kw)
                if tf == 0:
                    continue
                df = max(1, doc_freq.get(kw, 1))
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _load_from_db(self, chat_id: str) -> Tape:
        db = self._ensure_db()
        # Find the most recent anchor
        row = db.execute(
            "SELECT entry_id FROM entries "
            "WHERE chat_id=? AND kind='anchor' ORDER BY entry_id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()

        if row:
            anchor_id = row[0]
            rows = db.execute(
                "SELECT entry_id, kind, payload, meta, timestamp FROM entries "
                "WHERE chat_id=? AND entry_id>=? ORDER BY entry_id",
                (chat_id, anchor_id),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT entry_id, kind, payload, meta, timestamp FROM entries "
                "WHERE chat_id=? ORDER BY entry_id DESC LIMIT ?",
                (chat_id, _NO_ANCHOR_RECENT_LIMIT),
            ).fetchall()
            rows.reverse()

        entries = [
            Entry(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[4]) for r in rows
        ]
        return Tape(chat_id, entries)

    def _touch(self, chat_id: str) -> None:
        self._cache.move_to_end(chat_id)

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_chats:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._cache.clear()
        if self._db is not None:
            try:
                self._db.close()
            except Exception as exc:
                logger.warning("Failed to close TapeStore DB: %s", exc)
            self._db = None
