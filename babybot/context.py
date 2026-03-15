"""Tape context: Entry + Anchor + Tape + TapeStore (SQLite persistence)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# CJK Unicode ranges for keyword extraction
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_LATIN_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")


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
            bigram = segment[i:i + 2]
            if bigram not in seen:
                seen.add(bigram)
                keywords.append(bigram)

    # Latin words
    for match in _LATIN_WORD_RE.finditer(text):
        word = match.group().lower()
        if word not in seen:
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

    @property
    def token_estimate(self) -> int:
        text = json.dumps(self.payload, ensure_ascii=False)
        return len(text) // 3

    @staticmethod
    def message(entry_id: int, role: str, content: str, **meta: Any) -> Entry:
        return Entry(entry_id, "message", {"role": role, "content": content}, dict(meta), time.time())

    @staticmethod
    def anchor(entry_id: int, name: str, state: dict[str, Any] | None = None, **meta: Any) -> Entry:
        return Entry(entry_id, "anchor", {"name": name, "state": state or {}}, dict(meta), time.time())


class Tape:
    """Append-only timeline for a single conversation."""

    def __init__(self, chat_id: str, entries: list[Entry] | None = None):
        self.chat_id = chat_id
        self.entries: list[Entry] = list(entries) if entries else []
        self._next_id: int = (self.entries[-1].entry_id + 1) if self.entries else 1

    def append(self, kind: str, payload: dict[str, Any], meta: dict[str, Any] | None = None) -> Entry:
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
                return self.entries[i + 1:]
        return list(self.entries)

    def total_tokens_since_anchor(self) -> int:
        return sum(e.token_estimate for e in self.entries_since_anchor())

    def turn_count(self) -> int:
        return sum(
            1 for e in self.entries
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
        self._cache: dict[str, Tape] = {}
        self._access_order: list[str] = []
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
                timestamp REAL NOT NULL,
                PRIMARY KEY (chat_id, entry_id)
            );
            CREATE INDEX IF NOT EXISTS idx_entries_kind
                ON entries(chat_id, kind);
        """)

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
            "INSERT INTO entries (entry_id, chat_id, kind, payload, meta, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    entry.entry_id,
                    chat_id,
                    entry.kind,
                    json.dumps(entry.payload, ensure_ascii=False),
                    json.dumps(entry.meta, ensure_ascii=False),
                    entry.timestamp,
                )
                for entry in entries
            ],
        )
        db.commit()

    def search_relevant(
        self, chat_id: str, query: str, limit: int = 5,
        exclude_ids: set[int] | None = None,
    ) -> list[Entry]:
        """Search across ALL message entries for this chat (cross-anchor recall).

        Uses keyword extraction + LIKE substring matching to support both
        CJK and Latin text without external dependencies.
        Returns entries ranked by match count, excluding any in *exclude_ids*.
        """
        query = query.strip()
        if not query:
            return []
        db = self._ensure_db()
        exclude = exclude_ids or set()

        # Extract search keywords (2+ char segments for CJK, words for Latin)
        keywords = _extract_keywords(query)
        if not keywords:
            return []

        # Build LIKE conditions — match ANY keyword
        conditions = []
        params: list[str | int] = [chat_id, "message"]
        for kw in keywords:
            conditions.append("payload LIKE ?")
            params.append(f"%{kw}%")

        where_clause = " OR ".join(conditions)
        rows = db.execute(
            f"SELECT entry_id, kind, payload, meta, timestamp FROM entries "
            f"WHERE chat_id=? AND kind=? AND ({where_clause}) "
            f"ORDER BY entry_id DESC LIMIT ?",
            [*params, limit * 3],
        ).fetchall()

        if not rows:
            return []

        # Score by number of keyword hits and filter excludes
        scored: list[tuple[float, Entry]] = []
        for r in rows:
            eid = r[0]
            if eid in exclude:
                continue
            entry = Entry(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[4])
            content = entry.payload.get("content", "")
            hits = sum(1 for kw in keywords if kw in content)
            if hits > 0:
                scored.append((hits, entry))

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
                "WHERE chat_id=? ORDER BY entry_id",
                (chat_id,),
            ).fetchall()

        entries = [
            Entry(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[4])
            for r in rows
        ]
        return Tape(chat_id, entries)

    def _touch(self, chat_id: str) -> None:
        if chat_id in self._access_order:
            self._access_order.remove(chat_id)
        self._access_order.append(chat_id)

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_chats and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
