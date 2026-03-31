from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_BUSY_TIMEOUT_MS = 3000


def connect_sqlite(
    db_path: str | Path,
    *,
    row_factory: object | None = None,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    if row_factory is not None:
        db.row_factory = row_factory
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(f"PRAGMA busy_timeout={max(1, int(busy_timeout_ms))}")
    db.execute("PRAGMA synchronous=NORMAL")
    return db
