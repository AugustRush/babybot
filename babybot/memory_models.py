from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    memory_type: str
    key: str
    value: Any
    summary: str
    tier: str
    scope: str
    scope_id: str
    status: str = "active"
    confidence: float = 1.0
    evidence_count: int = 1
    contradiction_count: int = 0
    source_ids: tuple[int, ...] = ()
    last_observed_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0
    expires_at: float | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "key": self.key,
            "value": self.value,
            "summary": self.summary,
            "tier": self.tier,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "status": self.status,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "source_ids": list(self.source_ids),
            "last_observed_at": self.last_observed_at,
            "last_used_at": self.last_used_at,
            "expires_at": self.expires_at,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "tags": list(self.tags),
            "meta": dict(self.meta),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(payload.get("memory_id", ""))
            or f"mem_{uuid.uuid4().hex[:16]}",
            memory_type=str(payload.get("memory_type", "")),
            key=str(payload.get("key", "")),
            value=payload.get("value"),
            summary=str(payload.get("summary", "")),
            tier=str(payload.get("tier", "soft")),
            scope=str(payload.get("scope", "chat")),
            scope_id=str(payload.get("scope_id", "")),
            status=str(payload.get("status", "active")),
            confidence=float(payload.get("confidence", 1.0) or 0.0),
            evidence_count=int(payload.get("evidence_count", 1) or 1),
            contradiction_count=int(payload.get("contradiction_count", 0) or 0),
            source_ids=tuple(
                int(item)
                for item in (payload.get("source_ids") or ())
                if str(item).strip()
            ),
            last_observed_at=float(payload["last_observed_at"])
            if payload.get("last_observed_at") is not None
            else time.time(),
            last_used_at=float(payload["last_used_at"])
            if payload.get("last_used_at") is not None
            else 0.0,
            expires_at=(
                float(payload.get("expires_at"))
                if payload.get("expires_at") not in (None, "")
                else None
            ),
            supersedes=payload.get("supersedes"),
            superseded_by=payload.get("superseded_by"),
            tags=tuple(
                str(item) for item in (payload.get("tags") or ()) if str(item).strip()
            ),
            meta=dict(payload.get("meta") or {}),
            created_at=float(payload["created_at"])
            if payload.get("created_at") is not None
            else time.time(),
            updated_at=float(payload["updated_at"])
            if payload.get("updated_at") is not None
            else time.time(),
        )

    @classmethod
    def from_row(cls, row: Any) -> "MemoryRecord":
        if hasattr(row, "keys"):
            payload = {key: row[key] for key in row.keys()}
            return cls(
                memory_id=payload["memory_id"],
                memory_type=payload["memory_type"],
                key=payload["key"],
                value=json.loads(payload["value"]),
                summary=payload["summary"],
                tier=payload["tier"],
                scope=payload["scope"],
                scope_id=payload["scope_id"],
                status=payload["status"],
                confidence=float(payload["confidence"]),
                evidence_count=int(payload["evidence_count"]),
                contradiction_count=int(payload["contradiction_count"]),
                source_ids=tuple(json.loads(payload["source_ids"] or "[]")),
                last_observed_at=float(payload["last_observed_at"]),
                last_used_at=float(payload["last_used_at"]),
                expires_at=float(payload["expires_at"]) if payload.get("expires_at") is not None else None,
                supersedes=payload["supersedes"],
                superseded_by=payload["superseded_by"],
                tags=tuple(json.loads(payload["tags"] or "[]")),
                meta=json.loads(payload["meta"] or "{}"),
                created_at=float(payload["created_at"]),
                updated_at=float(payload["updated_at"]),
            )
        return cls(
            memory_id=row[0],
            memory_type=row[1],
            key=row[2],
            value=json.loads(row[3]),
            summary=row[4],
            tier=row[5],
            scope=row[6],
            scope_id=row[7],
            status=row[8],
            confidence=float(row[9]),
            evidence_count=int(row[10]),
            contradiction_count=int(row[11]),
            source_ids=tuple(json.loads(row[12] or "[]")),
            last_observed_at=float(row[13]),
            last_used_at=float(row[14]),
            expires_at=float(row[15]) if row[15] is not None else None,
            supersedes=row[16],
            superseded_by=row[17],
            tags=tuple(json.loads(row[18] or "[]")),
            meta=json.loads(row[19] or "{}"),
            created_at=float(row[20]),
            updated_at=float(row[21]),
        )
