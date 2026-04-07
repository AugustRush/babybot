"""Pure utility functions shared by OrchestrationPolicyStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# Defaults mirroring OrchestrationPolicyStore class constants
_OUTCOME_HALF_LIFE_DAYS = 21.0
_RECENT_WINDOW_DAYS = 7.0


def utc_now() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decay_weight(
    created_at: datetime,
    *,
    now: datetime,
    half_life_days: float | None = None,
) -> float:
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    half_life = half_life_days or _OUTCOME_HALF_LIFE_DAYS
    return 0.5 ** (age_days / half_life)


def parse_time(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        value = raw
    else:
        value = datetime.fromisoformat(str(raw))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_recent(created_at: datetime, *, now: datetime) -> bool:
    return (now - created_at).total_seconds() <= _RECENT_WINDOW_DAYS * 86400.0


def decode_json_object(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}
