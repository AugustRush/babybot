from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InteractiveReply:
    text: str = ""
    media_paths: list[str] = field(default_factory=list)


@dataclass
class InteractiveSession:
    session_id: str
    chat_key: str
    backend_name: str
    started_at: float
    last_active_at: float
    handle: Any


@dataclass
class InteractiveSessionStatus:
    session_id: str
    chat_key: str
    backend_name: str
    started_at: float
    last_active_at: float
    backend_status: dict[str, Any] = field(default_factory=dict)
