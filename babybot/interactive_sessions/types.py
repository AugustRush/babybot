from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class InteractiveOutputEvent:
    event: str
    text: str = ""
    delta: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractiveReply:
    text: str = ""
    media_paths: list[str] = field(default_factory=list)
    expired: bool = False


@dataclass(frozen=True)
class InteractiveRequest:
    text: str
    media_paths: tuple[str, ...] = ()
    job_id: str = ""
    contract_mode: str = "interactive_session"


@dataclass
class InteractiveSession:
    session_id: str
    chat_key: str
    backend_name: str
    started_at: float
    last_active_at: float
    handle: Any
    mode: str = ""
    runtime_root: str = ""
    process_pid: int | None = None
    last_error: str = ""


@dataclass
class InteractiveSessionStatus:
    session_id: str
    chat_key: str
    backend_name: str
    started_at: float
    last_active_at: float
    mode: str = ""
    runtime_root: str = ""
    process_pid: int | None = None
    last_error: str = ""
    backend_status: dict[str, Any] = field(default_factory=dict)


InteractiveOutputCallback = Callable[
    [InteractiveOutputEvent],
    Awaitable[None] | None,
]
