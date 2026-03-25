from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Callable

from .protocols import InteractiveBackend
from .types import InteractiveReply, InteractiveSession, InteractiveSessionStatus


class InteractiveSessionManager:
    def __init__(
        self,
        *,
        backends: dict[str, InteractiveBackend],
        max_age_seconds: int = 7200,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._backends = dict(backends)
        self._max_age_seconds = max(1, int(max_age_seconds))
        self._time_fn = time_fn or time.time
        self._sessions: dict[str, InteractiveSession] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def has_active_session(self, chat_key: str) -> bool:
        return chat_key in self._sessions

    async def start(self, *, chat_key: str, backend_name: str) -> InteractiveSession:
        async with self._locks[chat_key]:
            existing = self._sessions.get(chat_key)
            if existing is not None:
                if existing.backend_name != backend_name:
                    raise ValueError("active session backend mismatch; stop it first")
                if self._is_expired(existing):
                    await self._stop_locked(chat_key, reason="expired")
                else:
                    return existing

            backend = self._backends.get(backend_name)
            if backend is None:
                raise ValueError(f"unknown interactive backend: {backend_name}")
            now = float(self._time_fn())
            handle = await backend.start(chat_key)
            session_id = str(getattr(handle, "session_id", "") or "").strip()
            if not session_id:
                session_id = f"{backend_name}:{uuid.uuid4().hex[:12]}"
            session = InteractiveSession(
                session_id=session_id,
                chat_key=chat_key,
                backend_name=backend_name,
                started_at=now,
                last_active_at=now,
                handle=handle,
            )
            self._sessions[chat_key] = session
            return session

    async def send(self, chat_key: str, message: str) -> InteractiveReply:
        async with self._locks[chat_key]:
            session = self._sessions.get(chat_key)
            if session is None:
                raise ValueError(f"no active interactive session for {chat_key}")
            if self._is_expired(session):
                await self._stop_locked(chat_key, reason="expired")
                return InteractiveReply(
                    text="当前交互会话已超时关闭，请重新使用 @session start <backend> 启动。"
                )

            backend = self._backends[session.backend_name]
            reply = await backend.send(session.handle, message)
            session.last_active_at = float(self._time_fn())
            return reply

    async def stop(self, chat_key: str, reason: str = "user_stop") -> bool:
        async with self._locks[chat_key]:
            return await self._stop_locked(chat_key, reason=reason)

    def status(self, chat_key: str) -> InteractiveSessionStatus | None:
        session = self._sessions.get(chat_key)
        if session is None:
            return None
        backend = self._backends.get(session.backend_name)
        backend_status = (
            dict(backend.status(session.handle)) if backend is not None else {}
        )
        return InteractiveSessionStatus(
            session_id=session.session_id,
            chat_key=session.chat_key,
            backend_name=session.backend_name,
            started_at=session.started_at,
            last_active_at=session.last_active_at,
            backend_status=backend_status,
        )

    async def stop_all(self, reason: str = "reset") -> int:
        count = 0
        for chat_key in list(self._sessions):
            if await self.stop(chat_key, reason=reason):
                count += 1
        return count

    def summary(self) -> dict[str, object]:
        return {
            "active_count": len(self._sessions),
            "chat_keys": sorted(self._sessions.keys()),
        }

    def _is_expired(self, session: InteractiveSession) -> bool:
        return (float(self._time_fn()) - float(session.last_active_at)) > float(
            self._max_age_seconds
        )

    async def _stop_locked(self, chat_key: str, reason: str) -> bool:
        session = self._sessions.pop(chat_key, None)
        if session is None:
            return False
        backend = self._backends.get(session.backend_name)
        if backend is not None:
            await backend.stop(session.handle, reason=reason)
        return True
