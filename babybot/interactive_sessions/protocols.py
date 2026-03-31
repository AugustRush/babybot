from __future__ import annotations

from typing import Any, Protocol

from .types import (
    InteractiveOutputCallback,
    InteractiveReply,
    InteractiveRequest,
)


class InteractiveBackend(Protocol):
    async def start(self, chat_key: str) -> Any: ...

    async def send(
        self,
        handle: Any,
        message: InteractiveRequest,
        *,
        output_event_callback: InteractiveOutputCallback | None = None,
    ) -> InteractiveReply: ...

    async def stop(self, handle: Any, reason: str = "user_stop") -> None: ...

    def status(self, handle: Any) -> dict[str, Any]: ...
