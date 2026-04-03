"""Base channel abstraction for BabyBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..orchestrator import TaskResponse


@dataclass
class InboundMessage:
    """Normalized inbound message from any channel."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    media_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChannel(ABC):
    """Abstract base for all channel integrations."""

    name: str = ""
    display_name: str = ""

    def __init__(self, config: Any, manager: Any) -> None:
        self.config = config
        self.manager = manager
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the channel (connect, listen, etc.)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the channel."""
        ...

    @abstractmethod
    async def send_response(
        self, chat_id: str, response: TaskResponse, **kwargs: Any
    ) -> None:
        """Send a TaskResponse to the given chat."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check channel health status. Override in subclasses."""
        return {"status": "unknown", "channel": self.name}

    @property
    def supports_streaming(self) -> bool:
        """Return True if this channel supports live-streaming replies.

        When True and ``create_stream_message`` / ``patch_stream_message``
        are implemented, the MessageBus will use a single unified lifecycle
        card for ACK, animated progress updates, and the final reply.

        Subclasses should override this to return the appropriate value
        based on their own configuration (e.g. ``config.streaming``).
        """
        return False

    # ── Optional card lifecycle methods ──────────────────────────────────
    # Channels that support creating and patching a single message card
    # (e.g. Feishu interactive cards) can override these methods.  When
    # present the MessageBus will use a unified lifecycle card for ACK,
    # progress updates, and final reply instead of separate messages.
    #
    # Default implementations return ``None`` / ``False`` so that the
    # MessageBus duck-type checks (``callable(getattr(channel, ...)``)
    # still work, while channels that do not override simply get no-ops.
    # Channels with real card support (Feishu) already override these and
    # their implementations will take precedence.

    async def create_stream_message(
        self,
        chat_id: str,
        text: str,
        *,
        sender_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Create a patchable message card and return its ID.

        Returns ``None`` if the channel does not support patchable cards.
        Subclasses should override this to actually create a card.
        """
        return None

    async def patch_stream_message(self, message_id: str, text: str) -> bool:
        """Update the content of an existing card message.

        Returns ``True`` on success.  The default returns ``False``.
        """
        return False
