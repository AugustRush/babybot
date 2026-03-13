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
