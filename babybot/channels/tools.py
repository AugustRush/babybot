"""Channel tools protocol and base classes.

Channels expose tools that agents can use for rich media interactions.
"""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_channel_ctx_var: contextvars.ContextVar[ChannelToolContext | None] = contextvars.ContextVar(
    "_channel_ctx", default=None
)


@dataclass
class ChannelCapabilities:
    """Capabilities supported by a channel."""

    supports_text: bool = True
    supports_image: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_file: bool = False
    supports_card: bool = False
    supports_reaction: bool = False

    max_text_length: int = 4096
    max_image_size: int = 20 * 1024 * 1024
    max_file_size: int = 30 * 1024 * 1024
    supported_image_formats: list[str] = field(
        default_factory=lambda: ["png", "jpg", "jpeg", "gif"]
    )
    supported_audio_formats: list[str] = field(
        default_factory=lambda: ["opus", "mp3", "wav"]
    )
    supported_video_formats: list[str] = field(default_factory=lambda: ["mp4", "mov"])


class ChannelToolContext:
    """Runtime context for channel tool execution.

    Set before processing a message so tools know where to send.
    """

    def __init__(
        self,
        chat_id: str,
        channel_name: str = "",
        sender_id: str | None = None,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.channel_name = channel_name
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.message_id = message_id
        self.metadata = metadata or {}

    _current: "ChannelToolContext | None" = None

    @classmethod
    def get_current(cls) -> "ChannelToolContext | None":
        return _channel_ctx_var.get(None)

    @classmethod
    def set_current(cls, ctx: "ChannelToolContext | None") -> None:
        _channel_ctx_var.set(ctx)


class ChannelTools(ABC):
    """Base class for channel-specific tools.

    Implement get_tools() to return tool functions.
    Kernel runtime will extract JSON schema from signatures/docstrings.
    """

    channel_name: str = ""
    channel_display_name: str = ""

    @property
    @abstractmethod
    def capabilities(self) -> ChannelCapabilities:
        """Return channel capabilities."""
        ...

    @abstractmethod
    def get_tools(self) -> list:
        """Return list of tool functions.

        Each function should:
        - Have a descriptive docstring with Args section
        - Return string-like result
        - Use self._get_context() to access current chat info
        """
        ...

    @abstractmethod
    def get_prompt(self) -> str:
        """Return prompt snippet for using this channel."""
        ...

    def get_tool_group_name(self) -> str:
        """Return tool group name for registration."""
        return f"channel_{self.channel_name}"

    def get_tool_group_description(self) -> str:
        """Return tool group description."""
        return f"{self.channel_display_name} 渠道工具"

    def _get_context(self) -> ChannelToolContext | None:
        """Get current execution context."""
        return ChannelToolContext.get_current()
