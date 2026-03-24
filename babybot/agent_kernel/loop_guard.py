"""Loop guard helpers for single-agent tool execution."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any

from .model import ModelMessage


@dataclass(frozen=True)
class LoopGuardConfig:
    """Configuration for runtime loop protection."""

    enabled: bool = True
    max_identical_calls: int = 3
    ping_pong_window: int = 6
    per_tool_call_budget: int = 20
    max_context_messages: int = 100


@dataclass(frozen=True)
class LoopVerdict:
    """Loop guard decision."""

    blocked: bool = False
    reason: str = ""


class LoopGuard:
    """Detect repeated tool-call patterns and context overflow."""

    def __init__(self, config: LoopGuardConfig) -> None:
        self._config = config
        self._call_counts: dict[str, int] = {}
        self._per_tool_counts: dict[str, int] = {}
        self._tool_sequence: deque[str] = deque(
            maxlen=max(2, config.ping_pong_window * 2)
        )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def check_call(self, tool_name: str, arguments: Any) -> LoopVerdict:
        if not self._config.enabled:
            return LoopVerdict()

        digest = self._call_digest(tool_name, arguments)
        self._call_counts[digest] = self._call_counts.get(digest, 0) + 1
        if self._call_counts[digest] > self._config.max_identical_calls:
            return LoopVerdict(
                blocked=True,
                reason=(
                    f"Identical call to '{tool_name}' repeated "
                    f"{self._call_counts[digest]} times "
                    f"(max {self._config.max_identical_calls})."
                ),
            )

        self._per_tool_counts[tool_name] = self._per_tool_counts.get(tool_name, 0) + 1
        if self._per_tool_counts[tool_name] > self._config.per_tool_call_budget:
            return LoopVerdict(
                blocked=True,
                reason=(
                    f"Tool '{tool_name}' exceeded call budget "
                    f"({self._config.per_tool_call_budget})."
                ),
            )

        self._tool_sequence.append(f"{tool_name}:{digest}")
        if len(self._tool_sequence) >= max(4, self._config.ping_pong_window):
            if self._detect_ping_pong():
                return LoopVerdict(
                    blocked=True,
                    reason="Repetitive tool-calling pattern detected.",
                )

        return LoopVerdict()

    def compress_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Compact message history when context grows too large.

        Preserves tool_call / tool_result pairing integrity: when truncating,
        the cut point is adjusted so that assistant messages with tool_calls
        are never separated from their corresponding tool result messages.
        """
        if (
            not self._config.enabled
            or len(messages) <= self._config.max_context_messages
        ):
            return messages

        threshold = len(messages) // 2
        compacted: list[ModelMessage] = []
        for idx, msg in enumerate(messages):
            if idx < threshold and msg.role == "tool":
                compacted.append(
                    ModelMessage(
                        role="tool",
                        name=msg.name,
                        tool_call_id=msg.tool_call_id,
                        content="[Tool result truncated]",
                    )
                )
            else:
                compacted.append(msg)

        if len(compacted) <= self._config.max_context_messages:
            return compacted

        system_messages = [m for m in compacted if m.role == "system"]
        non_system = [m for m in compacted if m.role != "system"]
        window = max(0, self._config.max_context_messages - len(system_messages))
        if len(non_system) > window:
            cut = len(non_system) - window
            # Move the cut forward to avoid splitting tool_call / tool_result
            # pairs.  An assistant message with tool_calls must be followed by
            # its tool result messages; a tool result message must be preceded
            # by the assistant message that issued the call.
            while cut < len(non_system):
                msg = non_system[cut]
                if msg.role == "tool":
                    # This tool result belongs to a preceding assistant
                    # tool_call that would be discarded — skip it too.
                    cut += 1
                elif msg.role == "assistant" and msg.tool_calls:
                    # Count the tool_call results that follow this assistant
                    # message; keep them all together.
                    break
                else:
                    break
            non_system = non_system[cut:]
        compacted = [*system_messages, *non_system]

        if len(compacted) <= self._config.max_context_messages:
            return compacted

        system_messages = [m for m in compacted if m.role == "system"]
        tail = [m for m in compacted if m.role != "system"][-4:]
        while tail and tail[0].role == "tool":
            tail = tail[1:]
        return [*system_messages, *tail]

    @staticmethod
    def _call_digest(tool_name: str, arguments: Any) -> str:
        payload = f"{tool_name}:{arguments!r}".encode("utf-8", "ignore")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _detect_ping_pong(self) -> bool:
        seq = list(self._tool_sequence)
        n = len(seq)
        for period in (2, 3):
            span = period * 2
            if n < span:
                continue
            tail = seq[-span:]
            pattern = tail[:period]
            if all(tail[idx] == pattern[idx % period] for idx in range(span)):
                return True
        return False
