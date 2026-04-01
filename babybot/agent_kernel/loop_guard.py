"""Loop guard helpers for single-agent tool execution."""

from __future__ import annotations

import hashlib
import re
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
    max_exploration_streak: int = 6


@dataclass(frozen=True)
class LoopVerdict:
    """Loop guard decision."""

    blocked: bool = False
    reason: str = ""
    disable_tool: bool = False


class LoopGuard:
    """Detect repeated tool-call patterns and context overflow."""

    def __init__(self, config: LoopGuardConfig) -> None:
        self._config = config
        self._call_counts: dict[str, int] = {}
        self._per_tool_counts: dict[str, int] = {}
        self._tool_sequence: deque[str] = deque(
            maxlen=max(2, config.ping_pong_window * 2)
        )
        self._exploration_streak = 0

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
                    f"(max {self._config.max_identical_calls}). "
                    "Wait for new state or choose a different next step before retrying."
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
                disable_tool=True,
            )

        if self.is_exploration_call(tool_name, arguments):
            self._exploration_streak += 1
            if (
                self._config.max_exploration_streak > 0
                and self._exploration_streak >= self._config.max_exploration_streak
            ):
                return LoopVerdict(
                    blocked=True,
                    reason=(
                        "Exploration-only tool pattern repeated too many times "
                        f"({self._exploration_streak})."
                    ),
                )
        else:
            self._exploration_streak = 0

        self._tool_sequence.append(f"{tool_name}:{digest}")
        if len(self._tool_sequence) >= max(4, self._config.ping_pong_window):
            if self._detect_ping_pong():
                return LoopVerdict(
                    blocked=True,
                    reason="Repetitive tool-calling pattern detected.",
                    disable_tool=True,
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

    @staticmethod
    def is_exploration_tool(tool_name: str) -> bool:
        normalized = (tool_name or "").strip().lower().lstrip("_")
        if not normalized:
            return False
        if normalized.endswith("execute_shell_command"):
            return True
        leaf = normalized.split("__")[-1]
        markers = (
            "view_",
            "read_",
            "list_",
            "inspect_",
            "check_",
            "take_snapshot",
            "take_screenshot",
            "evaluate_script",
            "get_console",
            "get_network",
        )
        return leaf.startswith(markers) or any(
            f"_{marker}" in leaf for marker in markers
        )

    @classmethod
    def _is_read_only_shell_command(cls, command: str) -> bool:
        lowered = str(command or "").strip().lower()
        if not lowered:
            return True
        if any(hint in lowered for hint in cls._WRITE_SHELL_HINTS):
            return False
        return any(re.search(pattern, lowered) for pattern in cls._READ_ONLY_SHELL_PATTERNS)

    @classmethod
    def is_exploration_call(cls, tool_name: str, arguments: Any) -> bool:
        normalized = (tool_name or "").strip().lower().lstrip("_")
        if not normalized:
            return False
        if normalized.endswith("execute_shell_command"):
            if isinstance(arguments, dict):
                return cls._is_read_only_shell_command(
                    str(arguments.get("command", "") or "")
                )
            return True
        if normalized.endswith("execute_python_code"):
            if isinstance(arguments, dict):
                return cls._is_read_only_python_code(
                    str(arguments.get("code", "") or "")
                )
            return True
        return cls.is_exploration_tool(tool_name)

    _READ_ONLY_SHELL_PATTERNS = (
        r"^\s*(ls|pwd|whoami|which|type|cat|head|tail|grep|rg|find|tree|wc|stat|echo)\b",
        r"^\s*git\s+(status|log|show|diff|branch)\b",
    )
    _READ_ONLY_PYTHON_PATTERNS = (
        ".read_text(",
        ".read_bytes(",
        "print(",
        "json.dumps(",
        "os.listdir(",
        "glob(",
        "iterdir(",
    )
    _WRITE_SHELL_HINTS = (
        " >",
        ">>",
        "tee ",
        "sed -i",
        "mv ",
        "cp ",
        "rm ",
        "mkdir ",
        "touch ",
        "truncate ",
        "git add",
        "git commit",
        "git apply",
        "chmod ",
        "chown ",
    )
    _WRITE_PYTHON_HINTS = (
        ".write_text(",
        ".write_bytes(",
        ".mkdir(",
        ".unlink(",
        ".rename(",
        ".replace(",
        "shutil.move(",
        "shutil.copy(",
    )

    @classmethod
    def _is_read_only_python_code(cls, code: str) -> bool:
        lowered = str(code or "").strip().lower()
        if not lowered:
            return True
        if ".write_text(" in lowered or ".write_bytes(" in lowered:
            return False
        if any(hint in lowered for hint in ("'w'", '"w"', "'a'", '"a"', "'x'", '"x"')):
            return False
        if any(hint in lowered for hint in cls._WRITE_PYTHON_HINTS):
            return False
        return any(pattern in lowered for pattern in cls._READ_ONLY_PYTHON_PATTERNS)
