from __future__ import annotations

from babybot.agent_kernel.loop_guard import LoopGuard, LoopGuardConfig
from babybot.agent_kernel.model import ModelMessage, ModelToolCall


def test_compress_messages_fallback_does_not_start_with_tool_message() -> None:
    guard = LoopGuard(LoopGuardConfig(max_context_messages=3))
    messages = [
        ModelMessage(role="system", content="sys"),
        ModelMessage(role="user", content="u1"),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(ModelToolCall(call_id="c1", name="tool", arguments={}),),
        ),
        ModelMessage(role="tool", content="tool result", tool_call_id="c1"),
        ModelMessage(role="user", content="u2"),
        ModelMessage(role="assistant", content="a2"),
    ]

    compressed = guard.compress_messages(messages)
    non_system = [m for m in compressed if m.role != "system"]

    assert non_system
    assert non_system[0].role != "tool"
