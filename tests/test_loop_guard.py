from __future__ import annotations

from babybot.agent_kernel.loop_guard import LoopGuard, LoopGuardConfig, LoopVerdict
from babybot.agent_kernel.model import ModelMessage, ModelToolCall


# ---------------------------------------------------------------------------
# Ping-pong detection tests
# ---------------------------------------------------------------------------


def test_ping_pong_same_tool_same_args_blocked() -> None:
    """Identical write->exec->write->exec (same args) should trigger ping-pong."""
    guard = LoopGuard(LoopGuardConfig(ping_pong_window=6, max_identical_calls=10))
    calls = [
        ("write", {"path": "/tmp/a.txt", "content": "hello"}),
        ("exec", {"cmd": "cat /tmp/a.txt"}),
    ]
    # Repeat the pattern — first cycle fills sequence, second cycle triggers detection
    v = LoopVerdict()
    for _cycle in range(3):
        for name, args in calls:
            v = guard.check_call(name, args)
    # After 3 full cycles (6 calls), the last call should have been blocked
    assert v.blocked
    assert "Repetitive" in v.reason


def test_ping_pong_same_tool_different_args_not_blocked() -> None:
    """write(file_a)->exec(cmd_a)->write(file_b)->exec(cmd_b) must NOT trigger
    ping-pong, because the arguments differ each time."""
    guard = LoopGuard(
        LoopGuardConfig(
            ping_pong_window=6, max_identical_calls=10, per_tool_call_budget=20
        )
    )
    # Each iteration uses unique arguments
    for i in range(6):
        v1 = guard.check_call(
            "write", {"path": f"/tmp/file_{i}.txt", "content": f"v{i}"}
        )
        v2 = guard.check_call("exec", {"cmd": f"cat /tmp/file_{i}.txt"})
        assert not v1.blocked, f"write call {i} unexpectedly blocked: {v1.reason}"
        assert not v2.blocked, f"exec call {i} unexpectedly blocked: {v2.reason}"


def test_ping_pong_period3_same_args_blocked() -> None:
    """A-B-C-A-B-C with identical args each time should be caught (period 3)."""
    guard = LoopGuard(LoopGuardConfig(ping_pong_window=6, max_identical_calls=10))
    pattern = [
        ("read", {"file": "x"}),
        ("transform", {"mode": "upper"}),
        ("write", {"file": "x", "content": "X"}),
    ]
    v = LoopVerdict()
    for _cycle in range(3):
        for name, args in pattern:
            v = guard.check_call(name, args)
    assert v.blocked


def test_ping_pong_period3_different_args_not_blocked() -> None:
    """A-B-C repeated but with different args each cycle should NOT trigger."""
    guard = LoopGuard(
        LoopGuardConfig(
            ping_pong_window=6, max_identical_calls=10, per_tool_call_budget=20
        )
    )
    for i in range(4):
        for name in ("read", "transform", "write"):
            v = guard.check_call(name, {"iteration": i})
            assert not v.blocked, f"{name} call {i} unexpectedly blocked: {v.reason}"


# ---------------------------------------------------------------------------
# Identical-call detection tests
# ---------------------------------------------------------------------------


def test_identical_call_blocked_after_max() -> None:
    """Calling the exact same tool+args > max_identical_calls should block."""
    guard = LoopGuard(LoopGuardConfig(max_identical_calls=2, per_tool_call_budget=10))
    args = {"key": "value"}
    assert not guard.check_call("tool_a", args).blocked  # 1
    assert not guard.check_call("tool_a", args).blocked  # 2
    v = guard.check_call("tool_a", args)  # 3 — exceeds max of 2
    assert v.blocked
    assert "Identical call" in v.reason
    assert v.disable_tool is False


def test_identical_name_different_args_not_blocked_by_identical_check() -> None:
    """Same tool name but different args should not count as identical."""
    guard = LoopGuard(LoopGuardConfig(max_identical_calls=2, per_tool_call_budget=10))
    assert not guard.check_call("tool_a", {"v": 1}).blocked
    assert not guard.check_call("tool_a", {"v": 2}).blocked
    assert not guard.check_call("tool_a", {"v": 3}).blocked  # all unique digests


# ---------------------------------------------------------------------------
# Per-tool budget tests
# ---------------------------------------------------------------------------


def test_per_tool_budget_blocks_after_limit() -> None:
    """Exceeding per_tool_call_budget should block regardless of args."""
    guard = LoopGuard(
        LoopGuardConfig(
            per_tool_call_budget=3, max_identical_calls=100, ping_pong_window=100
        )
    )
    for i in range(3):
        assert not guard.check_call("tool_a", {"i": i}).blocked
    v = guard.check_call("tool_a", {"i": 99})  # 4th call, budget is 3
    assert v.blocked
    assert "budget" in v.reason
    assert v.disable_tool is True


# ---------------------------------------------------------------------------
# Disabled guard
# ---------------------------------------------------------------------------


def test_disabled_guard_never_blocks() -> None:
    guard = LoopGuard(LoopGuardConfig(enabled=False))
    for _ in range(100):
        assert not guard.check_call("tool_a", {"same": True}).blocked


def test_exploration_streak_blocks_read_only_wandering() -> None:
    guard = LoopGuard(
        LoopGuardConfig(
            max_identical_calls=20,
            per_tool_call_budget=20,
            ping_pong_window=20,
            max_exploration_streak=4,
        )
    )

    verdict = LoopVerdict()
    for idx in range(4):
        verdict = guard.check_call(
            "_workspace_view_text_file",
            {"file_path": f"skill_{idx}.md"},
        )

    assert verdict.blocked
    assert "exploration" in verdict.reason.lower()


def test_exploration_streak_counts_read_only_shell_commands() -> None:
    guard = LoopGuard(
        LoopGuardConfig(
            max_identical_calls=20,
            per_tool_call_budget=20,
            ping_pong_window=20,
            max_exploration_streak=3,
        )
    )

    verdict = LoopVerdict()
    for idx in range(3):
        verdict = guard.check_call(
            "_workspace_execute_shell_command",
            {"command": f"cat skills/demo_{idx}.md"},
        )

    assert verdict.blocked
    assert "exploration" in verdict.reason.lower()


def test_exploration_streak_does_not_block_shell_write_commands() -> None:
    guard = LoopGuard(
        LoopGuardConfig(
            max_identical_calls=20,
            per_tool_call_budget=20,
            ping_pong_window=20,
            max_exploration_streak=3,
        )
    )

    for idx in range(3):
        verdict = guard.check_call(
            "_workspace_execute_shell_command",
            {"command": f"printf 'x' > /tmp/demo_{idx}.txt"},
        )
        assert not verdict.blocked


# ---------------------------------------------------------------------------
# Compress messages tests
# ---------------------------------------------------------------------------


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
