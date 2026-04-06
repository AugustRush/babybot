from __future__ import annotations

from types import SimpleNamespace

from babybot.agent_kernel import TaskResult
from babybot.execution_outcome import build_execution_outcome, compute_policy_reward


def test_build_execution_outcome_prefers_task_result_metadata_counts() -> None:
    result = SimpleNamespace(
        task_results={
            "task-a": TaskResult(
                task_id="task-a",
                status="failed",
                error="child task heartbeat stalled",
                attempts=3,
                metadata={
                    "dead_lettered": True,
                    "tool_call_count": 5,
                    "tool_failure_count": 2,
                    "loop_guard_block_count": 1,
                    "max_step_exhausted_count": 1,
                },
            )
        }
    )
    events = [{"event": "executor.step"}, {"event": "retrying"}]

    outcome = build_execution_outcome(
        events,
        result=result,
        execution_elapsed_ms=1234.567,
    )

    assert outcome["retry_count"] == 2
    assert outcome["dead_letter_count"] == 1
    assert outcome["stalled_count"] == 1
    assert outcome["task_result_count"] == 1
    assert outcome["executor_step_count"] == 1
    assert outcome["tool_call_count"] == 5
    assert outcome["tool_failure_count"] == 2
    assert outcome["loop_guard_block_count"] == 1
    assert outcome["max_step_exhausted_count"] == 1
    assert outcome["execution_elapsed_ms"] == 1234.57


def test_compute_policy_reward_penalizes_failures_and_limits_range() -> None:
    result = SimpleNamespace(
        task_results={
            "task-a": TaskResult(
                task_id="task-a",
                status="failed",
                error="child task heartbeat stalled",
                attempts=2,
                metadata={
                    "dead_lettered": True,
                    "tool_failure_count": 3,
                    "loop_guard_block_count": 1,
                    "max_step_exhausted_count": 1,
                },
            )
        }
    )
    events = [{"event": "retrying"}, {"event": "dead_lettered"}]

    reward = compute_policy_reward(events, final_status="failed", result=result)

    assert reward >= -1.0
    assert reward <= 1.0
    assert reward < 0
