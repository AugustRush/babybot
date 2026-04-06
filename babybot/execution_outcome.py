"""Pure helpers for projecting execution events/results into policy outcomes."""

from __future__ import annotations

from typing import Any


def summarize_task_results(result: Any | None) -> dict[str, int]:
    task_results = dict(getattr(result, "task_results", {}) or {})
    retry_count = 0
    dead_letter_count = 0
    stalled_count = 0
    tool_call_count = 0
    tool_failure_count = 0
    loop_guard_block_count = 0
    max_step_exhausted_count = 0

    for task_result in task_results.values():
        attempts = max(1, int(getattr(task_result, "attempts", 1) or 1))
        retry_count += max(0, attempts - 1)
        metadata = dict(getattr(task_result, "metadata", {}) or {})
        if metadata.get("dead_lettered") is True:
            dead_letter_count += 1
        error_text = str(getattr(task_result, "error", "") or "").strip().lower()
        if (
            metadata.get("stalled") is True
            or metadata.get("error_type") == "stalled"
            or "heartbeat stalled" in error_text
        ):
            stalled_count += 1
        tool_call_count += max(0, int(metadata.get("tool_call_count", 0) or 0))
        tool_failure_count += max(0, int(metadata.get("tool_failure_count", 0) or 0))
        loop_guard_block_count += max(
            0, int(metadata.get("loop_guard_block_count", 0) or 0)
        )
        max_step_exhausted_count += max(
            0, int(metadata.get("max_step_exhausted_count", 0) or 0)
        )

    return {
        "task_result_count": len(task_results),
        "retry_count": retry_count,
        "dead_letter_count": dead_letter_count,
        "stalled_count": stalled_count,
        "tool_call_count": tool_call_count,
        "tool_failure_count": tool_failure_count,
        "loop_guard_block_count": loop_guard_block_count,
        "max_step_exhausted_count": max_step_exhausted_count,
    }


def build_execution_outcome(
    events: list[dict[str, Any]],
    *,
    result: Any | None = None,
    error: str | None = None,
    execution_elapsed_ms: float | None = None,
) -> dict[str, Any]:
    event_retry_count = sum(1 for event in events if event.get("event") == "retrying")
    event_dead_letter_count = sum(
        1 for event in events if event.get("event") == "dead_lettered"
    )
    event_stalled_count = sum(1 for event in events if event.get("event") == "stalled")
    result_details = summarize_task_results(result)

    payload = {
        "retry_count": max(event_retry_count, int(result_details["retry_count"])),
        "dead_letter_count": max(
            event_dead_letter_count, int(result_details["dead_letter_count"])
        ),
        "stalled_count": max(
            event_stalled_count, int(result_details["stalled_count"])
        ),
        "task_result_count": int(result_details["task_result_count"]),
        "executor_step_count": sum(
            1 for event in events if event.get("event") == "executor.step"
        ),
        "tool_call_count": int(result_details["tool_call_count"]),
        "tool_failure_count": int(result_details["tool_failure_count"]),
        "loop_guard_block_count": int(result_details["loop_guard_block_count"]),
        "max_step_exhausted_count": int(result_details["max_step_exhausted_count"]),
    }
    if execution_elapsed_ms is not None:
        payload["execution_elapsed_ms"] = round(max(0.0, float(execution_elapsed_ms)), 2)
    if error:
        payload["error"] = error
    return payload


def compute_policy_reward(
    events: list[dict[str, Any]],
    *,
    final_status: str,
    result: Any | None = None,
) -> float:
    reward = 1.0 if final_status == "succeeded" else -1.0
    outcome = build_execution_outcome(events, result=result)
    reward -= 0.15 * int(outcome["retry_count"])
    reward -= 0.25 * int(outcome["dead_letter_count"])
    reward -= 0.2 * int(outcome["stalled_count"])
    reward -= 0.08 * int(int(outcome["tool_failure_count"]) > 0)
    reward -= 0.1 * int(int(outcome["loop_guard_block_count"]) > 0)
    reward -= 0.18 * int(int(outcome["max_step_exhausted_count"]) > 0)
    return max(-1.0, min(1.0, reward))
