from __future__ import annotations

from babybot.orchestration_policy_store import OrchestrationPolicyStore
from babybot.task_evaluator import TaskEvaluationInput, TaskEvaluator


def test_task_evaluator_records_retry_heavy_reflection(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    evaluator = TaskEvaluator(store)

    reflection = evaluator.evaluate(
        TaskEvaluationInput(
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            state_features={
                "task_shape": "multi_step",
                "has_media": False,
                "independent_subtasks": 2,
            },
            execution_style="direct_execute",
            parallelism_hint="bounded_parallel",
            worker_hint="allow",
            final_status="succeeded",
            outcome={
                "retry_count": 2,
                "dead_letter_count": 0,
                "stalled_count": 0,
            },
        )
    )

    assert reflection is not None
    assert reflection.failure_pattern == "retried_too_much"
    assert reflection.recommended_action == "analyze_first"

    hints = store.list_reflection_hints(
        route_mode="tool_workflow",
        state_features={
            "task_shape": "multi_step",
            "has_media": False,
            "independent_subtasks": 2,
        },
    )
    assert hints
    assert hints[0]["recommended_action"] == "analyze_first"


def test_task_evaluator_skips_clean_low_signal_runs(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    evaluator = TaskEvaluator(store)

    reflection = evaluator.evaluate(
        TaskEvaluationInput(
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            state_features={
                "task_shape": "single_step",
                "has_media": False,
                "independent_subtasks": 1,
            },
            execution_style="direct_execute",
            parallelism_hint="serial",
            worker_hint="deny",
            final_status="succeeded",
            outcome={},
        )
    )

    assert reflection is None
    assert store.list_reflection_hints(
        route_mode="tool_workflow",
        state_features={
            "task_shape": "single_step",
            "has_media": False,
            "independent_subtasks": 1,
        },
    ) == []
