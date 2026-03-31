from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from babybot.orchestration_policy import ConservativePolicySelector


class _FakePolicyStore:
    def __init__(self, stats: dict[str, dict[str, float | int]]) -> None:
        self._stats = stats
        self.calls: list[tuple[str | None, str | None]] = []

    def summarize_action_stats(
        self,
        *,
        decision_kind: str | None = None,
        state_bucket: str | None = None,
    ) -> dict[str, dict[str, float | int]]:
        self.calls.append((decision_kind, state_bucket))
        if decision_kind and state_bucket:
            value = self._stats.get((decision_kind, state_bucket))
            if isinstance(value, Mapping) and all(
                isinstance(item, Mapping) for item in value.values()
            ):
                return dict(value)
            return {}
        if decision_kind and isinstance(self._stats.get(decision_kind), Mapping):
            value = self._stats[decision_kind]
            if all(isinstance(item, Mapping) for item in value.values()):
                return dict(value)
        return dict(self._stats)


def test_policy_prefers_historically_safer_decomposition_action() -> None:
    store = _FakePolicyStore(
        {
            "analyze_then_execute": {"mean_reward": 0.91, "samples": 12},
            "direct_execute": {"mean_reward": 0.42, "samples": 12},
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_decomposition(features={"task_shape": "multi_step"})

    assert action.name == "analyze_then_execute"


def test_policy_falls_back_to_safe_default_when_data_is_sparse() -> None:
    selector = ConservativePolicySelector(_FakePolicyStore({}), min_samples=8)

    action = selector.choose_decomposition(features={"task_shape": "unknown"})

    assert action.name == "analyze_then_execute"


def test_policy_prefers_serial_scheduling_when_parallel_history_is_risky() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {"mean_reward": 0.74, "samples": 10, "failure_rate": 0.0},
                "bounded_parallel": {
                    "mean_reward": 0.81,
                    "samples": 10,
                    "failure_rate": 0.45,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_scheduling(features={"independent_subtasks": 3})

    assert action.name == "serial"


def test_policy_denies_worker_when_data_is_sparse() -> None:
    selector = ConservativePolicySelector(_FakePolicyStore({}), min_samples=8)

    action = selector.choose_worker_gate(features={"task_shape": "unknown"})

    assert action.name == "deny_worker"


def test_policy_prefers_action_with_stronger_positive_feedback() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.7,
                    "samples": 12,
                    "feedback_score": 0.25,
                },
                "bounded_parallel": {
                    "mean_reward": 0.72,
                    "samples": 12,
                    "feedback_score": -0.2,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"


def test_policy_uses_bucket_specific_history_before_global_history() -> None:
    store = _FakePolicyStore(
        {
            "decomposition": {
                "analyze_then_execute": {"mean_reward": 0.92, "samples": 18},
                "retrieve_then_execute": {"mean_reward": 0.4, "samples": 18},
            },
            (
                "decomposition",
                "task_shape=single_step|has_media=1|subtasks=1",
            ): {
                "retrieve_then_execute": {"mean_reward": 0.88, "samples": 12},
                "analyze_then_execute": {"mean_reward": 0.5, "samples": 12},
            },
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8)

    action = selector.choose_decomposition(
        features={
            "task_shape": "single_step",
            "has_media": True,
            "independent_subtasks": 1,
        }
    )

    assert action.name == "retrieve_then_execute"


def test_policy_prefers_stabler_action_when_high_reward_action_is_under_sampled() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.71,
                    "samples": 20,
                    "feedback_score": 0.0,
                },
                "bounded_parallel": {
                    "mean_reward": 0.9,
                    "samples": 8,
                    "feedback_score": 0.0,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8, explore_ratio=0.1)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"


def test_policy_auto_mode_uses_internal_conservative_thresholds() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.75,
                    "samples": 12,
                },
                "bounded_parallel": {
                    "mean_reward": 0.95,
                    "samples": 7,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=0, explore_ratio=-1.0)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"


def test_policy_downweights_stale_high_reward_history() -> None:
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.8,
                    "samples": 12,
                    "effective_samples": 1.0,
                    "last_updated_at": now.isoformat(timespec="seconds"),
                },
                "bounded_parallel": {
                    "mean_reward": 1.0,
                    "samples": 12,
                    "effective_samples": 0.12,
                    "last_updated_at": (now - timedelta(days=45)).isoformat(timespec="seconds"),
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"


def test_sparse_feedback_does_not_overcorrect_action_score() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.74,
                    "samples": 12,
                    "feedback_score": 0.0,
                    "feedback_confidence": 0.0,
                },
                "bounded_parallel": {
                    "mean_reward": 0.75,
                    "samples": 12,
                    "feedback_score": -1.0,
                    "feedback_confidence": 0.1,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "bounded_parallel"


def test_policy_falls_back_from_specific_bucket_to_general_bucket() -> None:
    store = _FakePolicyStore(
        {
            "decomposition": {
                "analyze_then_execute": {"mean_reward": 0.9, "samples": 16},
                "retrieve_then_execute": {"mean_reward": 0.4, "samples": 16},
            },
            ("decomposition", "task_shape=single_step|has_media=1"): {
                "retrieve_then_execute": {"mean_reward": 0.85, "samples": 10},
                "analyze_then_execute": {"mean_reward": 0.45, "samples": 10},
            },
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8, explore_ratio=-1.0)

    action = selector.choose_decomposition(
        features={
            "task_shape": "single_step",
            "has_media": True,
            "independent_subtasks": 1,
        }
    )

    assert action.name == "retrieve_then_execute"


def test_policy_prefers_bucket_template_with_stronger_effective_samples() -> None:
    store = _FakePolicyStore(
        {
            ("decomposition", "task_shape=single_step|has_media=1|subtasks=1"): {
                "retrieve_then_execute": {
                    "mean_reward": 0.86,
                    "samples": 8,
                    "effective_samples": 1.2,
                }
            },
            ("decomposition", "task_shape=single_step|has_media=1"): {
                "retrieve_then_execute": {
                    "mean_reward": 0.8,
                    "samples": 14,
                    "effective_samples": 9.5,
                }
            },
            "decomposition": {
                "analyze_then_execute": {
                    "mean_reward": 0.9,
                    "samples": 30,
                    "effective_samples": 30.0,
                }
            },
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8, explore_ratio=-1.0)

    action = selector.choose_decomposition(
        features={
            "task_shape": "single_step",
            "has_media": True,
            "independent_subtasks": 1,
        }
    )

    assert action.name == "retrieve_then_execute"
    assert store.calls[:2] == [
        ("decomposition", "task_shape=single_step|has_media=1|subtasks=1"),
        ("decomposition", "task_shape=single_step|has_media=1"),
    ]


def test_policy_safeguard_downgrades_action_with_recent_failures() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.75,
                    "samples": 12,
                    "effective_samples": 12.0,
                    "recent_failure_rate": 0.0,
                    "recent_guard_samples": 2.0,
                },
                "bounded_parallel": {
                    "mean_reward": 0.9,
                    "samples": 16,
                    "effective_samples": 16.0,
                    "recent_failure_rate": 1.0,
                    "recent_guard_samples": 2.0,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"


def test_policy_prefers_more_discriminative_bucket_template() -> None:
    store = _FakePolicyStore(
        {
            ("decomposition", "task_shape=single_step|has_media=1|subtasks=1"): {
                "retrieve_then_execute": {
                    "mean_reward": 0.81,
                    "samples": 14,
                    "effective_samples": 14.0,
                },
                "analyze_then_execute": {
                    "mean_reward": 0.8,
                    "samples": 14,
                    "effective_samples": 14.0,
                },
            },
            ("decomposition", "task_shape=single_step|has_media=1"): {
                "retrieve_then_execute": {
                    "mean_reward": 0.84,
                    "samples": 9,
                    "effective_samples": 9.0,
                },
                "analyze_then_execute": {
                    "mean_reward": 0.42,
                    "samples": 9,
                    "effective_samples": 9.0,
                },
            },
            "decomposition": {
                "analyze_then_execute": {"mean_reward": 0.9, "samples": 30},
            },
        }
    )
    selector = ConservativePolicySelector(store, min_samples=8, explore_ratio=-1.0)

    action = selector.choose_decomposition(
        features={
            "task_shape": "single_step",
            "has_media": True,
            "independent_subtasks": 1,
        }
    )

    assert action.name == "retrieve_then_execute"


def test_policy_penalizes_action_when_recent_reward_drift_is_high() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.76,
                    "samples": 12,
                    "effective_samples": 12.0,
                    "recent_mean_reward": 0.74,
                    "drift_score": 0.02,
                },
                "bounded_parallel": {
                    "mean_reward": 0.92,
                    "samples": 20,
                    "effective_samples": 20.0,
                    "recent_mean_reward": 0.15,
                    "drift_score": 0.77,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_scheduling(features={"independent_subtasks": 2})

    assert action.name == "serial"


def test_policy_penalizes_parallel_action_with_loop_and_max_step_failures() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.68,
                    "samples": 16,
                    "effective_samples": 16.0,
                    "tool_failure_rate": 0.0,
                    "loop_guard_block_rate": 0.0,
                    "max_step_exhausted_rate": 0.0,
                },
                "bounded_parallel": {
                    "mean_reward": 0.83,
                    "samples": 16,
                    "effective_samples": 16.0,
                    "tool_failure_rate": 0.8,
                    "loop_guard_block_rate": 0.7,
                    "max_step_exhausted_rate": 0.5,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_scheduling(
        features={"task_shape": "multi_step", "independent_subtasks": 3}
    )

    assert action.name == "serial"


def test_policy_prefers_worker_for_multi_step_clean_success_history() -> None:
    store = _FakePolicyStore(
        {
            "worker": {
                "deny_worker": {
                    "mean_reward": 0.61,
                    "samples": 14,
                    "effective_samples": 14.0,
                    "failure_rate": 0.08,
                    "tool_failure_rate": 0.4,
                },
                "allow_worker": {
                    "mean_reward": 0.66,
                    "samples": 14,
                    "effective_samples": 14.0,
                    "failure_rate": 0.0,
                    "tool_failure_rate": 0.0,
                    "max_step_exhausted_rate": 0.0,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_worker_gate(
        features={"task_shape": "multi_step", "independent_subtasks": 3}
    )

    assert action.name == "allow_worker"


def test_policy_penalizes_worker_for_simple_task_even_if_reward_is_higher() -> None:
    store = _FakePolicyStore(
        {
            "worker": {
                "deny_worker": {
                    "mean_reward": 0.65,
                    "samples": 18,
                    "effective_samples": 18.0,
                },
                "allow_worker": {
                    "mean_reward": 0.72,
                    "samples": 18,
                    "effective_samples": 18.0,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    action = selector.choose_worker_gate(
        features={"task_shape": "single_step", "independent_subtasks": 1}
    )

    assert action.name == "deny_worker"


def test_policy_selection_explain_includes_score_breakdown() -> None:
    store = _FakePolicyStore(
        {
            "scheduling": {
                "serial": {
                    "mean_reward": 0.68,
                    "samples": 16,
                    "effective_samples": 16.0,
                    "tool_failure_rate": 0.0,
                    "loop_guard_block_rate": 0.0,
                    "max_step_exhausted_rate": 0.0,
                    "avg_execution_elapsed_ms": 900.0,
                    "avg_tool_call_count": 3.0,
                },
                "bounded_parallel": {
                    "mean_reward": 0.83,
                    "samples": 16,
                    "effective_samples": 16.0,
                    "tool_failure_rate": 0.8,
                    "loop_guard_block_rate": 0.7,
                    "max_step_exhausted_rate": 0.5,
                    "avg_execution_elapsed_ms": 2200.0,
                    "avg_tool_call_count": 9.0,
                },
            }
        }
    )
    selector = ConservativePolicySelector(store, min_samples=1, explore_ratio=-1.0)

    selection = selector.select_scheduling(
        features={"task_shape": "multi_step", "independent_subtasks": 3}
    )

    assert selection.action.name == "serial"
    assert "feature_bias=" in selection.explain
    assert "confidence_penalty=" in selection.explain
    assert "tool_failure_penalty=" in selection.explain
    assert "loop_guard_penalty=" in selection.explain
    assert "max_step_penalty=" in selection.explain
