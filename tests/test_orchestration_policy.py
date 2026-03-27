from __future__ import annotations

from collections.abc import Mapping

from babybot.orchestration_policy import ConservativePolicySelector


class _FakePolicyStore:
    def __init__(self, stats: dict[str, dict[str, float | int]]) -> None:
        self._stats = stats

    def summarize_action_stats(
        self,
        *,
        decision_kind: str | None = None,
        state_bucket: str | None = None,
    ) -> dict[str, dict[str, float | int]]:
        if (
            decision_kind
            and state_bucket
            and isinstance(self._stats.get((decision_kind, state_bucket)), Mapping)
        ):
            value = self._stats[(decision_kind, state_bucket)]
            if all(isinstance(item, Mapping) for item in value.values()):
                return dict(value)
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
