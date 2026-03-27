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
    ) -> dict[str, dict[str, float | int]]:
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
