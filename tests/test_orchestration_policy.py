from __future__ import annotations

from babybot.orchestration_policy import ConservativePolicySelector


class _FakePolicyStore:
    def __init__(self, stats: dict[str, dict[str, float | int]]) -> None:
        self._stats = stats

    def summarize_action_stats(
        self,
        *,
        decision_kind: str | None = None,
    ) -> dict[str, dict[str, float | int]]:
        del decision_kind
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
