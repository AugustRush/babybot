from __future__ import annotations

from babybot.orchestration_policy_store import OrchestrationPolicyStore


def test_policy_store_persists_decisions_and_feedback(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
        chat_key="feishu:c1",
        decision_kind="decomposition",
        action_name="analyze_then_execute",
        state_features={"task_shape": "multi_step"},
    )
    store.record_feedback(
        flow_id="flow-1",
        chat_key="feishu:c1",
        rating="good",
        reason="拆分合理",
    )

    row = store.latest_feedback("flow-1")

    assert row is not None
    assert row["rating"] == "good"
    assert row["reason"] == "拆分合理"


def test_policy_store_enables_wal_and_busy_timeout(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")

    assert store.pragma("journal_mode").lower() == "wal"
    assert int(store.pragma("busy_timeout")) >= 3000
