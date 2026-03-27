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


def test_policy_store_summarizes_risk_metrics_by_action(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="serial_dispatch",
        state_features={"task_shape": "multi_step"},
    )
    store.record_decision(
        flow_id="flow-2",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="serial_dispatch",
        state_features={"task_shape": "multi_step"},
    )
    store.record_outcome(
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="failed",
        reward=-1.0,
        outcome={
            "retry_count": 2,
            "dead_letter_count": 1,
            "stalled_count": 0,
        },
    )
    store.record_outcome(
        flow_id="flow-2",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=1.0,
        outcome={
            "retry_count": 0,
            "dead_letter_count": 0,
            "stalled_count": 1,
        },
    )

    stats = store.summarize_action_stats(decision_kind="scheduling")

    assert stats["serial_dispatch"]["samples"] == 2
    assert stats["serial_dispatch"]["failure_rate"] == 0.5
    assert stats["serial_dispatch"]["success_rate"] == 0.5
    assert stats["serial_dispatch"]["retry_rate"] == 1.0
    assert stats["serial_dispatch"]["dead_letter_rate"] == 0.5
    assert stats["serial_dispatch"]["stalled_rate"] == 0.5


def test_policy_store_includes_feedback_score_in_action_summary(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="serial_dispatch",
        state_features={"task_shape": "multi_step"},
    )
    store.record_outcome(
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.8,
        outcome={},
    )
    store.record_feedback(
        flow_id="flow-1",
        chat_key="feishu:c1",
        rating="good",
        reason="稳定",
    )
    store.record_feedback(
        flow_id="flow-1",
        chat_key="feishu:c1",
        rating="good",
        reason="可控",
    )

    stats = store.summarize_action_stats(decision_kind="scheduling")

    assert stats["serial_dispatch"]["feedback_good_count"] == 2
    assert stats["serial_dispatch"]["feedback_bad_count"] == 0
    assert stats["serial_dispatch"]["feedback_score"] > 0.0


def test_policy_store_filters_action_stats_by_bucket(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
        chat_key="feishu:c1",
        decision_kind="decomposition",
        action_name="analyze_then_execute",
        state_features={
            "task_shape": "multi_step",
            "has_media": False,
            "independent_subtasks": 1,
        },
    )
    store.record_outcome(
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.9,
        outcome={},
    )
    store.record_decision(
        flow_id="flow-2",
        chat_key="feishu:c1",
        decision_kind="decomposition",
        action_name="retrieve_then_execute",
        state_features={
            "task_shape": "single_step",
            "has_media": True,
            "independent_subtasks": 1,
        },
    )
    store.record_outcome(
        flow_id="flow-2",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.6,
        outcome={},
    )

    stats = store.summarize_action_stats(
        decision_kind="decomposition",
        state_bucket="task_shape=multi_step|has_media=0|subtasks=1",
    )

    assert set(stats) == {"analyze_then_execute"}
