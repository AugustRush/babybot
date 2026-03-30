from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_policy_store_downweights_stale_outcomes_via_time_decay(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    old = (now - timedelta(days=45)).isoformat(timespec="seconds")
    recent = now.isoformat(timespec="seconds")
    store.record_decision(
        flow_id="flow-old",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="bounded_parallel",
        state_features={"task_shape": "multi_step"},
        created_at=old,
    )
    store.record_outcome(
        flow_id="flow-old",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=1.0,
        outcome={},
        created_at=old,
    )
    store.record_decision(
        flow_id="flow-new",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="serial_dispatch",
        state_features={"task_shape": "multi_step"},
        created_at=recent,
    )
    store.record_outcome(
        flow_id="flow-new",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.8,
        outcome={},
        created_at=recent,
    )

    stats = store.summarize_action_stats(
        decision_kind="scheduling",
        now=now,
    )

    assert stats["bounded_parallel"]["effective_samples"] < 0.5
    assert stats["serial_dispatch"]["effective_samples"] == 1.0


def test_policy_store_decays_feedback_and_tracks_confidence(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    old = (now - timedelta(days=60)).isoformat(timespec="seconds")
    recent = now.isoformat(timespec="seconds")
    store.record_decision(
        flow_id="flow-1",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="serial_dispatch",
        state_features={"task_shape": "multi_step"},
        created_at=recent,
    )
    store.record_outcome(
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.8,
        outcome={},
        created_at=recent,
    )
    store.record_feedback(
        flow_id="flow-1",
        chat_key="feishu:c1",
        rating="bad",
        reason="太激进",
        created_at=old,
    )
    store.record_feedback(
        flow_id="flow-1",
        chat_key="feishu:c1",
        rating="good",
        reason="最近更稳",
        created_at=recent,
    )

    stats = store.summarize_action_stats(decision_kind="scheduling", now=now)

    assert 0.0 < stats["serial_dispatch"]["effective_feedback_samples"] < 2.0
    assert 0.0 < stats["serial_dispatch"]["feedback_confidence"] < 1.0
    assert stats["serial_dispatch"]["feedback_score"] > 0.0


def test_policy_store_matches_generalized_state_bucket(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_decision(
        flow_id="flow-1",
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
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.8,
        outcome={},
    )

    stats = store.summarize_action_stats(
        decision_kind="decomposition",
        state_bucket="task_shape=single_step|has_media=1",
    )

    assert set(stats) == {"retrieve_then_execute"}


def test_policy_store_tracks_recent_failure_window(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    recent = now.isoformat(timespec="seconds")
    old = (now - timedelta(days=30)).isoformat(timespec="seconds")
    store.record_decision(
        flow_id="flow-old",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="bounded_parallel",
        state_features={"task_shape": "multi_step"},
        created_at=old,
    )
    store.record_outcome(
        flow_id="flow-old",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=1.0,
        outcome={},
        created_at=old,
    )
    store.record_decision(
        flow_id="flow-new",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="bounded_parallel",
        state_features={"task_shape": "multi_step"},
        created_at=recent,
    )
    store.record_outcome(
        flow_id="flow-new",
        chat_key="feishu:c1",
        final_status="failed",
        reward=-1.0,
        outcome={},
        created_at=recent,
    )

    stats = store.summarize_action_stats(decision_kind="scheduling", now=now)

    assert stats["bounded_parallel"]["recent_failure_rate"] > 0.5
    assert stats["bounded_parallel"]["recent_guard_samples"] >= 1.0


def test_policy_store_tracks_recent_reward_drift(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    old = (now - timedelta(days=40)).isoformat(timespec="seconds")
    recent = now.isoformat(timespec="seconds")
    for idx in range(3):
        store.record_decision(
            flow_id=f"flow-old-{idx}",
            chat_key="feishu:c1",
            decision_kind="scheduling",
            action_name="bounded_parallel",
            state_features={"task_shape": "multi_step"},
            created_at=old,
        )
        store.record_outcome(
            flow_id=f"flow-old-{idx}",
            chat_key="feishu:c1",
            final_status="succeeded",
            reward=1.0,
            outcome={},
            created_at=old,
        )
    store.record_decision(
        flow_id="flow-new",
        chat_key="feishu:c1",
        decision_kind="scheduling",
        action_name="bounded_parallel",
        state_features={"task_shape": "multi_step"},
        created_at=recent,
    )
    store.record_outcome(
        flow_id="flow-new",
        chat_key="feishu:c1",
        final_status="failed",
        reward=-1.0,
        outcome={},
        created_at=recent,
    )

    stats = store.summarize_action_stats(decision_kind="scheduling", now=now)

    assert stats["bounded_parallel"]["recent_mean_reward"] < 0.3
    assert stats["bounded_parallel"]["drift_score"] > 0.4


def test_policy_store_records_and_queries_reflection_hints(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features={
            "task_shape": "multi_step",
            "has_media": False,
            "independent_subtasks": 2,
        },
        failure_pattern="retried_too_much",
        recommended_action="analyze_first",
        confidence=0.9,
    )
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features={
            "task_shape": "multi_step",
            "has_media": False,
            "independent_subtasks": 2,
        },
        failure_pattern="none",
        recommended_action="bounded_parallel",
        confidence=0.4,
    )

    hints = store.list_reflection_hints(
        route_mode="tool_workflow",
        state_features={
            "task_shape": "multi_step",
            "has_media": False,
            "independent_subtasks": 2,
        },
        limit=1,
    )

    assert len(hints) == 1
    assert hints[0]["recommended_action"] == "analyze_first"
