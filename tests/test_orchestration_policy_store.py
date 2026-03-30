from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from babybot.orchestration_policy_store import OrchestrationPolicyStore
from babybot.orchestration_router import build_routing_intent_bucket


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


def test_policy_store_recommends_router_timeout_from_recent_model_runs(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    for idx, latency_ms in enumerate((180.0, 220.0, 240.0, 200.0, 260.0), start=1):
        store.record_runtime_telemetry(
            flow_id=f"flow-{idx}",
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=latency_ms,
            router_fallback=False,
            router_source="model",
        )
    store.record_runtime_telemetry(
        flow_id="flow-rule",
        chat_key="feishu:c1",
        route_mode="answer",
        router_model="mini-router",
        router_latency_ms=1.0,
        router_fallback=False,
        router_source="rule",
    )

    recommendation = store.recommend_router_timeout(base_timeout=2.0)

    assert recommendation["samples"] == 5
    assert recommendation["router_source"] == "model_recent"
    assert 0.5 <= recommendation["timeout_seconds"] < 2.0


def test_policy_store_summarizes_reflection_dimension_rates(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_runtime_telemetry(
        flow_id="flow-1",
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        router_model="mini-router",
        router_latency_ms=120.0,
        router_fallback=False,
        router_source="reflection",
        reflection_hint_count=2,
        reflection_override_count=2,
        execution_style_reflection_count=1,
        parallelism_reflection_count=1,
        worker_reflection_count=0,
    )
    store.record_runtime_telemetry(
        flow_id="flow-2",
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        router_model="mini-router",
        router_latency_ms=180.0,
        router_fallback=False,
        router_source="model",
        reflection_hint_count=1,
        reflection_override_count=1,
        execution_style_reflection_count=0,
        parallelism_reflection_count=0,
        worker_reflection_count=1,
    )

    summary = store.summarize_runtime_telemetry()

    assert summary["overall"]["execution_style_reflection_rate"] == 0.5
    assert summary["overall"]["parallelism_reflection_rate"] == 0.5
    assert summary["overall"]["worker_reflection_rate"] == 0.5


def test_policy_store_recommends_reflection_guardrails(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    for idx in range(8):
        store.record_runtime_telemetry(
            flow_id=f"flow-{idx}",
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=120.0 + idx,
            router_fallback=False,
            router_source="model",
            execution_style_reflection_count=0,
            parallelism_reflection_count=1 if idx < 6 else 0,
            worker_reflection_count=0,
        )

    guardrails = store.recommend_reflection_guardrails(chat_key="feishu:c1")

    assert guardrails["samples"] == 8
    assert guardrails["execution_style"]["injection_level"] == "reduced"
    assert guardrails["parallelism"]["soften_default"] is True
    assert guardrails["worker"]["injection_level"] == "reduced"


def test_policy_store_summarizes_guardrail_hit_rates(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_runtime_telemetry(
        flow_id="flow-1",
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        router_model="mini-router",
        router_latency_ms=120.0,
        router_fallback=False,
        router_source="model",
        execution_style_guardrail_reduce_count=1,
        parallelism_guardrail_soften_count=1,
        worker_guardrail_soften_count=0,
    )
    store.record_runtime_telemetry(
        flow_id="flow-2",
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        router_model="mini-router",
        router_latency_ms=180.0,
        router_fallback=False,
        router_source="model",
        execution_style_guardrail_reduce_count=0,
        parallelism_guardrail_soften_count=0,
        worker_guardrail_soften_count=1,
    )

    summary = store.summarize_runtime_telemetry()

    assert summary["overall"]["execution_style_guardrail_reduce_rate"] == 0.5
    assert summary["overall"]["parallelism_guardrail_soften_rate"] == 0.5
    assert summary["overall"]["worker_guardrail_soften_rate"] == 0.5


def test_policy_store_recommends_route_from_clean_success_reflections(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    state_features = {
        "task_shape": "single_step",
        "has_media": False,
        "independent_subtasks": 1,
    }
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features=state_features,
        failure_pattern="clean_success",
        recommended_action="direct_execute",
        confidence=0.62,
    )
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features=state_features,
        failure_pattern="clean_success",
        recommended_action="direct_execute",
        confidence=0.68,
    )

    recommendation = store.recommend_route_from_reflections(
        chat_key="feishu:c1",
        state_features=state_features,
    )

    assert recommendation["route_mode"] == "tool_workflow"
    assert recommendation["recommended_action"] == "direct_execute"
    assert recommendation["samples"] == 2


def test_policy_store_recommends_route_from_stable_intent_bucket(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    intent_bucket = build_routing_intent_bucket("这个该怎么办", has_media=False)
    for idx in range(3):
        flow_id = f"flow-{idx}"
        store.record_runtime_telemetry(
            flow_id=flow_id,
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=100.0 + idx,
            router_fallback=False,
            router_source="model",
            execution_style="direct_execute",
            intent_bucket=intent_bucket,
        )
        store.record_outcome(
            flow_id=flow_id,
            chat_key="feishu:c1",
            final_status="succeeded",
            reward=0.9,
            outcome={"task_result_count": 1},
        )
    store.record_runtime_telemetry(
        flow_id="flow-other",
        chat_key="feishu:c1",
        route_mode="debate",
        router_model="mini-router",
        router_latency_ms=121.0,
        router_fallback=False,
        router_source="model",
        execution_style="analyze_first",
        intent_bucket=intent_bucket,
    )
    store.record_outcome(
        flow_id="flow-other",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.4,
        outcome={"task_result_count": 1},
    )

    recommendation = store.recommend_route_from_intent_bucket(
        chat_key="feishu:c1",
        intent_bucket=intent_bucket,
    )

    assert recommendation is not None
    assert recommendation["route_mode"] == "tool_workflow"
    assert recommendation["execution_style"] == "direct_execute"
    assert recommendation["samples"] == 4
    assert recommendation["wins"] == pytest.approx(3.0, abs=1e-5)


def test_policy_store_suppresses_stale_intent_bucket_successes(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    intent_bucket = build_routing_intent_bucket("这个该怎么办", has_media=False)
    for idx in range(3):
        flow_id = f"flow-{idx}"
        store.record_runtime_telemetry(
            flow_id=flow_id,
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=100.0 + idx,
            router_fallback=False,
            router_source="model",
            execution_style="direct_execute",
            intent_bucket=intent_bucket,
            created_at="2020-01-01T00:00:00+00:00",
        )
        store.record_outcome(
            flow_id=flow_id,
            chat_key="feishu:c1",
            final_status="succeeded",
            reward=0.9,
            outcome={"task_result_count": 1},
            created_at="2020-01-01T00:00:00+00:00",
        )

    recommendation = store.recommend_route_from_intent_bucket(
        chat_key="feishu:c1",
        intent_bucket=intent_bucket,
    )

    assert recommendation is None


def test_policy_store_suppresses_intent_bucket_after_shadow_disagreement(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    intent_bucket = build_routing_intent_bucket("这个该怎么办", has_media=False)
    for idx in range(3):
        flow_id = f"flow-{idx}"
        store.record_runtime_telemetry(
            flow_id=flow_id,
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=100.0 + idx,
            router_fallback=False,
            router_source="model",
            execution_style="direct_execute",
            intent_bucket=intent_bucket,
            shadow_routing_eval_count=1,
            shadow_routing_agree_count=0,
        )
        store.record_outcome(
            flow_id=flow_id,
            chat_key="feishu:c1",
            final_status="succeeded",
            reward=0.9,
            outcome={"task_result_count": 1},
        )

    recommendation = store.recommend_route_from_intent_bucket(
        chat_key="feishu:c1",
        intent_bucket=intent_bucket,
    )

    assert recommendation is None


def test_policy_store_recommends_shadow_budget_probe_after_low_agreement(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    intent_bucket = build_routing_intent_bucket("这个该怎么办", has_media=False)
    for idx in range(2):
        store.record_runtime_telemetry(
            flow_id=f"flow-{idx}",
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=100.0 + idx,
            router_fallback=False,
            router_source="intent_cache",
            execution_style="direct_execute",
            intent_bucket=intent_bucket,
            shadow_routing_eval_count=1,
            shadow_routing_agree_count=0,
        )

    budget = store.recommend_shadow_routing_budget(
        chat_key="feishu:c1",
        intent_bucket=intent_bucket,
    )

    assert budget["enabled"] is True
    assert budget["mode"] == "probe"


def test_policy_store_recommends_shadow_budget_suppress_between_probes(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    intent_bucket = build_routing_intent_bucket("这个该怎么办", has_media=False)
    for idx in range(3):
        store.record_runtime_telemetry(
            flow_id=f"flow-{idx}",
            chat_key="feishu:c1",
            route_mode="tool_workflow",
            router_model="mini-router",
            router_latency_ms=100.0 + idx,
            router_fallback=False,
            router_source="intent_cache",
            execution_style="direct_execute",
            intent_bucket=intent_bucket,
            shadow_routing_eval_count=1,
            shadow_routing_agree_count=0,
        )

    budget = store.recommend_shadow_routing_budget(
        chat_key="feishu:c1",
        intent_bucket=intent_bucket,
    )

    assert budget["enabled"] is False
    assert budget["mode"] == "suppressed"


def test_policy_store_reflection_route_recommendation_decays_stale_successes(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    state_features = {
        "task_shape": "single_step",
        "has_media": False,
        "independent_subtasks": 1,
    }
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features=state_features,
        failure_pattern="clean_success",
        recommended_action="direct_execute",
        confidence=0.9,
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features=state_features,
        failure_pattern="clean_success",
        recommended_action="analyze_first",
        confidence=0.66,
        created_at="2026-03-30T00:00:00+00:00",
    )
    store.record_reflection(
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        state_features=state_features,
        failure_pattern="clean_success",
        recommended_action="analyze_first",
        confidence=0.64,
        created_at="2026-03-30T00:10:00+00:00",
    )

    recommendation = store.recommend_route_from_reflections(
        chat_key="feishu:c1",
        state_features=state_features,
    )

    assert recommendation["recommended_action"] == "analyze_first"
    assert recommendation["min_samples_required"] == 2
    assert 1.0 < recommendation["effective_samples"] < recommendation["samples"]


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


def test_policy_store_summarizes_runtime_telemetry_and_reward(tmp_path) -> None:
    store = OrchestrationPolicyStore(tmp_path / "policy.db")
    store.record_runtime_telemetry(
        flow_id="flow-1",
        chat_key="feishu:c1",
        route_mode="tool_workflow",
        router_model="mini-router",
        router_latency_ms=180.0,
        router_fallback=False,
        reflection_hint_count=1,
        reflection_override_count=1,
    )
    store.record_outcome(
        flow_id="flow-1",
        chat_key="feishu:c1",
        final_status="succeeded",
        reward=0.8,
        outcome={},
    )
    store.record_runtime_telemetry(
        flow_id="flow-2",
        chat_key="feishu:c1",
        route_mode="debate",
        router_model="mini-router",
        router_latency_ms=320.0,
        router_fallback=True,
        reflection_hint_count=0,
        reflection_override_count=0,
    )
    store.record_outcome(
        flow_id="flow-2",
        chat_key="feishu:c1",
        final_status="failed",
        reward=-0.4,
        outcome={},
    )

    summary = store.summarize_runtime_telemetry()

    assert summary["overall"]["runs"] == 2
    assert summary["overall"]["fallback_rate"] == 0.5
    assert summary["overall"]["reflection_match_rate"] == 0.5
    assert summary["overall"]["reflection_override_rate"] == 0.5
    assert summary["by_route_mode"]["tool_workflow"]["mean_reward"] == 0.8
    assert summary["by_route_mode"]["debate"]["mean_reward"] == -0.4
