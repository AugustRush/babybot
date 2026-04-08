"""InspectService — read-only diagnostic service with zero side effects.

Single Responsibility: Provides observability into runtime state.

Owns the in-memory flow/policy caches and exposes them through structured
read methods.  The orchestrator calls record_flow() after each DAG run to keep
the caches fresh; all other callers only read.

Dependencies (injected, never created here):
  - policy_store  → summarize_action_stats, summarize_runtime_telemetry
  - tape_store    → get_or_create
  - memory_store  → list_memories (via build_context_view)
  - heartbeat_registry → snapshot
  - child_task_bus     → events_for
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .context_views import build_context_view
from .feedback_events import (
    normalize_runtime_feedback_event,
    runtime_event_primary_label,
)
from .orchestration_router import RoutingDecision


class InspectService:
    """Read-only diagnostic service that owns runtime observation caches."""

    _FLOW_CACHE_LIMIT = 256

    def __init__(
        self,
        *,
        policy_store: Any,
        tape_store: Any,
        memory_store: Any,
        heartbeat_registry: Any,
        child_task_bus: Any,
    ) -> None:
        self._policy_store = policy_store
        self._tape_store = tape_store
        self._memory_store = memory_store
        self._heartbeat_registry = heartbeat_registry
        self._child_task_bus = child_task_bus
        # LRU caches — owned here, written by record_flow()
        self._recent_flow_ids_by_chat: OrderedDict[str, str] = OrderedDict()
        self._recent_flows_by_chat: OrderedDict[str, list[str]] = OrderedDict()
        self._recent_policy_decisions_by_flow: OrderedDict[
            str, list[dict[str, Any]]
        ] = OrderedDict()

    # ------------------------------------------------------------------ #
    # Write API — called by the orchestrator after each DAG run            #
    # ------------------------------------------------------------------ #

    def record_flow(self, chat_key: str, flow_id: str) -> None:
        """Record a new flow_id for a chat, maintaining LRU bounds."""
        if not chat_key or not flow_id:
            return
        # Update latest-flow-id-by-chat
        self._recent_flow_ids_by_chat.pop(chat_key, None)
        self._recent_flow_ids_by_chat[chat_key] = flow_id
        while len(self._recent_flow_ids_by_chat) > self._FLOW_CACHE_LIMIT:
            self._recent_flow_ids_by_chat.popitem(last=False)
        # Update per-chat flow history (most recent first, up to 5)
        history = [
            item
            for item in self._recent_flows_by_chat.get(chat_key, [])
            if item != flow_id
        ]
        history.insert(0, flow_id)
        self._recent_flows_by_chat.pop(chat_key, None)
        self._recent_flows_by_chat[chat_key] = history[:5]
        while len(self._recent_flows_by_chat) > self._FLOW_CACHE_LIMIT:
            self._recent_flows_by_chat.popitem(last=False)

    def record_policy_decisions(
        self, flow_id: str, events: list[dict[str, Any]]
    ) -> None:
        """Cache policy decision events for a flow."""
        if not flow_id:
            return
        decisions: list[dict[str, Any]] = []
        for event in events:
            if event.get("event") != "policy_decision":
                continue
            decisions.append(
                {
                    "decision_kind": str(event.get("decision_kind", "") or "").strip(),
                    "action_name": str(event.get("action_name", "") or "").strip(),
                    "state_bucket": str(event.get("state_bucket", "") or "").strip(),
                    "explain": str(event.get("explain", "") or "").strip(),
                }
            )
        self._recent_policy_decisions_by_flow.pop(flow_id, None)
        self._recent_policy_decisions_by_flow[flow_id] = decisions
        while len(self._recent_policy_decisions_by_flow) > self._FLOW_CACHE_LIMIT:
            self._recent_policy_decisions_by_flow.popitem(last=False)

    # ------------------------------------------------------------------ #
    # Read helpers                                                          #
    # ------------------------------------------------------------------ #

    def latest_flow_id(self, chat_key: str) -> str:
        return self._recent_flow_ids_by_chat.get(chat_key, "")

    def recent_flow_ids(self, chat_key: str) -> list[str]:
        return list(self._recent_flows_by_chat.get(chat_key, []))

    # ------------------------------------------------------------------ #
    # Public inspect methods                                                #
    # ------------------------------------------------------------------ #

    def inspect_runtime_flow(self, flow_id: str = "", chat_key: str = "") -> str:
        resolved_flow_id = flow_id.strip()
        resolved_chat_key = chat_key.strip()
        if not resolved_flow_id and resolved_chat_key:
            resolved_flow_id = self._recent_flow_ids_by_chat.get(resolved_chat_key, "")
        if not resolved_flow_id:
            return "暂无可观测的 flow。"
        snapshot = self._heartbeat_registry.snapshot(resolved_flow_id)
        events = self._child_task_bus.events_for(resolved_flow_id)
        policy_decisions = list(
            self._recent_policy_decisions_by_flow.get(resolved_flow_id, []) or []
        )
        parts = ["[Runtime Flow]", f"flow_id={resolved_flow_id}"]
        if resolved_chat_key:
            parts.append(f"chat_key={resolved_chat_key}")
        if policy_decisions:
            lines = []
            for item in policy_decisions[-8:]:
                kind = str(item.get("decision_kind", "") or "").strip()
                action = str(item.get("action_name", "") or "").strip()
                state_bucket = str(item.get("state_bucket", "") or "").strip()
                explain = str(item.get("explain", "") or "").strip()
                suffix = []
                if state_bucket:
                    suffix.append(f"bucket={state_bucket}")
                if explain:
                    suffix.append(explain)
                lines.append(
                    f"- decision_kind={kind or '-'} action={action or '-'}"
                    + (f" ({'; '.join(suffix)})" if suffix else "")
                )
            parts.append("[Policy Decisions]\n" + "\n".join(lines))
        if snapshot:
            lines = []
            for task_id, state in sorted(snapshot.items()):
                lines.append(
                    f"- task_id={task_id} status={state.get('status', '')} progress={state.get('progress', None)}"
                )
            parts.append("[Tasks]\n" + "\n".join(lines))
        if events:
            lines = []
            for event in events[-12:]:
                normalized_event = normalize_runtime_feedback_event(
                    {
                        "event": event.event,
                        "task_id": event.task_id,
                        "flow_id": event.flow_id,
                        "payload": dict(event.payload or {}),
                    }
                )
                status = str(normalized_event.state or "")
                progress = normalized_event.progress
                desc = runtime_event_primary_label(normalized_event)
                suffix = []
                if desc:
                    suffix.append(desc)
                if status:
                    suffix.append(f"status={status}")
                if progress is not None:
                    suffix.append(f"progress={progress}")
                lines.append(
                    f"- task_id={event.task_id} event={event.event}"
                    + (f" ({', '.join(suffix)})" if suffix else "")
                )
            parts.append("[Recent Events]\n" + "\n".join(lines))
        if len(parts) == 2:
            parts.append("暂无 task/event 快照。")
        return "\n".join(parts)

    def inspect_chat_context(self, chat_key: str, query: str = "") -> str:
        if not chat_key:
            return "缺少 chat_key。"
        view = build_context_view(
            memory_store=self._memory_store, chat_id=chat_key, query=query
        )
        records = self._memory_store.list_memories(chat_id=chat_key)
        parts = ["[Chat Context]", f"chat_key={chat_key}"]
        if query:
            parts.append(f"query={query}")
        if view.hot:
            parts.append("[Hot Context]\n- " + "\n- ".join(view.hot))
        if view.warm:
            parts.append("[Warm Context]\n- " + "\n- ".join(view.warm))
        if view.cold:
            parts.append("[Cold Context]\n- " + "\n- ".join(view.cold))
        if records:
            lines = [
                f"- memory_type={record.memory_type} key={record.key} tier={record.tier} status={record.status} confidence={record.confidence:.2f} summary={record.summary}"
                for record in records[:12]
            ]
            parts.append("[Memory Records]\n" + "\n".join(lines))
        tape = self._tape_store.get_or_create(chat_key)
        anchor = tape.last_anchor()
        if anchor is not None:
            summary = str((anchor.payload.get("state") or {}).get("summary", "") or "")
            if summary:
                parts.append(f"[Tape Summary]\n{summary}")
        return "\n".join(parts)

    def inspect_policy(self, chat_key: str = "", decision_kind: str = "") -> str:
        kinds = (
            [decision_kind.strip()]
            if decision_kind.strip()
            else ["decomposition", "scheduling", "worker"]
        )
        parts = ["[Policy]"]
        if chat_key:
            parts.append(f"chat_key={chat_key}")
        for kind in kinds:
            stats = self._policy_store.summarize_action_stats(decision_kind=kind)
            parts.append(f"decision_kind={kind}")
            if not stats:
                parts.append("- no_stats")
                continue
            ranked = sorted(
                stats.items(),
                key=lambda item: (
                    float(
                        item[1].get("effective_samples", item[1].get("samples", 0.0))
                        or 0.0
                    ),
                    float(item[1].get("mean_reward", 0.0) or 0.0),
                ),
                reverse=True,
            )
            for action_name, payload in ranked[:5]:
                parts.append(
                    "- "
                    + f"action={action_name} "
                    + f"samples={int(payload.get('samples', 0) or 0)} "
                    + f"effective_samples={float(payload.get('effective_samples', payload.get('samples', 0.0)) or 0.0):.2f} "
                    + f"mean_reward={float(payload.get('mean_reward', 0.0) or 0.0):.2f} "
                    + f"recent_mean_reward={float(payload.get('recent_mean_reward', 0.0) or 0.0):.2f} "
                    + f"drift_score={float(payload.get('drift_score', 0.0) or 0.0):.2f} "
                    + f"failure_rate={float(payload.get('failure_rate', 0.0) or 0.0):.2f} "
                    + f"avg_execution_elapsed_ms={float(payload.get('avg_execution_elapsed_ms', 0.0) or 0.0):.2f} "
                    + f"avg_tool_call_count={float(payload.get('avg_tool_call_count', 0.0) or 0.0):.2f} "
                    + f"tool_failure_rate={float(payload.get('tool_failure_rate', 0.0) or 0.0):.2f} "
                    + f"loop_guard_block_rate={float(payload.get('loop_guard_block_rate', 0.0) or 0.0):.2f} "
                    + f"max_step_exhausted_rate={float(payload.get('max_step_exhausted_rate', 0.0) or 0.0):.2f} "
                    + f"feedback_score={float(payload.get('feedback_score', 0.0) or 0.0):.2f}"
                )
        telemetry_summary_fn = getattr(
            self._policy_store, "summarize_runtime_telemetry", None
        )
        if callable(telemetry_summary_fn):
            try:
                telemetry = telemetry_summary_fn(chat_key=chat_key or None)
            except TypeError:
                telemetry = telemetry_summary_fn()

            def _format_skip_breakdown(payload: dict[str, Any]) -> str:
                breakdown = payload.get("skip_breakdown")
                if not isinstance(breakdown, dict) or not breakdown:
                    return ""
                items: list[str] = []
                for reason, count in sorted(
                    (
                        (str(reason).strip() or "unknown", int(value or 0))
                        for reason, value in breakdown.items()
                    ),
                    key=lambda item: (-item[1], item[0]),
                ):
                    items.append(f"{reason}:{count}")
                return ",".join(items)

            overall = telemetry.get("overall") if isinstance(telemetry, dict) else None
            by_route_mode = (
                telemetry.get("by_route_mode", {})
                if isinstance(telemetry, dict)
                else {}
            )

            _TELEMETRY_FLOAT_FIELDS = (
                "avg_router_latency_ms",
                "avg_execution_elapsed_ms",
                "avg_task_result_count",
                "avg_executor_step_count",
                "avg_tool_call_count",
                "fallback_rate",
                "skipped_rate",
                "model_route_rate",
                "rule_hit_rate",
                "reflection_route_rate",
                "reflection_match_rate",
                "reflection_override_rate",
                "tool_failure_rate",
                "loop_guard_block_rate",
                "max_step_exhausted_rate",
                "dead_letter_rate",
                "stalled_rate",
                "execution_style_reflection_rate",
                "parallelism_reflection_rate",
                "worker_reflection_rate",
                "execution_style_guardrail_reduce_rate",
                "parallelism_guardrail_soften_rate",
                "worker_guardrail_soften_rate",
                "mean_reward",
            )

            def _format_telemetry_row(
                payload: dict[str, Any],
                *,
                prefix: str = "",
            ) -> str:
                tokens: list[str] = []
                if prefix:
                    tokens.append(prefix)
                tokens.append(f"runs={int(payload.get('runs', 0) or 0)}")
                for field_name in _TELEMETRY_FLOAT_FIELDS:
                    tokens.append(
                        f"{field_name}={float(payload.get(field_name, 0.0) or 0.0):.2f}"
                    )
                skip_bd = _format_skip_breakdown(payload)
                if skip_bd:
                    tokens.append(f"skip_breakdown={skip_bd}")
                return "- " + " ".join(tokens)

            if isinstance(overall, dict) and int(overall.get("runs", 0) or 0) > 0:
                parts.append("[Routing Telemetry]")
                parts.append(_format_telemetry_row(overall))
                if isinstance(by_route_mode, dict):
                    for route_mode, payload in sorted(by_route_mode.items()):
                        if not isinstance(payload, dict):
                            continue
                        parts.append(
                            _format_telemetry_row(
                                payload, prefix=f"route_mode={route_mode}"
                            )
                        )
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Formatting helpers (used by orchestrator debug output)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_debug_policy_line(
        label: str,
        value: str,
        *,
        explain: str = "",
    ) -> str:
        normalized_label = str(label or "").strip() or "unknown"
        normalized_value = str(value or "").strip() or "-"
        normalized_explain = str(explain or "").strip()
        if not normalized_explain:
            return f"{normalized_label}={normalized_value}"
        return f"{normalized_label}={normalized_value}; explain={normalized_explain}"

    @classmethod
    def build_debug_policy_summary(
        cls,
        *,
        flow_id: str,
        decomposition_action: str,
        decomposition_hint: str,
        route_mode: str,
        router_skip_reason: str,
        routing_decision: RoutingDecision | None,
        scheduling_policy: dict[str, Any],
        worker_policy: dict[str, Any],
    ) -> str:
        routing_value = (
            "/".join(
                part
                for part in (
                    str(routing_decision.decision_source or "").strip(),
                    str(routing_decision.route_mode or "").strip(),
                    str(routing_decision.execution_style or "").strip(),
                )
                if part
            )
            if routing_decision is not None
            else f"fallback/{str(route_mode or '').strip() or 'tool_workflow'}"
        )
        routing_explain = (
            str(routing_decision.explain or "").strip()
            if routing_decision is not None
            else f"router_skip_reason={str(router_skip_reason or 'fallback').strip()}"
        )
        lines = [
            "调试：编排决策",
            f"flow_id={str(flow_id or '').strip() or '-'}",
            cls.format_debug_policy_line(
                "decomposition",
                str(decomposition_action or "").strip(),
                explain=decomposition_hint,
            ),
            cls.format_debug_policy_line(
                "routing",
                routing_value,
                explain=routing_explain,
            ),
            cls.format_debug_policy_line(
                "scheduling",
                str(scheduling_policy.get("action_name", "") or "").strip(),
                explain=str(scheduling_policy.get("explain", "") or ""),
            ),
            cls.format_debug_policy_line(
                "worker",
                str(worker_policy.get("action_name", "") or "").strip(),
                explain=str(worker_policy.get("explain", "") or ""),
            ),
        ]
        return "\n".join(line for line in lines if line.strip())
