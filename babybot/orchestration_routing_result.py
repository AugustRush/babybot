"""RoutingResult — value object carrying all routing decisions.

Produced by _resolve_routing() in orchestrator.py and consumed by the
DAG execution phase. Immutable after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestration_router import RoutingDecision


@dataclass(frozen=True)
class RoutingResult:
    """All outputs of the routing pipeline for a single task."""

    # Core routing decision (None = fallback to tool_workflow)
    routing_decision: RoutingDecision | None
    route_mode: str  # "tool_workflow" | "answer" | "debate"

    # Policy decisions
    scheduling_policy: dict[str, Any]
    worker_policy: dict[str, Any]
    reflection_hints_payload: list[dict[str, Any]]

    # Assembled hints for the executor
    policy_hints: list[str]

    # Telemetry metadata (used for recording, not execution)
    resolved_router_model: str
    routing_latency_ms: float
    router_skip_reason: str
    should_use_model_router: bool
    intent_bucket: str
    scheduling_overridden: bool
    worker_overridden: bool
    execution_style_guardrail_reduced: bool
    relaxed_reflection_route_payload: Any
    guardrail_softened_scheduling: bool
    guardrail_softened_worker: bool

    # Scheduling features (needed for policy recording later)
    scheduling_features: dict[str, Any] = field(default_factory=dict)
