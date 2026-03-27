from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyDecisionRecord:
    flow_id: str
    chat_key: str
    decision_kind: str
    action_name: str
    state_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyOutcomeRecord:
    flow_id: str
    chat_key: str
    final_status: str
    reward: float
    outcome: dict[str, Any] = field(default_factory=dict)
