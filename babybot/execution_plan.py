"""Execution planning layer for user-facing orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .task_contract import TaskContract


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    kind: str
    title: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    contract: TaskContract
    steps: tuple[PlanStep, ...]
    round_budget: int | None
    stopping_condition: str


def build_execution_plan(contract: TaskContract) -> ExecutionPlan:
    if contract.mode == "debate":
        participants = list(contract.allowed_agents or ())
        step = PlanStep(
            step_id="step_debate",
            kind="debate",
            title="Structured debate",
            payload={
                "participants": participants,
                "round_budget": contract.round_budget,
                "stopping_condition": contract.termination_rule,
            },
        )
        steps = (step,)
    else:
        step = PlanStep(
            step_id="step_answer",
            kind="direct_answer",
            title="Direct answer",
            payload={"deliverable": contract.deliverable},
        )
        steps = (step,)
    return ExecutionPlan(
        plan_id=f"plan:{uuid.uuid4().hex[:12]}",
        contract=contract,
        steps=steps,
        round_budget=contract.round_budget,
        stopping_condition=contract.termination_rule,
    )
