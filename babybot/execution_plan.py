"""Execution planning layer for user-facing orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .agent_kernel.plan_notebook import PlanNotebook, create_root_notebook
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
            kind="team_debate",
            title="Structured debate",
            payload={
                "participants": participants,
                "round_budget": contract.round_budget,
                "stopping_condition": contract.termination_rule,
                "allowed_tools": list(contract.allowed_tools or ()),
                "allowed_agents": list(contract.allowed_agents or ()),
            },
        )
        steps = (step,)
    else:
        step = PlanStep(
            step_id="step_tool_workflow",
            kind="tool_workflow",
            title="Tool-guided answer",
            payload={
                "deliverable": contract.deliverable,
                "allowed_tools": list(contract.allowed_tools or ()),
                "allowed_agents": list(contract.allowed_agents or ()),
                "allow_clarification": contract.allow_clarification,
            },
        )
        steps = (step,)
    return ExecutionPlan(
        plan_id=f"plan:{uuid.uuid4().hex[:12]}",
        contract=contract,
        steps=steps,
        round_budget=contract.round_budget,
        stopping_condition=contract.termination_rule,
    )


def compile_execution_plan_to_notebook(
    plan: ExecutionPlan,
    *,
    flow_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> PlanNotebook:
    notebook = create_root_notebook(
        goal=plan.contract.goal,
        flow_id=flow_id,
        plan_id=plan.plan_id,
        metadata={
            "deliverable": plan.contract.deliverable,
            "termination_rule": plan.stopping_condition,
            **dict(metadata or {}),
        },
    )
    for step in plan.steps:
        notebook.add_child_node(
            parent_id=notebook.root_node_id,
            kind=step.kind,
            title=step.title,
            objective=step.title,
            owner="planner",
            metadata={
                "step_id": step.step_id,
                "step_kind": step.kind,
                "execution_mode": step.kind,
                "payload": dict(step.payload),
            },
        )
    return notebook
