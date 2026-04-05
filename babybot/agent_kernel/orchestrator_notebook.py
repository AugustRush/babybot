"""Notebook/runtime helper for DynamicOrchestrator."""

from __future__ import annotations

import json
import logging
from typing import Any

from .plan_notebook import PlanNotebook, create_root_notebook
from .runtime_state import RuntimeState
from .types import ExecutionContext, TaskResult

logger = logging.getLogger(__name__)

_EXECUTION_NODE_KINDS = {
    "tool_workflow",
    "team_debate",
    "team_cooperative",
    "plan_step",
}


class NotebookRuntimeHelper:
    """Encapsulates notebook lifecycle and convergence bookkeeping."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def ensure_plan_notebook(
        self,
        goal: str,
        context: ExecutionContext,
    ) -> PlanNotebook:
        state_view = RuntimeState(context)
        notebook = state_view.notebook_binding().notebook
        if notebook is not None:
            context.state.setdefault("plan_notebook_id", notebook.notebook_id)
            context.state.setdefault(
                "current_notebook_node_id",
                notebook.primary_frontier_node_id(),
            )
            context.state.setdefault("notebook_context_budget", 2400)
            return notebook
        notebook = create_root_notebook(
            goal=goal,
            flow_id=context.session_id,
            plan_id=str(
                getattr(context.state.get("execution_plan"), "plan_id", "") or ""
            ),
            metadata={
                "original_goal": str(context.state.get("original_goal", goal) or goal)
            },
        )
        context.state["plan_notebook"] = notebook
        context.state["plan_notebook_id"] = notebook.notebook_id
        context.state["current_notebook_node_id"] = notebook.primary_frontier_node_id()
        context.state.setdefault("notebook_context_budget", 2400)
        return notebook

    @staticmethod
    def notebook_task_map(context: ExecutionContext) -> dict[str, str]:
        mapping = RuntimeState(context).notebook_task_map()
        return mapping  # type: ignore[return-value]

    @staticmethod
    def task_title_from_description(description: str) -> str:
        for line in str(description or "").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
        return "Child task"

    @staticmethod
    def current_notebook_node(
        notebook: PlanNotebook,
        context: ExecutionContext,
    ) -> Any:
        binding = RuntimeState(context).notebook_binding()
        node_id = (
            binding.node_id
            if binding.notebook is notebook and binding.node_id in notebook.nodes
            else notebook.primary_frontier_node_id()
        )
        return notebook.get_node(node_id)

    @staticmethod
    def child_notebook_nodes(notebook: PlanNotebook, parent_id: str) -> list[Any]:
        return [
            node
            for node in notebook.nodes.values()
            if str(node.parent_id or "") == str(parent_id)
        ]

    def ensure_team_notebook_node(
        self,
        *,
        notebook: PlanNotebook,
        context: ExecutionContext,
        topic: str,
        mode: str,
        agents: list[dict[str, Any]],
        max_rounds: int,
    ) -> Any:
        current_node = self.current_notebook_node(notebook, context)
        node_kind = "team_cooperative" if mode == "cooperative" else "team_debate"
        node = current_node if current_node.kind == node_kind else None
        if node is None:
            node = notebook.add_child_node(
                parent_id=current_node.node_id,
                kind=node_kind,
                title=self.task_title_from_description(topic) or "Team execution",
                objective=str(topic or "").strip(),
                owner="team",
                metadata={
                    "team_mode": mode,
                    "agents": [
                        {
                            "id": str(agent.get("id", "") or "").strip(),
                            "role": str(agent.get("role", "") or "").strip(),
                        }
                        for agent in agents
                    ],
                    "max_rounds": max_rounds,
                },
            )
        if node.status == "pending":
            notebook.transition_node(
                node.node_id,
                "running",
                summary="已启动团队节点",
                detail=str(topic or ""),
                metadata={"team_mode": mode, "max_rounds": max_rounds},
            )
        return node

    def complete_converged_execution_nodes(self, notebook: PlanNotebook) -> None:
        for node in notebook.nodes.values():
            if node.kind not in _EXECUTION_NODE_KINDS:
                continue
            if node.status in {"completed", "failed", "cancelled"}:
                continue
            children = [
                child
                for child in self.child_notebook_nodes(notebook, node.node_id)
                if child.kind != "scheduled_task"
            ]
            if children and any(
                child.status not in {"completed", "failed", "cancelled"}
                for child in children
            ):
                continue
            summary = node.latest_summary or "执行节点完成"
            detail = node.result_text or node.objective
            notebook.transition_node(
                node.node_id,
                "completed",
                summary=summary,
                detail=detail,
            )

    @staticmethod
    def build_team_debate_payload(
        *,
        result: Any,
        agents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        last_arguments = [
            {
                "agent": entry["agent"],
                "role": entry["role"],
                "content": entry["content"][:500],
            }
            for entry in result.transcript[-len(agents) :]
        ]
        recommendation = str(result.summary or "").strip().splitlines()[0].strip()
        disagreements = [
            f"{entry['role']}: {entry['content'][:200]}" for entry in last_arguments
        ]
        return {
            "topic": result.topic,
            "rounds": result.rounds,
            "summary": result.summary,
            "completed": result.completed,
            "termination_reason": result.termination_reason,
            "transcript_length": len(result.transcript),
            "recommendation": recommendation,
            "disagreements": disagreements,
            "last_arguments": last_arguments,
        }

    def finalize_team_debate_node(
        self,
        *,
        notebook: PlanNotebook,
        node: Any,
        payload: dict[str, Any],
    ) -> None:
        recommendation = str(payload.get("recommendation", "") or "").strip()
        if recommendation:
            rationale = "\n".join(str(item) for item in payload.get("disagreements", []))
            notebook.add_decision(
                node_id=node.node_id,
                summary=recommendation,
                rationale=rationale,
                metadata={
                    "team_mode": "debate",
                    "rounds": payload.get("rounds"),
                    "termination_reason": payload.get("termination_reason"),
                    "progress": True,
                },
            )
        notebook.record_event(
            node_id=node.node_id,
            kind="summary",
            summary=str(payload.get("summary", "") or "团队讨论完成")[:200],
            detail=json.dumps(payload, ensure_ascii=False),
            metadata={"team_mode": "debate", "progress": True},
        )
        notebook.transition_node(
            node.node_id,
            "completed" if payload.get("completed") else "failed",
            summary=str(
                payload.get("recommendation")
                or payload.get("summary")
                or "团队讨论完成"
            )[:200],
            detail=str(payload.get("summary", "") or ""),
            metadata={
                "team_mode": "debate",
                "rounds": payload.get("rounds"),
                "termination_reason": payload.get("termination_reason"),
            },
        )
        if not payload.get("completed"):
            notebook.mark_needs_repair(
                node.node_id,
                message=str(
                    payload.get("termination_reason", "") or "team debate incomplete"
                ),
            )

    def initialize_team_cooperative_children(
        self,
        *,
        notebook: PlanNotebook,
        node: Any,
        tasks: list[dict[str, Any]],
    ) -> dict[str, str]:
        task_nodes: dict[str, str] = {}
        for task in tasks:
            task_id = str(task.get("task_id", "") or "").strip()
            if not task_id:
                continue
            dep_node_ids = tuple(
                task_nodes[dep_id]
                for dep_id in (task.get("deps") or [])
                if dep_id in task_nodes
            )
            task_node = notebook.add_child_node(
                parent_id=node.node_id,
                kind="team_task",
                title=task_id,
                objective=str(task.get("description", "") or task_id),
                owner="team",
                deps=dep_node_ids,
                metadata={"team_task_id": task_id},
            )
            task_nodes[task_id] = task_node.node_id
        return task_nodes

    def finalize_team_cooperative_node(
        self,
        *,
        notebook: PlanNotebook,
        node: Any,
        result: Any,
        task_nodes: dict[str, str],
    ) -> dict[str, Any]:
        for task_id, task_payload in dict(result.task_statuses or {}).items():
            node_id = task_nodes.get(task_id)
            if not node_id or node_id not in notebook.nodes:
                continue
            status = str(task_payload.get("status", "") or "pending")
            output = str(task_payload.get("output", "") or "")
            if status == "completed":
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary=f"团队子任务完成: {task_id}",
                    detail=output,
                )
            elif status == "failed":
                notebook.transition_node(
                    node_id,
                    "failed",
                    summary=f"团队子任务失败: {task_id}",
                    detail=output,
                )
                notebook.add_issue(
                    node_id=node_id,
                    title=f"团队子任务失败: {task_id}",
                    detail=output,
                    severity="high",
                )
            else:
                notebook.transition_node(
                    node_id,
                    "cancelled",
                    summary=f"团队子任务未完成: {task_id}",
                    detail=output or status,
                )
        payload = {
            "topic": result.topic,
            "tasks_completed": result.tasks_completed,
            "tasks_failed": result.tasks_failed,
            "tasks_total": result.tasks_total,
            "summary": result.summary,
            "completed": result.completed,
            "termination_reason": result.termination_reason,
            "task_statuses": result.task_statuses,
        }
        notebook.record_event(
            node_id=node.node_id,
            kind="summary",
            summary=str(result.summary or "团队协作完成")[:200],
            detail=json.dumps(payload, ensure_ascii=False),
            metadata={"team_mode": "cooperative", "progress": True},
        )
        notebook.transition_node(
            node.node_id,
            "completed" if result.completed else "failed",
            summary=str(result.summary or "团队协作完成")[:200],
            detail=str(result.summary or ""),
            metadata={
                "team_mode": "cooperative",
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
            },
        )
        if not result.completed:
            notebook.mark_needs_repair(
                node.node_id,
                message=str(
                    result.termination_reason or "team cooperative incomplete"
                ),
            )
        return payload

    def update_notebook_from_task_payloads(
        self,
        *,
        context: ExecutionContext,
        runtime: Any,
        task_ids: list[str],
    ) -> None:
        notebook = self.ensure_plan_notebook(
            str(context.state.get("original_goal", "") or ""),
            context,
        )
        task_map = self.notebook_task_map(context)
        candidate_task_ids = list(
            dict.fromkeys(list(task_ids) + list(runtime.results.keys()))
        )
        for task_id in candidate_task_ids:
            node_id = task_map.get(task_id)
            if not node_id or node_id not in notebook.nodes:
                continue
            result = runtime.results.get(task_id)
            if result is None:
                continue
            if result.status == "succeeded":
                auto_converged = result.metadata.get("auto_converged") is True
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary="子任务自动收敛" if auto_converged else "子任务完成",
                    detail=str(result.output or ""),
                    metadata=(
                        {
                            "auto_converged": True,
                            "completion_mode": result.metadata.get("completion_mode"),
                        }
                        if auto_converged
                        else None
                    ),
                )
                for artifact in getattr(result, "artifacts", ()) or ():
                    if str(artifact).strip():
                        notebook.add_artifact(
                            node_id=node_id,
                            path=str(artifact),
                            label="runtime artifact",
                        )
            elif result.status == "failed":
                notebook.transition_node(
                    node_id,
                    "failed",
                    summary="子任务失败",
                    detail=str(result.error or ""),
                )
                node = notebook.get_node(node_id)
                task_error = str(result.error or "")
                issue_exists = any(
                    issue.status == "open"
                    and issue.title == "子任务失败"
                    and issue.detail == task_error
                    for issue in node.issues
                )
                if not issue_exists:
                    notebook.add_issue(
                        node_id=node_id,
                        title="子任务失败",
                        detail=task_error,
                        severity="high",
                    )
                repair_message = task_error or "子任务失败，需要修复"
                repair_checkpoint_exists = any(
                    checkpoint.status == "open"
                    and checkpoint.kind == "needs_repair"
                    and checkpoint.message == repair_message
                    and str(checkpoint.metadata.get("task_id", "") or "") == task_id
                    for checkpoint in node.checkpoints
                )
                if not repair_checkpoint_exists:
                    notebook.mark_needs_repair(
                        node_id,
                        message=repair_message,
                        metadata={"task_id": task_id},
                    )

    @staticmethod
    def notebook_blocking_node_ids(
        notebook: PlanNotebook,
    ) -> list[str]:
        blocking: list[str] = []
        for node_id, node in notebook.nodes.items():
            if node.kind in {"root", "scheduled_task"}:
                continue
            if node.status in {"completed", "failed", "cancelled"}:
                continue
            blocking.append(node_id)
        return blocking

    @staticmethod
    def notebook_reply_blocking_checkpoints(
        notebook: PlanNotebook,
    ) -> list[str]:
        blocking_kinds = {"needs_human_input", "verification_failed"}
        return [
            f"{checkpoint.node_id}:{checkpoint.kind}"
            for checkpoint in notebook.open_checkpoints()
            if checkpoint.kind in blocking_kinds
        ]

    def force_converge_state(
        self,
        *,
        runtime: Any,
        context: ExecutionContext,
    ) -> tuple[str, dict[str, Any]] | None:
        threshold = max(
            1,
            int(getattr(self._config, "force_converge_dead_letter_threshold", 3) or 3),
        )
        notebook = self.ensure_plan_notebook(
            str(context.state.get("original_goal", "") or ""),
            context,
        )
        current_node = self.current_notebook_node(notebook, context)
        task_map = self.notebook_task_map(context)
        dead_ids: list[str] = []
        immediate_dead_ids: list[str] = []
        auto_converged_ids: list[str] = []
        success_ids: list[str] = []
        pending_ids: list[str] = []

        for task_id, node_id in task_map.items():
            normalized_node_id = str(node_id or "").strip()
            if not normalized_node_id or normalized_node_id not in notebook.nodes:
                continue
            node = notebook.get_node(normalized_node_id)
            if str(node.parent_id or "") != current_node.node_id:
                continue
            if node.kind == "scheduled_task":
                continue

            result = runtime.results.get(task_id)
            if result is not None:
                resource_id = str(result.metadata.get("resource_id", "") or "").strip()
                if resource_id == "group.scheduler":
                    continue
                if result.status == "succeeded":
                    if result.metadata.get("auto_converged") is True:
                        auto_converged_ids.append(task_id)
                    else:
                        success_ids.append(task_id)
                elif (
                    result.status == "failed"
                    and result.metadata.get("dead_lettered") is True
                ):
                    dead_ids.append(task_id)
                    no_progress_turns = int(
                        result.metadata.get("no_progress_turns", 0) or 0
                    )
                    error_text = str(result.error or "")
                    if no_progress_turns > 0 or "No progress after" in error_text:
                        immediate_dead_ids.append(task_id)
                continue

            if task_id in runtime.in_flight:
                state = runtime.task_state_snapshot(task_id)
                resource_id = str(state.get("resource_id", "") or "").strip()
                if resource_id != "group.scheduler":
                    pending_ids.append(task_id)

        if success_ids or pending_ids:
            return None

        if auto_converged_ids:
            stalled_preview = ", ".join(auto_converged_ids)
            reason = (
                "convergence required: a child task auto-converged after exhausting "
                f"its exploration budget ({stalled_preview}). "
                "Do not redispatch similar exploratory work. Inspect the existing "
                "task result and reply with a concise blocker summary, or switch "
                "to a materially different action path."
            )
        elif immediate_dead_ids:
            dead_preview = ", ".join(immediate_dead_ids)
            reason = (
                "convergence required: a child task already exhausted its exploration "
                f"budget without making progress ({dead_preview}). "
                "Do not redispatch similar exploratory work. Inspect the existing "
                "task result and reply with a concise blocker/failure summary, or "
                "switch to a materially different action path."
            )
        else:
            if len(dead_ids) < threshold:
                return None
            dead_preview = ", ".join(dead_ids[-threshold:])
            reason = (
                "convergence required: repeated child-task failures exhausted the retry "
                f"budget for the current notebook node ({dead_preview}). "
                "Do not dispatch more child tasks. Inspect existing task results and "
                "reply with a concise blocker/failure summary."
            )
        return reason, {
            "node_id": current_node.node_id,
            "dead_task_ids": dead_ids,
            "immediate_dead_task_ids": immediate_dead_ids,
            "auto_converged_task_ids": auto_converged_ids,
            "successful_task_ids": success_ids,
            "pending_task_ids": pending_ids,
        }

    def refresh_force_converge_state(
        self,
        *,
        runtime: Any,
        context: ExecutionContext,
    ) -> str | None:
        payload = self.force_converge_state(runtime=runtime, context=context)
        current_reason = str(
            context.state.get("orchestrator_force_converge", "") or ""
        ).strip()

        if payload is None:
            if current_reason:
                logger.info("DynamicOrchestrator: clearing force-converge mode")
            context.state.pop("orchestrator_force_converge", None)
            context.state.pop("orchestrator_force_converge_node_id", None)
            context.state.pop("orchestrator_force_converge_dead_task_ids", None)
            context.state.pop(
                "orchestrator_force_converge_immediate_dead_task_ids",
                None,
            )
            context.state.pop(
                "orchestrator_force_converge_auto_converged_task_ids",
                None,
            )
            return None

        reason, metadata = payload
        context.state["orchestrator_force_converge"] = reason
        context.state["orchestrator_force_converge_node_id"] = metadata["node_id"]
        context.state["orchestrator_force_converge_dead_task_ids"] = list(
            metadata["dead_task_ids"]
        )
        context.state["orchestrator_force_converge_immediate_dead_task_ids"] = list(
            metadata.get("immediate_dead_task_ids") or []
        )
        context.state["orchestrator_force_converge_auto_converged_task_ids"] = list(
            metadata.get("auto_converged_task_ids") or []
        )
        if current_reason == reason:
            return None

        logger.warning(
            "DynamicOrchestrator: force-converge activated node=%s dead_tasks=%s auto_converged=%s",
            metadata["node_id"],
            metadata["dead_task_ids"],
            metadata.get("auto_converged_task_ids") or [],
        )
        notebook = RuntimeState(context).notebook_binding().notebook
        if notebook is not None and metadata["node_id"] in notebook.nodes:
            notebook.record_event(
                node_id=metadata["node_id"],
                kind="decision",
                summary="进入强制收敛模式",
                detail=reason,
                metadata={
                    "force_converge": True,
                    "dead_task_ids": list(metadata["dead_task_ids"]),
                    "immediate_dead_task_ids": list(
                        metadata.get("immediate_dead_task_ids") or []
                    ),
                    "auto_converged_task_ids": list(
                        metadata.get("auto_converged_task_ids") or []
                    ),
                },
            )
        return reason

    def finalize_notebook_for_reply(
        self,
        *,
        context: ExecutionContext,
        reply_text: str,
    ) -> None:
        notebook = self.ensure_plan_notebook(
            str(context.state.get("original_goal", "") or reply_text),
            context,
        )
        self.complete_converged_execution_nodes(notebook)
        if not notebook.completion_summary:
            notebook.set_completion_summary(
                {
                    "final_summary": str(reply_text or "").strip(),
                    "decision_register": [
                        decision.summary
                        for node in notebook.nodes.values()
                        for decision in node.decisions[-3:]
                    ],
                    "artifact_manifest": [
                        artifact.path
                        for node in notebook.nodes.values()
                        for artifact in node.artifacts
                    ],
                }
            )
        for node_id, node in notebook.nodes.items():
            if node.status in {"completed", "failed", "cancelled"}:
                continue
            if node.kind == "root":
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary="编排收尾完成",
                    detail=str(reply_text or ""),
                )
