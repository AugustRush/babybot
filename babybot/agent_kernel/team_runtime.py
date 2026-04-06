"""Structured team execution runtime used by the dynamic orchestrator."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from ..task_contract import assert_runtime_matches_contract
from .context import ContextManager
from .execution_constraints import (
    build_team_execution_policy,
    normalize_execution_constraints,
)
from .model import ModelMessage, ModelRequest
from .orchestrator_notebook import NotebookRuntimeHelper
from .team import TeamRunner
from .types import ExecutionContext, TaskContract

if TYPE_CHECKING:
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager
    from .orchestrator_config import OrchestratorConfig
    from .plan_notebook import PlanNotebook
    from .protocols import ExecutorPort

logger = logging.getLogger(__name__)


class TeamDispatchRuntime:
    """Runs structured team debate/cooperative execution for one orchestrator."""

    def __init__(
        self,
        *,
        resource_manager: "ResourceManager",
        gateway: "OpenAICompatibleGateway",
        executor: "ExecutorPort",
        config: "OrchestratorConfig",
        notebook_runtime: NotebookRuntimeHelper,
    ) -> None:
        self._rm = resource_manager
        self._gateway = gateway
        self._executor = executor
        self._config = config
        self._notebook_runtime = notebook_runtime

    async def run(self, args: dict[str, Any], context: ExecutionContext) -> str:
        topic = args.get("topic", "")
        agents = args.get("agents", [])
        mode = str(args.get("mode", "debate") or "debate").strip().lower()
        task_contract = context.state.get("task_contract")
        execution_constraints = normalize_execution_constraints(
            context.state.get("execution_constraints")
        )
        effective_args = dict(args)
        if getattr(task_contract, "round_budget", None) is not None:
            effective_args["max_rounds"] = int(task_contract.round_budget)
        team_policy = build_team_execution_policy(effective_args, execution_constraints)
        if task_contract is not None:
            assert_runtime_matches_contract(
                task_contract,
                max_rounds=team_policy.max_rounds,
            )
        max_rounds = team_policy.max_rounds

        if len(agents) < 2:
            return "error: dispatch_team requires at least 2 agents"

        if mode == "cooperative":
            tasks = args.get("tasks") or []
            if not tasks:
                return "error: cooperative mode requires a non-empty 'tasks' list"
        else:
            tasks = []

        notebook = self._notebook_runtime.ensure_plan_notebook(
            str(context.state.get("original_goal", "") or topic),
            context,
        )

        logger.info(
            "Team dispatch: topic=%r agents=%d max_rounds=%d mode=%s",
            topic[:80],
            len(agents),
            max_rounds,
            mode,
        )

        heartbeat = context.state.get("heartbeat")
        send_intermediate = context.state.get("send_intermediate_message")
        stream_callback = context.state.get("stream_callback")
        reset_stream = (
            getattr(stream_callback, "reset", None) if stream_callback else None
        )
        team_streaming = stream_callback is not None and reset_stream is not None
        runtime_event_callback = context.state.get("runtime_event_callback")
        stage_name = "cooperative" if mode == "cooperative" else "debate"
        plan_step_id = ""
        execution_plan = context.state.get("execution_plan")
        if getattr(execution_plan, "steps", None):
            plan_step_id = str(execution_plan.steps[0].step_id or "").strip()

        async def _emit_team_event(
            event_name: str,
            *,
            state: str,
            message: str,
            progress: float | None = None,
            error: str = "",
        ) -> None:
            text = str(message or "").strip()
            if (
                event_name == "progress"
                and not team_streaming
                and send_intermediate is not None
                and text
            ):
                await send_intermediate(text)
            if runtime_event_callback is None:
                return
            payload: dict[str, Any] = {
                "state": state,
                "stage": stage_name,
                "message": text,
            }
            if plan_step_id:
                payload["plan_step_id"] = plan_step_id
            if progress is not None:
                payload["progress"] = progress
            if error:
                payload["error"] = error
            maybe = runtime_event_callback(
                {
                    "event": event_name,
                    "flow_id": context.session_id,
                    "task_id": stage_name,
                    "payload": payload,
                }
            )
            if inspect.isawaitable(maybe):
                await maybe

        async def gateway_executor(
            agent_id: str, prompt: str, ctx: dict[str, Any]
        ) -> str:
            del agent_id
            sys_prompt = ctx.get(
                "system_prompt",
                self._config.team_default_agent_system_prompt,
            )
            messages = [
                ModelMessage(role="system", content=sys_prompt),
                ModelMessage(role="user", content=prompt),
            ]
            request = ModelRequest(messages=tuple(messages))
            gen_ctx = ExecutionContext(
                state={"stream_callback": stream_callback} if team_streaming else {},
            )
            if heartbeat is not None:
                async with heartbeat.keep_alive():
                    response = await self._gateway.generate(request, gen_ctx)
            else:
                response = await self._gateway.generate(request, gen_ctx)
            return response.text

        enriched_agents: list[dict[str, Any]] = []
        skills = getattr(self._rm, "skills", {})
        for agent in agents:
            agent_copy = dict(agent)
            skill_id = agent_copy.pop("skill_id", None) or agent_copy.pop(
                "profile_id", None
            )
            if skill_id:
                skill = skills.get(skill_id) or skills.get(skill_id.strip().lower())
                if skill is not None:
                    if not agent_copy.get("role") and skill.role:
                        agent_copy["role"] = skill.role
                    if not agent_copy.get("description"):
                        agent_copy["description"] = skill.description
                    if not agent_copy.get("system_prompt") and skill.prompt:
                        agent_copy["system_prompt"] = skill.prompt
            enriched_agents.append(agent_copy)

        if (
            team_policy.max_agents is not None
            and len(enriched_agents) > team_policy.max_agents
        ):
            enriched_agents = enriched_agents[: team_policy.max_agents]
        if len(enriched_agents) < 2:
            return "error: dispatch_team requires at least 2 agents after applying constraints"

        for ea in enriched_agents:
            logger.info(
                "Team agent resolved: id=%s role=%s skill=%s resource=%s",
                ea.get("id"),
                ea.get("role", ""),
                ea.get("skill_id", ea.get("profile_id", "")),
                ea.get("resource_id", ""),
            )

        team_node = self._notebook_runtime.ensure_team_notebook_node(
            notebook=notebook,
            context=context,
            topic=topic,
            mode=mode,
            agents=enriched_agents,
            max_rounds=max_rounds,
        )
        team_context = ContextManager(context).fork(
            session_id=f"{context.session_id}:{team_node.node_id}",
        )
        team_context.state["current_notebook_node_id"] = team_node.node_id

        async def resource_executor(
            resource_id: str, agent_id: str, prompt: str, ctx: dict[str, Any]
        ) -> str:
            del ctx
            task = TaskContract(
                task_id=f"team_{agent_id}",
                description=prompt,
                metadata={
                    "resource_id": resource_id,
                    "team_node_id": team_node.node_id,
                },
            )
            result = await self._executor.execute(task, team_context)
            if result.status != "succeeded":
                return f"[error: {result.error}]"
            return result.output

        for agent_copy in enriched_agents:
            rid = agent_copy.get("resource_id")
            if not rid:
                continue
            scope = self._rm.resolve_resource_scope(rid, require_tools=True)
            if scope is not None:
                agent_copy["executor"] = functools.partial(resource_executor, rid)

        if mode == "cooperative":
            return await self._run_cooperative(
                args=args,
                topic=topic,
                enriched_agents=enriched_agents,
                runner_executor=gateway_executor,
                team_policy=team_policy,
                max_rounds=max_rounds,
                heartbeat=heartbeat,
                team_streaming=team_streaming,
                reset_stream=reset_stream,
                emit_team_event=_emit_team_event,
                notebook=notebook,
                team_node=team_node,
            )

        async def on_turn(agent_id: str, role: str, round_num: int, text: str) -> None:
            if heartbeat is not None:
                heartbeat.beat()
            notebook.record_event(
                node_id=team_node.node_id,
                kind="observation",
                summary=f"Round {round_num}: {role}",
                detail=str(text or ""),
                metadata={
                    "agent_id": agent_id,
                    "role": role,
                    "round": round_num,
                },
            )
            if send_intermediate is not None and not team_streaming:
                header = f"**[{role} — Round {round_num}]**"
                await send_intermediate(header + "\n" + text)
            if team_streaming:
                await reset_stream()

        logger.info(
            "Team streaming=%s intermediate=%s heartbeat=%s",
            team_streaming,
            send_intermediate is not None,
            heartbeat is not None,
        )

        await _emit_team_event(
            "progress",
            state="planning",
            message=self._config.team_debate_started_message.format(
                n_agents=len(enriched_agents), max_rounds=max_rounds
            ),
            progress=0.0,
        )

        async def on_round_start(round_num: int, total_rounds: int) -> None:
            if heartbeat is not None:
                heartbeat.beat()
            await _emit_team_event(
                "progress",
                state="running",
                message=self._config.team_debate_round_message.format(
                    round_num=round_num, total_rounds=total_rounds
                ),
                progress=round_num / max(1, total_rounds + 1),
            )

        if team_streaming:
            await reset_stream()

        runner = TeamRunner(
            executor=gateway_executor,
            max_rounds=max_rounds,
            policy=team_policy,
        )
        result = await runner.run_debate(
            topic=topic,
            agents=enriched_agents,
            on_turn=on_turn,
            on_round_start=on_round_start,
        )

        logger.info(
            "Team debate finished: topic=%r rounds=%d transcript_len=%d",
            result.topic[:80],
            result.rounds,
            len(result.transcript),
        )

        if team_streaming:
            await reset_stream()

        await _emit_team_event(
            "completed" if result.completed else "failed",
            state="completed" if result.completed else "repairing",
            message=result.summary[:200] or self._config.team_debate_ended_message,
            progress=1.0 if result.completed else None,
        )
        payload = NotebookRuntimeHelper.build_team_debate_payload(
            result=result,
            agents=enriched_agents,
        )
        self._notebook_runtime.finalize_team_debate_node(
            notebook=notebook,
            node=team_node,
            payload=payload,
        )
        self._notebook_runtime.complete_converged_execution_nodes(notebook)
        return json.dumps(payload, ensure_ascii=False)

    async def _run_cooperative(
        self,
        *,
        args: dict[str, Any],
        topic: str,
        enriched_agents: list[dict[str, Any]],
        runner_executor: Any,
        team_policy: Any,
        max_rounds: int,
        heartbeat: Any,
        team_streaming: bool,
        reset_stream: Any,
        emit_team_event: Callable[..., Any],
        notebook: "PlanNotebook",
        team_node: Any,
    ) -> str:
        tasks = args.get("tasks") or []

        logger.info(
            "Team cooperative: topic=%r agents=%d tasks=%d",
            topic[:80],
            len(enriched_agents),
            len(tasks),
        )

        await emit_team_event(
            "progress",
            state="planning",
            message=self._config.team_coop_started_message.format(
                n_agents=len(enriched_agents), n_tasks=len(tasks)
            ),
            progress=0.0,
        )
        task_nodes = self._notebook_runtime.initialize_team_cooperative_children(
            notebook=notebook,
            node=team_node,
            tasks=tasks,
        )

        completed_count = {"n": 0}

        async def on_task_complete(agent_id: str, task_id: str, output: str) -> None:
            del agent_id
            completed_count["n"] += 1
            if heartbeat is not None:
                heartbeat.beat()
            node_id = task_nodes.get(task_id, "")
            if node_id and node_id in notebook.nodes:
                notebook.transition_node(
                    node_id,
                    "completed",
                    summary=f"团队子任务完成: {task_id}",
                    detail=str(output or ""),
                )
            progress = completed_count["n"] / max(1, len(tasks))
            await emit_team_event(
                "progress",
                state="running",
                message=self._config.team_coop_task_done_message.format(
                    task_id=task_id,
                    done=completed_count["n"],
                    total=len(tasks),
                ),
                progress=progress,
            )

        async def on_task_failed(agent_id: str, task_id: str, error: str) -> None:
            del agent_id
            node_id = task_nodes.get(task_id, "")
            if node_id and node_id in notebook.nodes:
                notebook.transition_node(
                    node_id,
                    "failed",
                    summary=f"团队子任务失败: {task_id}",
                    detail=str(error or ""),
                )
                notebook.add_issue(
                    node_id=node_id,
                    title=f"团队子任务失败: {task_id}",
                    detail=str(error or ""),
                    severity="high",
                )

        if team_streaming and reset_stream is not None:
            await reset_stream()

        runner = TeamRunner(
            executor=runner_executor,
            max_rounds=max_rounds,
            policy=team_policy,
        )
        result = await runner.run_cooperative(
            topic=topic,
            agents=enriched_agents,
            tasks=tasks,
            on_task_complete=on_task_complete,
            on_task_failed=on_task_failed,
        )

        logger.info(
            "Team cooperative finished: topic=%r completed=%d/%d failed=%d",
            result.topic[:80],
            result.tasks_completed,
            result.tasks_total,
            result.tasks_failed,
        )

        if team_streaming and reset_stream is not None:
            await reset_stream()

        await emit_team_event(
            "completed" if result.completed else "failed",
            state="completed" if result.completed else "repairing",
            message=result.summary[:200] or self._config.team_coop_ended_message,
            progress=1.0 if result.completed else None,
        )
        payload = self._notebook_runtime.finalize_team_cooperative_node(
            notebook=notebook,
            node=team_node,
            result=result,
            task_nodes=task_nodes,
        )
        payload["mode"] = "cooperative"
        payload["task_outputs"] = {
            tid: output[:500] for tid, output in result.task_outputs.items()
        }
        payload["mailbox_messages"] = len(result.mailbox_log)
        self._notebook_runtime.complete_converged_execution_nodes(notebook)
        return json.dumps(payload, ensure_ascii=False)


__all__ = ["TeamDispatchRuntime"]
