"""Orchestrator built on lightweight kernel and scheduler."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .config import Config
from .model_gateway import OpenAICompatibleGateway
from .resource import ResourceManager
from .scheduler import Scheduler, TaskSpec

if TYPE_CHECKING:
    from .heartbeat import Heartbeat

logger = logging.getLogger(__name__)


@dataclass
class TaskResponse:
    """Structured response from process_task with text and optional media."""

    text: str = ""
    media_paths: list[str] = field(default_factory=list)


class RouteDecision(BaseModel):
    route: Literal["simple", "complex"]
    reason: str = ""


class PlanTask(BaseModel):
    task_id: str
    description: str
    deps: list[str] = Field(default_factory=list)
    include_groups: list[str] | None = None
    include_tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    timeout: int | None = None
    retries: int | None = None


class ExecutionPlan(BaseModel):
    mode: Literal["serial", "parallel", "hybrid"] = "hybrid"
    tasks: list[PlanTask] = Field(default_factory=list)
    rationale: str = ""


class SynthesizedAnswer(BaseModel):
    conclusion: str
    evidence: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)


class OrchestratorAgent:
    """Orchestrator with model-driven planning and scheduler execution."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.model.validate()
        self.resource_manager = ResourceManager(self.config)
        self.gateway = OpenAICompatibleGateway(self.config)
        self._scheduler = Scheduler(max_parallel=self.config.system.max_parallel)
        self._plan_events: list[dict[str, Any]] = []
        self._initialized = False
        self._collected_media: list[str] = []

    def _get_router_prompt(self) -> str:
        return """将用户任务分为 simple 或 complex：
- simple: 单轮问答或轻量改写，不依赖多步过程
- complex: 需要工具、外部信息、多步骤或子任务编排
只输出 JSON：{"route":"simple|complex","reason":"..."}"""

    def _get_planner_prompt(self) -> str:
        return """你是任务规划器。将复杂请求拆分为可执行子任务列表并给出依赖。
要求：
- task_id 用简短英文标识，如 t1/t2
- description 明确可执行目标
- deps 仅引用已存在 task_id
- 能并行就减少 deps
- include_groups/include_tools/exclude_tools 仅在明确需要时提供
- mode 选择 serial/parallel/hybrid
只输出 ExecutionPlan 对应 JSON。"""

    def _get_direct_prompt(self) -> str:
        return """你是高效助手。对任务直接回答：
- 简洁准确
- 不虚构工具执行结果
- 如需外部信息必须明确指出并建议调用工具"""

    def _get_synth_prompt(self) -> str:
        return """你是结果整合器。根据子任务结果生成最终答复。
输出 JSON：
- conclusion: 面向用户的最终结论
- evidence: 支撑结论的关键证据列表
- failed_tasks: 失败任务与原因列表"""

    async def _route_task(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> RouteDecision:
        heuristic_complex = any(
            key in user_input.lower()
            for key in ["调研", "规划", "并行", "分步", "实现", "重构", "multi", "agent"]
        )
        logger.info("Routing task (calling LLM)...")
        structured = await self.gateway.complete_structured(
            self._get_router_prompt(),
            user_input,
            RouteDecision,
            heartbeat=heartbeat,
        )
        decision = structured or RouteDecision(
            route="complex" if heuristic_complex else "complex",
            reason="fallback",
        )
        logger.info("Route decision route=%s reason=%s", decision.route, decision.reason)
        return decision

    def _normalize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        used: set[str] = set()
        normalized: list[PlanTask] = []
        for idx, task in enumerate(plan.tasks, start=1):
            task_id = task.task_id.strip() or f"t{idx}"
            if task_id in used:
                task_id = f"{task_id}_{idx}"
            used.add(task_id)
            normalized.append(
                PlanTask(
                    task_id=task_id,
                    description=task.description.strip() or f"子任务 {idx}",
                    deps=[dep for dep in task.deps if dep in used],
                    include_groups=task.include_groups,
                    include_tools=task.include_tools,
                    exclude_tools=task.exclude_tools,
                    timeout=task.timeout,
                    retries=task.retries,
                )
            )
        return ExecutionPlan(mode=plan.mode, tasks=normalized, rationale=plan.rationale)

    async def _make_execution_plan(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> ExecutionPlan:
        logger.info("Making execution plan (calling LLM)...")
        plan = await self.gateway.complete_structured(
            self._get_planner_prompt(),
            user_input,
            ExecutionPlan,
            heartbeat=heartbeat,
        )
        if not plan or not plan.tasks:
            plan = ExecutionPlan(
                mode="hybrid",
                tasks=[PlanTask(task_id="t1", description=user_input)],
                rationale="fallback_single_task",
            )
        normalized = self._normalize_plan(plan)
        logger.info(
            "Execution plan mode=%s tasks=%d rationale=%s",
            normalized.mode,
            len(normalized.tasks),
            normalized.rationale,
        )
        for task in normalized.tasks:
            logger.info(
                "Plan task id=%s deps=%s timeout=%s retries=%s include_groups=%s include_tools=%s desc=%s",
                task.task_id,
                task.deps,
                task.timeout,
                task.retries,
                task.include_groups,
                task.include_tools,
                task.description[:160],
            )
        self._plan_events.append(
            {
                "event": "plan_created",
                "plan": normalized.model_dump(),
            }
        )
        self._plan_events = self._plan_events[-20:]
        return normalized

    async def _answer_direct(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> str:
        logger.info("_answer_direct calling run_subagent_task...")
        text, media = await self.resource_manager.run_subagent_task(
            task_description=user_input,
            lease={},
            agent_name="DirectAssistant",
            heartbeat=heartbeat,
        )
        logger.info("_answer_direct subagent done text_len=%d media=%d", len(text or ""), len(media or []))
        if media:
            self._collected_media.extend(media)
        if text.strip():
            return text
        return await self.gateway.complete(
            self._get_direct_prompt(), user_input, heartbeat=heartbeat
        )

    async def _execute_subtask(
        self, task: TaskSpec, heartbeat: Heartbeat | None = None
    ) -> str:
        logger.info(
            "Subtask start id=%s deps=%s timeout=%s retries=%s",
            task.task_id,
            task.deps,
            task.timeout,
            task.retries,
        )
        started = time.perf_counter()
        text, media = await self.resource_manager.run_subagent_task(
            task_description=task.description,
            lease=task.lease,
            agent_name=f"SubAgent-{task.task_id}",
            heartbeat=heartbeat,
        )
        logger.info(
            "Subtask done id=%s elapsed=%.2fs output_len=%d media=%d",
            task.task_id,
            time.perf_counter() - started,
            len(text or ""),
            len(media or []),
        )
        if media:
            self._collected_media.extend(media)
        return text

    async def _run_plan_with_scheduler(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> str | None:
        plan = await self._make_execution_plan(user_input, heartbeat)
        if heartbeat is not None:
            heartbeat.beat()
        logger.info("Plan ready, building task specs...")
        task_specs = [
            TaskSpec(
                task_id=task.task_id,
                description=task.description,
                deps=task.deps,
                lease={
                    "include_groups": task.include_groups,
                    "include_tools": task.include_tools,
                    "exclude_tools": task.exclude_tools,
                },
                timeout=task.timeout or self.config.system.subtask_timeout,
                retries=task.retries or 0,
            )
            for task in plan.tasks
        ]
        if not task_specs:
            return None

        async def _executor_with_heartbeat(task: TaskSpec) -> str:
            return await self._execute_subtask(task, heartbeat)

        results = await self._scheduler.run(
            tasks=task_specs,
            executor=_executor_with_heartbeat,
            mode=plan.mode,
            heartbeat=heartbeat,
        )
        status_summary: dict[str, int] = {}
        for res in results.values():
            status_summary[res.status] = status_summary.get(res.status, 0) + 1
        logger.info("Scheduler finished summary=%s", status_summary)
        for task_id, res in results.items():
            if res.status != "succeeded":
                logger.error(
                    "Task failed id=%s status=%s error=%s",
                    task_id,
                    res.status,
                    res.error,
                )
        payload = {
            "user_task": user_input,
            "plan": plan.model_dump(),
            "results": {
                task_id: {
                    "status": res.status,
                    "output": res.output,
                    "error": res.error,
                }
                for task_id, res in results.items()
            },
        }
        logger.info("Synthesizing results (calling LLM)...")
        structured = await self.gateway.complete_structured(
            self._get_synth_prompt(),
            json.dumps(payload, ensure_ascii=False, indent=2),
            SynthesizedAnswer,
            heartbeat=heartbeat,
        )
        if heartbeat is not None:
            heartbeat.beat()
        if structured:
            lines = [structured.conclusion.strip()]
            if structured.evidence:
                lines.append("\n依据：")
                lines.extend(f"- {e}" for e in structured.evidence if e)
            if structured.failed_tasks:
                lines.append("\n失败任务：")
                lines.extend(f"- {e}" for e in structured.failed_tasks if e)
            return "\n".join(line for line in lines if line).strip()
        return None

    async def process_task(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> TaskResponse:
        if not self._initialized:
            logger.info("Initializing resource manager...")
            await self.resource_manager.initialize_async()
            self._initialized = True
            logger.info("Resource manager initialized")

        self._collected_media.clear()
        task_started = time.perf_counter()
        logger.info("Process task input=%s", user_input[:200])

        # Beat immediately so the idle timer starts fresh.
        if heartbeat is not None:
            heartbeat.beat()

        try:
            decision = await self._route_task(user_input, heartbeat)
            if heartbeat is not None:
                heartbeat.beat()

            if decision.route == "simple":
                logger.info("Route=simple, calling _answer_direct...")
                text = await self._answer_direct(user_input, heartbeat)
                if heartbeat is not None:
                    heartbeat.beat()
                logger.info("Process task done route=simple elapsed=%.2fs", time.perf_counter() - task_started)
                return TaskResponse(text=text, media_paths=list(self._collected_media))

            logger.info("Route=complex, calling _run_plan_with_scheduler...")
            scheduled = await self._run_plan_with_scheduler(user_input, heartbeat)
            if heartbeat is not None:
                heartbeat.beat()
            if scheduled:
                logger.info("Process task done route=complex elapsed=%.2fs", time.perf_counter() - task_started)
                return TaskResponse(
                    text=scheduled,
                    media_paths=list(self._collected_media),
                )

            logger.info("Scheduler returned None, falling back to _answer_direct...")
            text = await self._answer_direct(user_input, heartbeat)
            if heartbeat is not None:
                heartbeat.beat()
            logger.info("Process task done route=fallback elapsed=%.2fs", time.perf_counter() - task_started)
            return TaskResponse(text=text, media_paths=list(self._collected_media))
        except Exception as exc:
            logger.exception("Error processing task")
            return TaskResponse(text=f"处理任务时出错：{exc}")

    def reset(self) -> None:
        self.resource_manager.reset()
        self._scheduler.reset()
        self._plan_events.clear()
        self._initialized = False

    def get_status(self) -> dict[str, Any]:
        return {
            "resource_manager": "initialized",
            "available_tools": len(self.resource_manager.get_available_tools()),
            "resources": self.resource_manager.search_resources(),
            "scheduler": self._scheduler.get_status(),
            "plan_events": list(self._plan_events),
        }
