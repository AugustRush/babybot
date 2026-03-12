"""Orchestrator Agent with planning, scheduling and resource leasing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.plan import PlanNotebook

from .config import Config
from .resource import ResourceManager
from .scheduler import Scheduler, TaskSpec


@dataclass
class TaskResponse:
    """Structured response from process_task with text and optional media."""

    text: str = ""
    media_paths: list[str] = field(default_factory=list)


class RouteDecision(BaseModel):
    """Router decision for choosing simple vs complex execution."""

    route: Literal["simple", "complex"]
    reason: str = ""


class PlanTask(BaseModel):
    """A single planned subtask."""

    task_id: str
    description: str
    deps: list[str] = Field(default_factory=list)
    include_groups: list[str] | None = None
    include_tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    timeout: int | None = None
    retries: int = 0


class ExecutionPlan(BaseModel):
    """Structured execution plan with scheduling mode and tasks."""

    mode: Literal["serial", "parallel", "hybrid"] = "hybrid"
    tasks: list[PlanTask] = Field(default_factory=list)
    rationale: str = ""


class SynthesizedAnswer(BaseModel):
    """User-facing final answer contract."""

    conclusion: str
    evidence: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)


class OrchestratorAgent:
    """Orchestrator agent with centralized resource management."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.model.validate()

        self.resource_manager = ResourceManager(self.config)
        self._scheduler = Scheduler(max_parallel=4)

        model_kwargs = {
            "model_name": self.config.model.model_name,
            "api_key": self.config.model.api_key,
            "stream": False,
            "generate_kwargs": {
                "temperature": self.config.model.temperature,
                "max_tokens": self.config.model.max_tokens,
            },
        }
        if self.config.model.api_base:
            model_kwargs["client_kwargs"] = {"base_url": self.config.model.api_base}

        self._plan_notebook = PlanNotebook(max_subtasks=12)
        self._plan_events: list[dict[str, Any]] = []
        if hasattr(self._plan_notebook, "register_plan_change_hook"):
            self._plan_notebook.register_plan_change_hook(
                "orchestrator_progress",
                self._on_plan_change,
            )

        self._agent = ReActAgent(
            name="Orchestrator",
            sys_prompt=self._get_orchestrator_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=self._create_formatter(),
            toolkit=self.resource_manager.toolkit,
            enable_meta_tool=self.config.system.enable_meta_tool,
            parallel_tool_calls=True,
            plan_notebook=self._plan_notebook,
            max_iters=12,
        )
        self._router_agent = ReActAgent(
            name="Router",
            sys_prompt=self._get_router_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=self._create_formatter(),
            toolkit=None,
            enable_meta_tool=False,
            max_iters=1,
        )
        self._planner_agent = ReActAgent(
            name="Planner",
            sys_prompt=self._get_planner_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=self._create_formatter(),
            toolkit=None,
            enable_meta_tool=False,
            max_iters=1,
        )
        self._direct_agent = ReActAgent(
            name="DirectAssistant",
            sys_prompt=self._get_direct_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=self._create_formatter(),
            toolkit=self.resource_manager.toolkit,
            enable_meta_tool=False,
            max_iters=4,
        )
        self._synth_agent = ReActAgent(
            name="Synthesizer",
            sys_prompt=self._get_synth_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=self._create_formatter(),
            toolkit=None,
            enable_meta_tool=False,
            max_iters=2,
        )

        for agent in (
            self._agent,
            self._router_agent,
            self._planner_agent,
            self._direct_agent,
            self._synth_agent,
        ):
            agent.set_console_output_enabled(self.config.system.console_output)

        self._initialized = False
        self._collected_media: list[str] = []

    def _create_formatter(self) -> OpenAIChatFormatter | DeepSeekChatFormatter:
        model_name = (self.config.model.model_name or "").lower()
        if "deepseek" in model_name:
            return DeepSeekChatFormatter()
        return OpenAIChatFormatter()

    def _get_orchestrator_prompt(self) -> str:
        available_tools = self.resource_manager.get_available_tools()
        tool_names = [
            t["function"]["name"]
            for t in available_tools
            if "function" in t and "name" in t["function"]
        ]
        tools_info = (
            f"当前可用工具：{', '.join(tool_names[:12])}{'...' if len(tool_names) > 12 else ''}"
            if tool_names
            else "当前没有激活的工具组"
        )
        return f"""你是主控调度 Agent，负责：
1. 为复杂任务制定多步骤策略
2. 使用可用工具推进关键步骤
3. 在必要时将任务分配给子 Agent 并整合结果

{tools_info}

原则：
- 能并行的子任务尽量并行
- 必须引用工具真实执行结果，不可臆造
- 输出最终结论时说明关键依据"""

    def _get_router_prompt(self) -> str:
        return """将用户任务分为 simple 或 complex：
- simple: 单轮问答或轻量改写，不依赖多步过程
- complex: 需要工具、外部信息、多步骤或子任务编排
仅输出结构化结果。"""

    def _get_planner_prompt(self) -> str:
        return """你是任务规划器。将复杂请求拆分为可执行子任务列表并给出依赖。
要求：
- task_id 用简短英文标识，如 t1/t2
- description 明确可执行目标
- deps 仅引用已存在 task_id
- 能并行就减少 deps
- include_groups/include_tools/exclude_tools 仅在明确需要时提供
- mode 选择 serial/parallel/hybrid"""

    def _get_direct_prompt(self) -> str:
        return """你是高效助手。对简单任务直接回答：
- 简洁准确
- 不虚构工具执行结果
- 如需实时/网页信息，必须使用工具获取后再回答"""

    def _get_synth_prompt(self) -> str:
        return """你是结果整合器。根据子任务结果生成最终答复。
输出必须符合结构化字段：
- conclusion: 面向用户的最终结论
- evidence: 支撑结论的关键证据列表
- failed_tasks: 失败任务与原因列表
要求：
- 不要编造不存在的子任务输出
- 有失败任务时也要给出当前可回答的部分"""

    def _extract_text(self, response: Msg) -> str:
        text = response.get_text_content()
        if text:
            return text
        content = response.content if hasattr(response, "content") else []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text", "")
                if value:
                    return value
        return ""

    def _on_plan_change(self, _notebook: Any, plan: Any) -> None:
        """Hook from PlanNotebook for lightweight progress tracing."""
        try:
            if plan is None:
                payload: dict[str, Any] = {"event": "plan_cleared"}
            else:
                dump_fn = getattr(plan, "model_dump", None) or getattr(plan, "dict", None)
                payload = (
                    dump_fn()
                    if callable(dump_fn)
                    else {"event": "plan_changed", "raw": str(plan)}
                )
            self._plan_events.append(payload)
            self._plan_events = self._plan_events[-20:]
        except Exception:
            pass

    def _extract_structured(
        self,
        response: Msg,
        model_cls: type[BaseModel],
    ) -> BaseModel | None:
        metadata = getattr(response, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        structured = metadata.get("structured_output")
        if not isinstance(structured, dict):
            return None
        try:
            validate_fn = getattr(model_cls, "model_validate", None)
            if callable(validate_fn):
                return validate_fn(structured)
            parse_fn = getattr(model_cls, "parse_obj", None)
            if callable(parse_fn):
                return parse_fn(structured)
            return None
        except Exception:
            return None

    def _dump_model(self, model: BaseModel) -> dict[str, Any]:
        dump_fn = getattr(model, "model_dump", None)
        if callable(dump_fn):
            return dump_fn()
        dict_fn = getattr(model, "dict", None)
        if callable(dict_fn):
            return dict_fn()
        return {}

    async def _route_task(self, user_input: str) -> RouteDecision:
        router_msg = Msg(name="user", content=user_input, role="user")
        try:
            response = await self._router_agent(
                router_msg,
                structured_model=RouteDecision,
            )
            decision = self._extract_structured(response, RouteDecision)
            if isinstance(decision, RouteDecision):
                return decision
        except Exception:
            pass
        return RouteDecision(route="complex", reason="fallback")

    async def _make_execution_plan(self, user_input: str) -> ExecutionPlan:
        planner_msg = Msg(name="user", content=user_input, role="user")
        try:
            response = await self._planner_agent(
                planner_msg,
                structured_model=ExecutionPlan,
            )
            plan = self._extract_structured(response, ExecutionPlan)
            if isinstance(plan, ExecutionPlan) and plan.tasks:
                return self._normalize_plan(plan)
        except Exception:
            pass
        # Fallback: single task hybrid plan.
        return ExecutionPlan(
            mode="hybrid",
            tasks=[PlanTask(task_id="t1", description=user_input)],
            rationale="fallback_single_task",
        )

    def _normalize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        used: set[str] = set()
        normalized: list[PlanTask] = []
        for idx, task in enumerate(plan.tasks, start=1):
            task_id = task.task_id.strip() or f"t{idx}"
            if task_id in used:
                task_id = f"{task_id}_{idx}"
            used.add(task_id)

            deps = [dep for dep in task.deps if dep in used]
            normalized.append(
                PlanTask(
                    task_id=task_id,
                    description=task.description.strip() or f"子任务 {idx}",
                    deps=deps,
                    include_groups=task.include_groups,
                    include_tools=task.include_tools,
                    exclude_tools=task.exclude_tools,
                    timeout=task.timeout,
                    retries=task.retries,
                )
            )

        return ExecutionPlan(
            mode=plan.mode,
            tasks=normalized,
            rationale=plan.rationale,
        )

    async def _execute_subtask(self, task: TaskSpec) -> str:
        text, media = await self.resource_manager.run_subagent_task(
            task_description=task.description,
            lease=task.lease,
            agent_name=f"SubAgent-{task.task_id}",
        )
        if media:
            self._collected_media.extend(media)
        return text

    async def _run_plan_with_scheduler(self, user_input: str) -> str | None:
        plan = await self._make_execution_plan(user_input)
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
                timeout=task.timeout or self.config.system.timeout,
                retries=max(0, int(task.retries)),
            )
            for task in plan.tasks
        ]
        if not task_specs:
            return None

        results = await self._scheduler.run(
            tasks=task_specs,
            executor=self._execute_subtask,
            mode=plan.mode,
        )
        payload = {
            "user_task": user_input,
            "plan": self._dump_model(plan),
            "results": {
                task_id: {
                    "status": result.status,
                    "output": result.output,
                    "error": result.error,
                }
                for task_id, result in results.items()
            },
        }
        synth_msg = Msg(
            name="user",
            role="user",
            content=f"请整合以下执行结果并回答用户原问题：\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        )
        try:
            response = await self._synth_agent(
                synth_msg,
                structured_model=SynthesizedAnswer,
            )
            structured = self._extract_structured(response, SynthesizedAnswer)
            if isinstance(structured, SynthesizedAnswer):
                lines = [structured.conclusion.strip()]
                if structured.evidence:
                    lines.append("\n依据：")
                    lines.extend(f"- {item}" for item in structured.evidence if item)
                if structured.failed_tasks:
                    lines.append("\n失败任务：")
                    lines.extend(f"- {item}" for item in structured.failed_tasks if item)
                answer = "\n".join(line for line in lines if line).strip()
                if answer:
                    return answer
            text = self._extract_text(response)
            if text:
                return text
        except Exception:
            pass
        return None

    async def process_task(self, user_input: str) -> TaskResponse:
        if not self._initialized:
            await self.resource_manager.initialize_async()
            self._initialized = True

        self._collected_media.clear()
        msg = Msg(name="user", content=user_input, role="user")
        try:
            decision = await self._route_task(user_input)
            if decision.route == "simple":
                response = await self._direct_agent(msg)
                text = self._extract_text(response)
                if text:
                    return TaskResponse(text=text, media_paths=list(self._collected_media))

            scheduled_output = await self._run_plan_with_scheduler(user_input)
            if scheduled_output:
                return TaskResponse(text=scheduled_output, media_paths=list(self._collected_media))

            response = await self._agent(msg)
            text = self._extract_text(response)
            if text:
                return TaskResponse(text=text, media_paths=list(self._collected_media))

            print(f"Debug: Response content: {response.content}")
            return TaskResponse(
                text="任务已处理，但没有生成文本回复。",
                media_paths=list(self._collected_media),
            )
        except Exception as e:
            print(f"Error processing task: {e}")
            import traceback

            traceback.print_exc()
            return TaskResponse(text=f"处理任务时出错：{e}")

    def reset(self) -> None:
        self.resource_manager.reset()
        self._scheduler.reset()
        self._plan_events.clear()
        self._initialized = False
        for agent in (
            self._agent,
            self._router_agent,
            self._planner_agent,
            self._direct_agent,
            self._synth_agent,
        ):
            if hasattr(agent, "memory"):
                agent.memory.clear()
        if hasattr(self._plan_notebook, "reset"):
            self._plan_notebook.reset()

    def get_status(self) -> dict[str, Any]:
        return {
            "resource_manager": "initialized",
            "available_tools": len(self.resource_manager.get_available_tools()),
            "resources": self.resource_manager.search_resources(),
            "scheduler": self._scheduler.get_status(),
            "plan_events": list(self._plan_events),
        }
