"""Orchestrator Agent with resource management."""

import json

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.plan import PlanNotebook

from .config import Config
from .resource import ResourceManager


class OrchestratorAgent:
    """Orchestrator agent with centralized resource management."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.model.validate()

        # Initialize resource manager (singleton)
        self.resource_manager = ResourceManager(self.config)

        # Setup model
        model_kwargs = {
            "model_name": self.config.model.model_name,
            "api_key": self.config.model.api_key,
            "stream": False,
        }
        if self.config.model.api_base:
            model_kwargs["client_kwargs"] = {"base_url": self.config.model.api_base}
        generate_kwargs = {
            "temperature": self.config.model.temperature,
            "max_tokens": self.config.model.max_tokens,
        }
        model_kwargs["generate_kwargs"] = generate_kwargs

        self._plan_notebook = PlanNotebook(max_subtasks=12)

        # Create orchestrator agent with toolkit
        self._agent = ReActAgent(
            name="Orchestrator",
            sys_prompt=self._get_orchestrator_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
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
            formatter=OpenAIChatFormatter(),
            toolkit=None,
            enable_meta_tool=False,
            max_iters=1,
        )
        self._direct_agent = ReActAgent(
            name="DirectAssistant",
            sys_prompt=self._get_direct_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
            toolkit=self.resource_manager.toolkit,
            enable_meta_tool=False,
            max_iters=3,
        )
        self._agent.set_console_output_enabled(self.config.system.console_output)
        self._router_agent.set_console_output_enabled(self.config.system.console_output)
        self._direct_agent.set_console_output_enabled(self.config.system.console_output)
        self._initialized = False

    def _get_orchestrator_prompt(self) -> str:
        """Get system prompt for orchestrator."""
        # Get available tools info
        available_tools = self.resource_manager.get_available_tools()
        tool_names = [
            t["function"]["name"]
            for t in available_tools
            if "function" in t and "name" in t["function"]
        ]

        tools_info = (
            f"当前可用的工具：{', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}"
            if tool_names
            else "当前没有激活的工具组"
        )

        return f"""你是一个任务协调器 (Orchestrator)。你的职责是：

1. 分析复杂任务并分解为子任务
2. 使用 create_worker 工具创建专门的 Worker 来完成每个子任务
3. 使用可用的工具来完成任务
4. 汇总所有 Worker 的结果给出最终答案

{tools_info}

重要提示：
- 使用 create_worker 工具来创建专门的 Worker 执行具体任务
- 当任务可以拆分时，优先使用 dispatch_workers 一次并发处理多个子任务
- Worker 可以访问所有已激活的工具
- 如果有浏览器相关任务，可以使用 browser_* 工具"""

    def _get_router_prompt(self) -> str:
        """Get routing prompt for quick simple/complex classification."""
        return """你是任务路由器。只输出一个 JSON 对象，不要输出额外文本：
{"route":"simple|complex","reason":"..."}

规则：
- simple: 单步问答、解释、润色、简短事实查询，不需要工具和多阶段推理
- complex: 需要多步骤、调用工具、代码执行、网页操作、调研对比、拆分子任务
- 不确定时选 complex"""

    def _get_direct_prompt(self) -> str:
        """Get prompt for direct-response assistant."""
        return """你是高效助手。对简单任务直接回答，要求：
- 简洁准确
- 不虚构工具执行结果
- 遇到需要实时信息、网页访问、搜索、浏览器操作时，必须调用可用工具获取结果
- 如果信息不足，明确说明缺失信息"""

    def _extract_text(self, response: Msg) -> str:
        """Extract text content from agent response."""
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

    async def _route_task(self, user_input: str) -> str:
        """Route task to simple direct response or complex orchestrator."""
        router_msg = Msg(name="user", content=user_input, role="user")
        try:
            response = await self._router_agent(router_msg)
            text = self._extract_text(response).strip()
            if not text:
                return "complex"
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end >= start:
                text = text[start : end + 1]
            parsed = json.loads(text)
            route = str(parsed.get("route", "complex")).strip().lower()
            return "simple" if route == "simple" else "complex"
        except Exception:
            return "complex"

    async def process_task(self, user_input: str) -> str:
        """Process a task using orchestrator-workers pattern."""
        # Initialize MCP servers on first call
        if not self._initialized:
            await self.resource_manager.initialize_async()
            self._initialized = True

        msg = Msg(name="user", content=user_input, role="user")

        try:
            route = await self._route_task(user_input)
            if route == "simple":
                response = await self._direct_agent(msg)
            else:
                response = await self._agent(msg)

            text = self._extract_text(response)
            if text:
                return text

            print(f"Debug: Response content: {response.content}")
            return "任务已处理，但没有生成文本回复。"

        except Exception as e:
            print(f"Error processing task: {e}")
            import traceback

            traceback.print_exc()
            return f"处理任务时出错：{e}"

    def reset(self) -> None:
        """Reset orchestrator and resource manager."""
        self.resource_manager.reset()
        self._initialized = False
        if hasattr(self._agent, "memory"):
            self._agent.memory.clear()
        if hasattr(self._router_agent, "memory"):
            self._router_agent.memory.clear()
        if hasattr(self._direct_agent, "memory"):
            self._direct_agent.memory.clear()
        if hasattr(self._plan_notebook, "reset"):
            self._plan_notebook.reset()

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "resource_manager": "initialized",
            "available_tools": len(self.resource_manager.get_available_tools()),
        }
