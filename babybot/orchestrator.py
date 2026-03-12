"""Orchestrator Agent with resource management."""

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

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

        # Create orchestrator agent with toolkit
        self._agent = ReActAgent(
            name="Orchestrator",
            sys_prompt=self._get_orchestrator_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
            toolkit=self.resource_manager.toolkit,
            enable_meta_tool=True,
        )
        self._agent.set_console_output_enabled(self.config.system.console_output)
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
- Worker 可以访问所有已激活的工具
- 如果有浏览器相关任务，可以使用 browser_* 工具"""

    async def process_task(self, user_input: str) -> str:
        """Process a task using orchestrator-workers pattern."""
        # Initialize MCP servers on first call
        if not self._initialized:
            await self.resource_manager.initialize_async()
            self._initialized = True

        msg = Msg(name="user", content=user_input, role="user")

        try:
            response = await self._agent(msg)

            # Extract text from response
            content = response.content if hasattr(response, "content") else []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        return text

            # If no text content, check for tool responses
            print(f"Debug: Response content: {content}")
            return "任务已处理，但没有生成文本回复。"

        except Exception as e:
            print(f"Error processing task: {e}")
            import traceback

            traceback.print_exc()
            return f"处理任务时出错：{e}"

    def reset(self) -> None:
        """Reset orchestrator and resource manager."""
        self.resource_manager.reset()
        if hasattr(self._agent, "memory"):
            self._agent.memory.clear()

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "resource_manager": "initialized",
            "available_tools": len(self.resource_manager.get_available_tools()),
        }
