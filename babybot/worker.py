"""Worker Agent that inherits resources from parent."""

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit

from .config import Config


class WorkerAgent:
    """Worker agent that inherits toolkit from orchestrator."""

    def __init__(
        self,
        config: Config,
        toolkit: Toolkit,
        sys_prompt: str | None = None,
    ):
        """Initialize worker agent.

        Args:
            config: Configuration object.
            toolkit: Inherited toolkit from orchestrator.
            sys_prompt: Optional custom system prompt.
        """
        self.config = config
        self.toolkit = toolkit

        model_kwargs = {
            "model_name": self.config.model.model_name,
            "api_key": self.config.model.api_key,
            "stream": False,
        }
        if self.config.model.api_base:
            model_kwargs["client_kwargs"] = {"base_url": self.config.model.api_base}
        if self.config.model.temperature:
            model_kwargs["generate_kwargs"] = {
                "temperature": self.config.model.temperature
            }

        self._agent = ReActAgent(
            name="Worker",
            sys_prompt=sys_prompt or self._get_default_sys_prompt(),
            model=OpenAIChatModel(**model_kwargs),
            memory=InMemoryMemory(),
            formatter=OpenAIChatFormatter(),
            toolkit=toolkit,  # Inherit toolkit
            enable_meta_tool=True,  # Enable tool management
        )
        self._agent.set_console_output_enabled(False)

    def _get_default_sys_prompt(self) -> str:
        """Get default system prompt for worker."""
        return """你是一个专业的助手。你的任务是完成分配给你的具体工作。
        
你已配备必要的工具来完成任务。请：
1. 仔细分析任务需求
2. 使用合适的工具执行任务
3. 提供详细、准确的回答

如果有不清楚的地方，请说明需要更多信息。"""

    async def execute(self, task_description: str) -> str:
        """Execute a task.

        Args:
            task_description: Description of the task to execute.

        Returns:
            Result of task execution.
        """
        msg = Msg(name="user", content=task_description, role="user")
        response = await self._agent(msg)

        # Extract text from response
        content = response.content if hasattr(response, "content") else []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return ""
