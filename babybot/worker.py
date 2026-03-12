"""Worker Agent factory for Handoffs workflow."""

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from .config import Config


def create_worker_agent(
    config: Config,
    toolkit: Toolkit,
    name: str = "Worker",
) -> ReActAgent:
    """Create a temporary worker agent with shared toolkit.

    Args:
        config: Configuration object.
        toolkit: Shared toolkit from orchestrator.

    Returns:
        A new ReActAgent instance for the worker.
    """
    model_kwargs = {
        "model_name": config.model.model_name,
        "api_key": config.model.api_key,
        "stream": False,
    }
    if config.model.api_base:
        model_kwargs["client_kwargs"] = {"base_url": config.model.api_base}
    model_kwargs["generate_kwargs"] = {
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
    }

    model_name = (config.model.model_name or "").lower()
    formatter = DeepSeekChatFormatter() if "deepseek" in model_name else OpenAIChatFormatter()
    available_tools = ", ".join(sorted(toolkit.tools.keys())) or "无"
    return ReActAgent(
        name=name,
        sys_prompt=f"""你是一个专业的助手。你的任务是完成分配给你的具体工作。

可用工具（仅以下这些）：{available_tools}

请：
1. 仔细分析任务需求
2. 必须优先使用工具完成需要外部信息/浏览器/文件操作的步骤
3. 如果任务完成，输出结构化终态：status=done, answer=最终答案, evidence=关键信息列表, errors=[]
4. 如果任务失败，输出结构化终态：status=failed, answer="", evidence=[], errors=[失败原因]
5. 不要只停留在 tool call 或 thinking；必须给出终态

如果有不清楚的地方，请说明需要更多信息。""",
        model=OpenAIChatModel(**model_kwargs),
        memory=InMemoryMemory(),
        formatter=formatter,
        toolkit=toolkit,
        enable_meta_tool=False,
        max_iters=14,
    )
