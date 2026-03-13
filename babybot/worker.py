"""Worker Agent factory for Handoffs workflow."""

from typing import Any

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from .config import Config


def _create_model_kwargs(config: Config) -> dict[str, Any]:
    """Create model kwargs dict from config."""
    model_kwargs: dict[str, Any] = {
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
    return model_kwargs


def _create_formatter(config: Config) -> OpenAIChatFormatter | DeepSeekChatFormatter:
    """Create appropriate formatter based on model name."""
    model_name = (config.model.model_name or "").lower()
    return (
        DeepSeekChatFormatter() if "deepseek" in model_name else OpenAIChatFormatter()
    )


def create_agent(
    config: Config,
    name: str,
    sys_prompt: str,
    toolkit: Toolkit | None = None,
    enable_meta_tool: bool = False,
    max_iters: int = 10,
    plan_notebook: Any = None,
    parallel_tool_calls: bool = False,
) -> ReActAgent:
    """Create a ReActAgent with shared configuration.

    Args:
        config: Configuration object.
        name: Agent name.
        sys_prompt: System prompt.
        toolkit: Optional toolkit. None means no tools.
        enable_meta_tool: Whether to enable meta tool.
        max_iters: Maximum iterations.
        plan_notebook: Optional plan notebook.
        parallel_tool_calls: Whether to allow parallel tool calls.

    Returns:
        A new ReActAgent instance.
    """
    model_kwargs = _create_model_kwargs(config)
    agent_kwargs: dict[str, Any] = {
        "name": name,
        "sys_prompt": sys_prompt,
        "model": OpenAIChatModel(**model_kwargs),
        "memory": InMemoryMemory(),
        "formatter": _create_formatter(config),
        "toolkit": toolkit,
        "enable_meta_tool": enable_meta_tool,
        "max_iters": max_iters,
    }
    if plan_notebook is not None:
        agent_kwargs["plan_notebook"] = plan_notebook
    if parallel_tool_calls:
        agent_kwargs["parallel_tool_calls"] = True
    return ReActAgent(**agent_kwargs)


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
    available_tools = ", ".join(sorted(toolkit.tools.keys())) or "无"
    return create_agent(
        config=config,
        name=name,
        sys_prompt=f"""你是一个专业的助手。你的任务是完成分配给你的具体工作。

可用工具（仅以下这些）：{available_tools}

请：
1. 仔细分析任务需求
2. 必须优先使用工具完成需要外部信息/浏览器/文件操作的步骤
3. 如果任务完成，输出结构化终态：status=done, answer=最终答案, evidence=关键信息列表, errors=[], media_paths=生成的文件路径列表
4. 如果任务失败，输出结构化终态：status=failed, answer="", evidence=[], errors=[失败原因], media_paths=[]
5. 不要只停留在 tool call 或 thinking；必须给出终态
6. 如果生成了图片、文件等媒体资源，务必在 media_paths 中填写完整的文件绝对路径

如果有不清楚的地方，请说明需要更多信息。""",
        toolkit=toolkit,
        enable_meta_tool=False,
        max_iters=14,
    )
