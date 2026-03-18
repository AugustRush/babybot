"""Minimal multi-agent orchestration kernel.

This package provides a dynamic orchestration loop via DynamicOrchestrator.
"""

from .context import ContextManager, ContextSnapshot
from .dag_ports import ResourceBridgeExecutor
from .dynamic_orchestrator import (
    ChildTaskEvent,
    DynamicOrchestrator,
    FileChildTaskStateStore,
    InMemoryChildTaskBus,
    InProcessChildTaskRuntime,
)
from .executor import EchoModelProvider, ExecutorPolicy, SingleAgentExecutor
from .model import ModelMessage, ModelProvider, ModelRequest, ModelResponse, ModelToolCall
from .mcp import MCPClientPort, MCPToolAdapter, MCPToolDescriptor, register_mcp_tools
from .protocols import ExecutorPort
from .skills import SkillPack
from .tools import RegisteredTool, Tool, ToolContext, ToolRegistry, ToolResult
from .types import (
    ExecutionContext,
    ExecutionPlan,
    FinalResult,
    RunPolicy,
    TaskContract,
    TaskResult,
    ToolLease,
)

__all__ = [
    "ContextManager",
    "ContextSnapshot",
    "ChildTaskEvent",
    "DynamicOrchestrator",
    "ExecutionContext",
    "ExecutionPlan",
    "EchoModelProvider",
    "ExecutorPolicy",
    "ExecutorPort",
    "FileChildTaskStateStore",
    "FinalResult",
    "InMemoryChildTaskBus",
    "InProcessChildTaskRuntime",
    "MCPClientPort",
    "MCPToolAdapter",
    "ModelMessage",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelToolCall",
    "MCPToolDescriptor",
    "RegisteredTool",
    "ResourceBridgeExecutor",
    "RunPolicy",
    "SingleAgentExecutor",
    "SkillPack",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "TaskContract",
    "TaskResult",
    "ToolLease",
    "register_mcp_tools",
]
