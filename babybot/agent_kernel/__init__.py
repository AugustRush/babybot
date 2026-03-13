"""Minimal multi-agent orchestration kernel.

This package intentionally keeps one built-in orchestration mode:
Planner -> Executor -> Synthesizer.
"""

from .context import ContextManager, ContextSnapshot
from .engine import PlanValidationError, WorkflowEngine
from .executor import EchoModelProvider, ExecutorPolicy, SingleAgentExecutor
from .model import ModelMessage, ModelProvider, ModelRequest, ModelResponse, ModelToolCall
from .mcp import MCPClientPort, MCPToolAdapter, MCPToolDescriptor, register_mcp_tools
from .protocols import ExecutorPort, PlannerPort, SynthesizerPort
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
    "ExecutionContext",
    "ExecutionPlan",
    "EchoModelProvider",
    "ExecutorPolicy",
    "ExecutorPort",
    "FinalResult",
    "MCPClientPort",
    "MCPToolAdapter",
    "ModelMessage",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelToolCall",
    "MCPToolDescriptor",
    "PlanValidationError",
    "PlannerPort",
    "RegisteredTool",
    "RunPolicy",
    "SingleAgentExecutor",
    "SkillPack",
    "SynthesizerPort",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "TaskContract",
    "TaskResult",
    "ToolLease",
    "WorkflowEngine",
    "register_mcp_tools",
]
