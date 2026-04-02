"""Minimal multi-agent orchestration kernel.

This package provides a dynamic orchestration loop via DynamicOrchestrator.
"""

from .context import ContextManager, ContextSnapshot
from .dag_ports import ResourceBridgeExecutor
from .dynamic_orchestrator import (
    ChildTaskEvent,
    DynamicOrchestrator,
    InMemoryChildTaskBus,
    InProcessChildTaskRuntime,
)
from .executor import EchoModelProvider, ExecutorPolicy, SingleAgentExecutor
from .model import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from .mcp import MCPClientPort, MCPToolAdapter, MCPToolDescriptor, register_mcp_tools
from .protocols import ExecutorPort
from .skills import SkillPack
from .tools import RegisteredTool, Tool, ToolContext, ToolRegistry, ToolResult
from .types import (
    AgentEvent,
    AgentEventKind,
    EventBus,
    EventSubscriber,
    ExecutionContext,
    ExecutionPlan,
    FinalResult,
    OrchestratorState,
    RunPolicy,
    SystemPromptBuilder,
    SystemPromptSection,
    TaskContract,
    TaskResult,
    ToolLease,
    WorkerState,
)

__all__ = [
    "AgentEvent",
    "AgentEventKind",
    "ContextManager",
    "ContextSnapshot",
    "ChildTaskEvent",
    "DynamicOrchestrator",
    "EventBus",
    "EventSubscriber",
    "ExecutionContext",
    "ExecutionPlan",
    "EchoModelProvider",
    "ExecutorPolicy",
    "ExecutorPort",
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
    "OrchestratorState",
    "RegisteredTool",
    "ResourceBridgeExecutor",
    "RunPolicy",
    "SingleAgentExecutor",
    "SkillPack",
    "SystemPromptBuilder",
    "SystemPromptSection",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "TaskContract",
    "TaskResult",
    "ToolLease",
    "WorkerState",
    "register_mcp_tools",
]
