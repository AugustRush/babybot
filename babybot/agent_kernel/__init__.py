"""Minimal multi-agent orchestration kernel.

This package provides a dynamic orchestration loop via DynamicOrchestrator.
"""

from .context import ContextManager, ContextSnapshot
from .orchestrator_child_tasks import ChildTaskRuntimeHelper
from .orchestrator_notebook import NotebookRuntimeHelper
from .dag_ports import ResourceBridgeExecutor
from .dynamic_orchestrator import (
    ChildTaskEvent,
    DynamicOrchestrator,
    InMemoryChildTaskBus,
    InProcessChildTaskRuntime,
)
from .executor import EchoModelProvider, ExecutorPolicy, SingleAgentExecutor
from .lease_utils import filter_tool_lease, lease_to_dict, merge_tool_leases
from .model import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from .mcp import MCPClientPort, MCPToolAdapter, MCPToolDescriptor, register_mcp_tools
from .plan_notebook import (
    NotebookArtifact,
    NotebookCheckpoint,
    NotebookDecision,
    NotebookEvent,
    NotebookIssue,
    NotebookNode,
    PlanNotebook,
    create_root_notebook,
)
from .protocols import ExecutorPort
from .runtime_state import NotebookBinding, RuntimeState
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
    NotebookState,
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
    "ChildTaskRuntimeHelper",
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
    "NotebookBinding",
    "NotebookRuntimeHelper",
    "NotebookArtifact",
    "NotebookCheckpoint",
    "NotebookDecision",
    "NotebookEvent",
    "NotebookIssue",
    "NotebookNode",
    "NotebookState",
    "OrchestratorState",
    "PlanNotebook",
    "RegisteredTool",
    "ResourceBridgeExecutor",
    "RunPolicy",
    "RuntimeState",
    "SingleAgentExecutor",
    "SkillPack",
    "SystemPromptBuilder",
    "SystemPromptSection",
    "Tool",
    "ToolContext",
    "filter_tool_lease",
    "lease_to_dict",
    "merge_tool_leases",
    "ToolRegistry",
    "ToolResult",
    "TaskContract",
    "TaskResult",
    "ToolLease",
    "WorkerState",
    "create_root_notebook",
    "register_mcp_tools",
]
