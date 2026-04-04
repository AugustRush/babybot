"""Core types for the minimal orchestration kernel."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union
from typing import cast

try:
    from typing import TypedDict, NotRequired
except ImportError:  # Python <3.11
    from typing_extensions import TypedDict, NotRequired


TaskStatus = Literal["pending", "running", "succeeded", "failed", "blocked", "skipped"]


# ── Prompt Section Assembly ──────────────────────────────────────────────
# Inspired by Claude Code's layered prompt management: prompts as composable
# runtime sections rather than flat strings.  Each section carries metadata
# for caching, observability, and priority-based ordering.


@dataclass(frozen=True)
class SystemPromptSection:
    """One composable unit of a system prompt.

    Sections are assembled into a final string by priority (lower = earlier).
    The *name* serves as a stable key for caching and observability.
    """

    name: str
    content: str
    priority: int = 50  # 0-99; lower values appear first
    cacheable: bool = False  # hint: content is static across calls
    token_count: int = 0  # filled during assembly if desired


class SystemPromptBuilder:
    """Composable prompt assembler.

    Usage:
        builder = SystemPromptBuilder()
        builder.add("identity", "You are a Worker agent.", priority=0)
        builder.add("task", f"Task: {desc}", priority=10)
        builder.add("skills", catalog_text, priority=30)
        prompt_text = builder.build()
        sections = builder.sections  # for observability

    Design rationale (Claude Code analysis):
    - Prompts are runtime, not strings — assembled from typed sections
    - Section-level observability — each section emitted as event
    - Stable section names enable caching, diffing, and A/B testing
    """

    __slots__ = ("_sections",)

    def __init__(self) -> None:
        self._sections: list[SystemPromptSection] = []

    def add(
        self,
        name: str,
        content: str,
        *,
        priority: int = 50,
        cacheable: bool = False,
    ) -> "SystemPromptBuilder":
        """Add a named section. Returns self for chaining."""
        content = content.strip()
        if content:
            self._sections.append(
                SystemPromptSection(
                    name=name,
                    content=content,
                    priority=priority,
                    cacheable=cacheable,
                )
            )
        return self

    @property
    def sections(self) -> list[SystemPromptSection]:
        """Sections in assembly order (sorted by priority, stable)."""
        return sorted(self._sections, key=lambda s: s.priority)

    def build(self, separator: str = "\n") -> str:
        """Assemble all sections into a single prompt string."""
        return separator.join(s.content for s in self.sections)

    def section_names(self) -> list[str]:
        """Return section names in assembly order (for logging)."""
        return [s.name for s in self.sections]


# ── Structured Agent Events ──────────────────────────────────────────────
# Inspired by pi-agent's lifecycle events: typed, observable, subscribable.
# Each event carries a structured payload rather than a loose dict.

AgentEventKind = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "tool_execution_start",
    "tool_execution_end",
    "llm_request_start",
    "llm_request_end",
    "context_transform",
    "policy_decision",
    "prompt_section_assembled",
]


@dataclass(frozen=True)
class AgentEvent:
    """Structured lifecycle event emitted during agent execution.

    Fields:
        kind: Event type discriminator (see AgentEventKind).
        timestamp: Monotonic timestamp (time.monotonic()).
        session_id: Session that produced this event.
        task_id: Task being executed (if applicable).
        step: Current step number (if applicable).
        data: Event-specific payload; varies by kind.
    """

    kind: AgentEventKind
    timestamp: float = field(default_factory=time.monotonic)
    session_id: str = ""
    task_id: str = ""
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


# Subscriber callback type — receives an AgentEvent, may return None.
EventSubscriber = Callable[[AgentEvent], None]


class EventBus:
    """Typed event bus with subscription support.

    Replaces the untyped list[dict] pattern. Subscribers receive AgentEvent
    instances and can filter by kind.

    Usage:
        bus = EventBus()
        bus.subscribe(lambda e: print(e), kinds={"turn_start", "turn_end"})
        bus.emit(AgentEvent(kind="turn_start", task_id="t1", step=1))
    """

    __slots__ = ("_subscribers", "_events")

    def __init__(self) -> None:
        self._subscribers: list[tuple[EventSubscriber, frozenset[str] | None]] = []
        self._events: list[AgentEvent] = []

    def subscribe(
        self,
        callback: EventSubscriber,
        kinds: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Register a subscriber, optionally filtered to specific event kinds."""
        frozen = frozenset(kinds) if kinds else None
        self._subscribers.append((callback, frozen))

    def unsubscribe(self, callback: EventSubscriber) -> None:
        """Remove a subscriber."""
        self._subscribers = [
            (cb, kinds) for cb, kinds in self._subscribers if cb is not callback
        ]

    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all matching subscribers and record it."""
        self._events.append(event)
        for callback, kinds in self._subscribers:
            if kinds is not None and event.kind not in kinds:
                continue
            callback(event)

    @property
    def events(self) -> list[AgentEvent]:
        """All emitted events (for observability/testing)."""
        return self._events

    def clear(self) -> None:
        """Clear recorded events."""
        self._events.clear()


# ── Tool Lease ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolLease:
    """Least-privilege tool policy for one task."""

    include_groups: tuple[str, ...] = ()
    include_tools: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()
    include_groups_set: frozenset[str] = field(init=False, repr=False, compare=False)
    include_tools_set: frozenset[str] = field(init=False, repr=False, compare=False)
    exclude_tools_set: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_groups_set", frozenset(self.include_groups))
        object.__setattr__(self, "include_tools_set", frozenset(self.include_tools))
        object.__setattr__(self, "exclude_tools_set", frozenset(self.exclude_tools))


# ── Task / Plan / Result ─────────────────────────────────────────────────


@dataclass
class TaskContract:
    """Generic task unit in an execution plan."""

    task_id: str
    description: str
    deps: tuple[str, ...] = ()
    lease: ToolLease = field(default_factory=ToolLease)
    timeout_s: float | None = None
    retries: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    """DAG of executable tasks."""

    tasks: tuple[TaskContract, ...]
    rationale: str = ""


@dataclass
class TaskResult:
    """Execution result of one task."""

    task_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""
    artifacts: tuple[str, ...] = ()
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalResult:
    """Final result contract for a complete goal."""

    conclusion: str
    evidence: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    task_results: dict[str, TaskResult] = field(default_factory=dict)


@dataclass
class RunPolicy:
    """Execution policy for orchestration runtime."""

    max_parallel: int = 4
    default_timeout_s: float | None = None
    default_retries: int = 0


# ── Typed State Views ────────────────────────────────────────────────────
# ExecutionContext.state is a dict[str, Any] shared across all layers.
# These TypedDicts document the known key contracts without changing the
# runtime type — existing code can keep using state["key"] unmodified.
# New code can use context.worker_state / context.orchestrator_state for
# typed access and IDE completion.


class WorkerState(TypedDict, total=False):
    """Keys written/read by SingleAgentExecutor and subagent workers."""

    # ── Conversation history ──────────────────────────────────────────
    tape: NotRequired[Any]  # ConversationTape-like object
    tape_store: NotRequired[Any]  # Persistence store for tape entries
    memory_store: NotRequired[Any]  # Long-term memory store
    context_history_tokens: NotRequired[int]  # Budget for injected history
    max_model_tokens: NotRequired[int]  # Per-request token cap (0 = default)

    # ── Media / artifacts ────────────────────────────────────────────
    media_paths: NotRequired[tuple[str, ...] | list[str]]  # Attached media
    media_paths_collected: NotRequired[list[str]]  # Runtime collection bucket
    artifacts: NotRequired[list[str]]  # Output artifact paths

    # ── Runtime hints & callbacks ────────────────────────────────────
    pending_runtime_hints: NotRequired[list[Any]]  # Messages to inject next turn
    heartbeat: NotRequired[Any]  # Heartbeat object; call .beat()
    stream_callback: NotRequired[Any]  # Async streaming callback

    # ── Structured output slots ──────────────────────────────────────
    summary: NotRequired[str]
    decisions: NotRequired[list[str]]
    entities: NotRequired[list[str]]
    next_steps: NotRequired[list[str]]
    open_questions: NotRequired[list[str]]
    user_intent: NotRequired[str]
    pending: NotRequired[Any]


class OrchestratorState(TypedDict, total=False):
    """Keys written/read by DynamicOrchestrator and orchestration policy."""

    # ── Goal & plan ──────────────────────────────────────────────────
    original_goal: NotRequired[str]
    execution_plan: NotRequired[Any]  # ExecutionPlan or similar
    task_contract: NotRequired[Any]  # TaskContract for current task
    execution_constraints: NotRequired[str]

    # ── Policy & routing ────────────────────────────────────────────
    policy_hints: NotRequired[str]
    resource_id: NotRequired[str]
    resource_ids: NotRequired[list[str]]
    upstream_results: NotRequired[dict[str, Any]]

    # ── Progress & status ───────────────────────────────────────────
    status: NotRequired[str]
    progress: NotRequired[str]

    # ── Callbacks ────────────────────────────────────────────────────
    runtime_event_callback: NotRequired[Any]
    send_intermediate_message: NotRequired[Any]
    stream_callback: NotRequired[Any]
    heartbeat: NotRequired[Any]


class NotebookState(TypedDict, total=False):
    """Keys written/read by notebook runtime and context builders."""

    plan_notebook: NotRequired[Any]
    plan_notebook_id: NotRequired[str]
    current_notebook_node_id: NotRequired[str]
    notebook_context_budget: NotRequired[int]


# ── Execution Context ────────────────────────────────────────────────────


@dataclass
class ExecutionContext:
    """Runtime context shared by planner, executor, and synthesizer.

    Dual event interface (backward compatible):
    - ``emit(event_str, **payload)`` — legacy dict-based emission; writes to
      ``self.events`` list AND the structured ``EventBus`` (as ``AgentEvent``
      with ``kind="policy_decision"`` or generic kind).
    - ``event_bus`` — typed EventBus for structured lifecycle events.
    - ``on(kinds, callback)`` — convenience subscription shortcut.
    """

    session_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    event_bus: EventBus = field(default_factory=EventBus)

    # ── Hook points (inspired by pi-agent) ───────────────────────────
    # These are optional callbacks the host can set.
    # transform_context: called before each LLM request with the message list,
    #   returns a (possibly modified) message list.
    # before_tool_call / after_tool_call: called around tool invocations.
    transform_context: Any = field(default=None, repr=False)
    before_tool_call: Any = field(default=None, repr=False)
    after_tool_call: Any = field(default=None, repr=False)

    # ── Typed state views ─────────────────────────────────────────────
    # These properties expose the same underlying dict with TypedDict types
    # so new code gets IDE completion; existing state["key"] usage is unchanged.

    @property
    def worker_state(self) -> WorkerState:
        """Typed view of state keys used by worker/executor layer."""
        return cast(WorkerState, self.state)

    @property
    def orchestrator_state(self) -> OrchestratorState:
        """Typed view of state keys used by orchestrator/policy layer."""
        return cast(OrchestratorState, self.state)

    @property
    def notebook_state(self) -> NotebookState:
        """Typed view of state keys used by notebook runtime."""
        return cast(NotebookState, self.state)

    def emit(self, event: str, **payload: Any) -> None:
        """Legacy dict-based event emission — also bridges to EventBus."""
        entry = {"event": event, **payload}
        self.events.append(entry)
        # Bridge to structured EventBus so subscribers see legacy events too.
        kind = event if event in _KNOWN_EVENT_KINDS else "policy_decision"
        self.event_bus.emit(
            AgentEvent(
                kind=kind,  # type: ignore[arg-type]
                session_id=self.session_id,
                task_id=str(payload.get("task_id", "")),
                step=int(payload.get("step", 0) or 0),
                data=payload,
            )
        )

    def on(
        self,
        kinds: set[str] | frozenset[str] | None,
        callback: EventSubscriber,
    ) -> None:
        """Convenience: subscribe to structured events on this context."""
        self.event_bus.subscribe(callback, kinds=kinds)


# Known event kinds for bridging legacy emit() calls.
_KNOWN_EVENT_KINDS: frozenset[str] = frozenset(AgentEventKind.__args__)  # type: ignore[attr-defined]
