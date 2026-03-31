# Agent Kernel Architecture

This folder implements a minimal orchestration framework centered on a dynamic main-agent loop.

Current runtime shape:

`DynamicOrchestrator -> ChildTaskRuntime -> SingleAgentExecutor`

## Components

- `model.py`: model request/response contracts and provider port.
- `context.py`: context manager, snapshot, restore, and fork utilities.
- `tools.py`: tool protocol, normalized result, and lease-aware registry.
- `mcp.py`: MCP client port and MCP-to-tool adapter.
- `skills.py`: skill pack and prompt/lease composition helpers.
- `executor.py`: reference single-agent executor built on model/tools/skills.
- `dynamic_orchestrator.py`: dynamic orchestration loop, child-task runtime, and event bus.

## Runtime Layers

1. `types.py` (Domain Contracts)
- Immutable task/plan contracts and runtime result types.
- No infrastructure dependency.

2. `DynamicOrchestrator` (Main-Agent Loop)
- Asks the model what to do next.
- Dispatches child tasks through an internal runtime adapter.
- Waits on task results and decides when to reply to the user.

3. `InProcessChildTaskRuntime` (Current Child-Task Transport)
- Owns in-flight tasks and completed task results.
- Emits lifecycle events through `InMemoryChildTaskBus`.
- Applies runtime-level retry classification for transient child-task failures.
- Captures terminal child-task failures into dead-letter records.

4. `SingleAgentExecutor` (Worker Execution)
- Runs one sub-agent loop with model/tool calling.
- Enforces tool leases and loop guards.

5. `Heartbeat` / `TaskHeartbeatRegistry`
- Outer `Heartbeat` protects the whole user request from hanging.
- `TaskHeartbeatRegistry` tracks child-task liveness independently.

## Runtime Boundaries

- Ingress `MessageBus`: owns inbound user/scheduled message queuing, concurrency limits, request watchdogs, and optional runtime-event capture for observability.
- Child-task bus: `InMemoryChildTaskBus` carries orchestration-local child lifecycle events such as `queued`, `started`, `retrying`, `succeeded`, `dead_lettered`, and `stalled`.
- Heartbeat registry: `TaskHeartbeatRegistry` is separate from both buses and answers "is this child task still healthy?" using per-task liveness records.
- Current bridge point: `OrchestratorAgent` wires the ingress bus to the dynamic orchestrator by passing an optional runtime-event callback while preserving the existing public request entrypoints.

## Extensibility Rules

1. Keep orchestration decisions in `DynamicOrchestrator`, not in worker executors.
2. Keep child-task transport behind runtime/store abstractions so it can later move off-process.
3. New tools/skills/models are injected through `ExecutionContext` and adapter implementations.
4. Legacy static-DAG modules are compatibility shims, not the preferred extension path.

## Public API

- `DynamicOrchestrator.run(goal, context) -> FinalResult`
- `SingleAgentExecutor.execute(task, context) -> TaskResult`
- `ResourceBridgeExecutor.execute(task, context) -> TaskResult`

## Compatibility Notes

- The current child-task runtime is still in-process, with explicit event bus and retry/dead-letter boundaries for future transport changes.
