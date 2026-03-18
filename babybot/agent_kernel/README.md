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
- `dynamic_orchestrator.py`: dynamic orchestration loop, child-task runtime, event bus, and snapshot persistence.

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
- Persists minimal flow snapshots through `FileChildTaskStateStore`.

4. `SingleAgentExecutor` (Worker Execution)
- Runs one sub-agent loop with model/tool calling.
- Enforces tool leases and loop guards.

5. `Heartbeat` / `TaskHeartbeatRegistry`
- Outer `Heartbeat` protects the whole user request from hanging.
- `TaskHeartbeatRegistry` tracks child-task liveness independently.

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

- `scheduler.py` is deprecated and kept only for backward compatibility.
- `engine.py` is a compatibility stub; new code should use `DynamicOrchestrator`.
- The current child-task runtime is still in-process, but it now has explicit event bus and snapshot boundaries for future durable transports.
