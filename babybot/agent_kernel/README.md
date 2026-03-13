# Agent Kernel Architecture

This folder implements a minimal orchestration framework with one built-in mode:

`Planner -> Executor -> Synthesizer`

## Components

- `model.py`: model request/response contracts and provider port.
- `context.py`: context manager, snapshot, restore, and fork utilities.
- `tools.py`: tool protocol, normalized result, and lease-aware registry.
- `mcp.py`: MCP client port and MCP-to-tool adapter.
- `skills.py`: skill pack and prompt/lease composition helpers.
- `executor.py`: reference single-agent executor built on model/tools/skills.

## Layering (SOLID)

1. `types.py` (Domain Contracts)
- Immutable task/plan contracts and runtime result types.
- No infrastructure dependency.

2. `protocols.py` (Ports)
- `PlannerPort`, `ExecutorPort`, `SynthesizerPort`.
- Framework depends on abstractions, not concrete business implementations.

3. `engine.py` (Application Orchestrator)
- Validates plan DAG.
- Executes tasks with dependency-aware parallel scheduling.
- Handles timeout/retry/failure propagation.
- Calls synthesizer to produce final output.
 
4. `Adapters` (outside this folder in real apps)
- Implement `ModelProvider`, `Tool`, `MCPClientPort`, planner/synthesizer ports.
- Business code belongs in adapters, not in kernel.

## Extensibility Rules

1. Business extends by implementing ports in adapters.
2. Engine is not aware of business models.
3. New tools/skills/models are injected through `ExecutionContext` and adapter implementations.
4. Do not add extra built-in orchestration modes in kernel.

## Public API

- `WorkflowEngine.run(goal, context) -> FinalResult`
- `PlannerPort.plan(goal, context) -> ExecutionPlan`
- `ExecutorPort.execute(task, context) -> TaskResult`
- `SynthesizerPort.synthesize(goal, plan, results, context) -> FinalResult`
