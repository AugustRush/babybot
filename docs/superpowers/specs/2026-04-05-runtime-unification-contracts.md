# Runtime Unification Contracts

## Goal

Keep complex-task execution stable by making the runtime surface small, explicit, and reusable.

This document records the contract introduced by the runtime unification refactor. New orchestration features should extend these contracts instead of adding new ad-hoc state keys, prompt builders, or notebook mutations inside hot-path orchestrator code.

## Unified Runtime State

Primary entrypoint: `babybot.agent_kernel.runtime_state.RuntimeState`

Rules:

- Read shared execution state through `RuntimeState` instead of directly indexing `context.state` in hot paths.
- Use `RuntimeState.notebook_binding()` for notebook/node resolution. Stale node ids must fall back to the notebook root/frontier instead of failing open.
- Use `RuntimeState.collected_media_bucket()` / `extend_collected_media()` for artifact aggregation.
- Use `RuntimeState.upstream_results_bucket()` and `notebook_task_map()` for mutable execution buckets.
- Use `RuntimeState.policy_hints()` / `set_policy_hints()` for orchestrator hint persistence.

Current canonical runtime buckets:

- `plan_notebook`
- `plan_notebook_id`
- `current_notebook_node_id`
- `notebook_context_budget`
- `media_paths`
- `media_paths_collected`
- `upstream_results`
- `notebook_task_map`
- `policy_hints`
- `pending_runtime_hints`

## Unified Prompt Assembly

Primary entrypoints:

- `babybot.agent_kernel.types.SystemPromptBuilder`
- `babybot.agent_kernel.prompt_assembly.add_text_section`
- `babybot.agent_kernel.prompt_assembly.add_list_section`
- `babybot.agent_kernel.prompt_assembly.dedupe_prompt_items`

Rules:

- Worker and orchestrator prompts should be built as named sections, not flat string concatenation.
- Repeated prompt-list logic must go through `dedupe_prompt_items()`.
- Section formatting should use shared helpers so headers, bullets, and trimming stay consistent.

## Exploration Budget Contract

Primary entrypoint: `babybot.agent_kernel.executor.SingleAgentExecutor`

Rules:

- Read/search/check-only loops are bounded by `max_no_progress_turns`.
- After the exploration budget is exhausted, the executor may grant one finalize turn.
- If the model still chooses read/search/check tools on that finalize turn, the executor must auto-converge into a deterministic evidence/blocker summary.
- This branch should not depend on another model round to finish.
- The auto-converged result is reported with metadata:
  - `auto_converged = True`
  - `completion_mode = "auto_summary_after_exploration_stall"`

## Notebook Runtime Boundary

Primary entrypoint: `babybot.agent_kernel.orchestrator_notebook.NotebookRuntimeHelper`

Responsibilities:

- Create or rebind the plan notebook for a request.
- Maintain notebook task maps and node lookup.
- Materialize team nodes and team-task children.
- Project child-task results back into notebook state.
- Manage force-converge activation/clearing.
- Finalize notebook state before reply.

Non-goals:

- Model calling
- dispatch/wait tool schema management
- resource selection

## Child Task Policy Boundary

Primary entrypoint: `babybot.agent_kernel.orchestrator_child_tasks.ChildTaskRuntimeHelper`

Responsibilities:

- Merge maintenance-mode dispatches into one bounded child task.
- Determine serialized dependency chains for maintenance work.
- Normalize child-task prompt payloads through configured prompt builders.
- Select recent successful upstream results for reuse.

Non-goals:

- child task execution
- notebook mutation
- runtime event publication

## Public Facade

The following helpers are part of the public kernel contract and are exported from `babybot.agent_kernel`:

- `RuntimeState`
- `NotebookRuntimeHelper`
- `ChildTaskRuntimeHelper`

## Extension Rule

Before adding new orchestrator state, prompt sections, or notebook behavior:

1. Check whether it belongs in `RuntimeState`, `NotebookRuntimeHelper`, or `ChildTaskRuntimeHelper`.
2. Add or update a contract test first.
3. Prefer extending an existing bucket/helper over introducing a new hot-path code path in `DynamicOrchestrator`.
