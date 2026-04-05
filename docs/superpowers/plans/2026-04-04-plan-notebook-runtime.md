# Plan Notebook Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a durable `Plan Notebook` runtime model that becomes the canonical source of execution state, step context, repair checkpoints, and completion summaries for complex multi-agent tasks.

**Architecture:** Keep the current `TaskContract` and `ExecutionPlan` as entry contracts, but compile them into a runtime `PlanNotebook` that owns append-only execution records plus derived projections. The orchestrator, worker executor, feedback renderer, and persistence layer will all read and write notebook state, while prompt construction will switch from ad-hoc `context.state` assembly to budgeted notebook-derived context views. Completion persistence will store both a compact summary/index layer and a full searchable raw event layer.

**Tech Stack:** Python dataclasses, existing `ExecutionContext`, SQLite (`runtime_job_store`), existing `memory_store`, current orchestration/event bus infrastructure, optional SQLite FTS5 with LIKE fallback.

---

### Task 1: Define Notebook Domain Model

**Files:**
- Create: `babybot/agent_kernel/plan_notebook.py`
- Modify: `babybot/agent_kernel/types.py`
- Test: `tests/test_plan_notebook.py`

- [ ] **Step 1: Write failing tests for notebook creation, node lifecycle, and append-only event recording**

Run: `pytest -q tests/test_plan_notebook.py`
Expected: FAIL because notebook module and types do not exist yet.

- [ ] **Step 2: Add notebook core types**

Implement dataclasses / typed helpers for:
- `PlanNotebook`
- `NotebookNode`
- `NotebookEvent`
- `NotebookCheckpoint`
- `NotebookArtifact`
- `NotebookIssue`
- `NotebookDecision`

Required behaviors:
- root notebook creation from goal / plan / flow metadata
- child node creation with `parent_id`, `deps`, `owner`, `resource_ids`
- append-only event recording
- derived status transitions with validation

- [ ] **Step 3: Extend `ExecutionContext` typed state views**

Add typed accessors / state contracts for:
- `plan_notebook`
- `plan_notebook_id`
- `current_notebook_node_id`
- `notebook_context_budget`

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_plan_notebook.py`
Expected: PASS


### Task 2: Build Notebook Projection and Context View Layer

**Files:**
- Create: `babybot/agent_kernel/plan_notebook_context.py`
- Modify: `babybot/agent_kernel/plan_notebook.py`
- Modify: `babybot/context_views.py`
- Test: `tests/test_plan_notebook_context.py`

- [ ] **Step 1: Write failing tests for budgeted context projection**

Cover:
- current-step context only includes high-priority slots
- direct deps outrank older background events
- full notebook remains queryable even when prompt context is compacted
- completion summary view excludes low-signal raw noise

Run: `pytest -q tests/test_plan_notebook_context.py`
Expected: FAIL

- [ ] **Step 2: Implement notebook projection slots**

Implement builders for:
- `build_orchestrator_context_view(notebook, token_budget)`
- `build_worker_context_view(notebook, node_id, token_budget)`
- `build_completion_context_view(notebook, token_budget)`

Required slot priority:
1. goal + hard constraints
2. current objective
3. direct dependency outputs
4. latest verified decisions
5. blockers / open questions
6. artifact manifest
7. optional raw excerpts

- [ ] **Step 3: Integrate notebook-aware memory summary hooks**

Update `context_views.py` to prefer notebook completion summaries and indexed notebook references over raw ephemeral task-state blobs when both are present.

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_plan_notebook_context.py`
Expected: PASS


### Task 3: Add Durable Notebook Persistence and Searchable Indexes

**Files:**
- Create: `babybot/agent_kernel/plan_notebook_store.py`
- Modify: `babybot/runtime_job_store.py`
- Modify: `babybot/runtime_jobs.py`
- Modify: `babybot/memory_store.py`
- Test: `tests/test_plan_notebook_store.py`
- Test: `tests/test_runtime_job_store.py`

- [ ] **Step 1: Write failing persistence tests**

Cover:
- notebook snapshot create/load/update
- append-only raw entry persistence
- completion summary persistence
- raw full-text lookup by notebook / chat / flow
- FTS-disabled fallback path

Run: `pytest -q tests/test_plan_notebook_store.py tests/test_runtime_job_store.py`
Expected: FAIL

- [ ] **Step 2: Add notebook persistence schema**

Add SQLite tables for:
- notebook headers
- notebook nodes
- notebook events
- notebook summary/index rows

If FTS5 is available:
- create FTS virtual table for raw text search

Otherwise:
- keep normalized `search_text` columns and SQL `LIKE` fallback

- [ ] **Step 3: Add completion summary/index writer**

Persist at least:
- `final_summary`
- `decision_register`
- `artifact_manifest`
- `open_followups`
- `node_summaries`
- `search_terms`

- [ ] **Step 4: Update memory store integration**

Add notebook-derived completion observations so future context retrieval uses compact structured summaries instead of replaying raw logs.

- [ ] **Step 5: Run tests**

Run: `pytest -q tests/test_plan_notebook_store.py tests/test_runtime_job_store.py`
Expected: PASS


### Task 4: Compile Contracts and Plans Into a Root Notebook

**Files:**
- Modify: `babybot/task_contract.py`
- Modify: `babybot/execution_plan.py`
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/agent_kernel/types.py`
- Test: `tests/test_execution_plan.py`
- Test: `tests/test_orchestrator_routing.py`

- [ ] **Step 1: Write failing tests for notebook bootstrap**

Cover:
- every task gets a root notebook before orchestration starts
- `TaskContract` / `ExecutionPlan` are represented as initial notebook nodes
- context carries notebook id and root node id

Run: `pytest -q tests/test_execution_plan.py tests/test_orchestrator_routing.py -k 'notebook or execution_plan'`
Expected: FAIL

- [ ] **Step 2: Add notebook bootstrap in orchestrator entrypoint**

When the top-level task is created:
- create root notebook from `TaskContract`
- compile first plan steps into notebook nodes
- attach notebook to `ExecutionContext.state`
- attach notebook id / root node id / context budget defaults

- [ ] **Step 3: Keep backward-compatible execution plan access**

Do not remove `execution_plan` yet; preserve it as an entry artifact while routing runtime logic through notebook state.

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_execution_plan.py tests/test_orchestrator_routing.py`
Expected: PASS


### Task 5: Refactor Dynamic Orchestrator to be Notebook-Driven

**Files:**
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/orchestrator_prompts.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] **Step 1: Write failing tests for notebook-backed orchestration**

Cover:
- same-turn maintenance dispatches are recorded as a single owner path in notebook
- later maintenance follow-ups serialize onto the existing notebook frontier
- `wait_for_tasks` updates notebook node state before further dispatch
- `reply_to_user` requires notebook finalization readiness
- fallback summary is derived from notebook rather than ad-hoc runtime results

Run: `pytest -q tests/test_dynamic_orchestrator.py -k 'maintenance or notebook or fallback'`
Expected: FAIL

- [ ] **Step 2: Replace prompt assembly with notebook context projection**

`_build_initial_messages()` should read:
- root notebook summary
- active frontier nodes
- current checkpoints
- high-priority decisions / blockers

Avoid injecting raw full notebook data unless explicitly requested by the current projection builder.

- [ ] **Step 3: Route orchestration tool calls through notebook mutations**

For each tool:
- `dispatch_task`: create child notebook node first, then dispatch runtime task
- `wait_for_tasks`: merge completed child results into notebook frontier
- `get_task_result`: reflect current node state, not just raw runtime result
- `reply_to_user`: finalize notebook summary and verify no unresolved blocking nodes
- `dispatch_team`: create team node and child participant/task nodes

- [ ] **Step 4: Move maintenance / anti-loop rules onto notebook frontier**

Use notebook state to enforce:
- single-owner maintenance flow
- no parallel sibling fanout without explicit parallel intent
- no fresh no-deps redispatch when a live or recently completed notebook chain already exists

- [ ] **Step 5: Run tests**

Run: `pytest -q tests/test_dynamic_orchestrator.py`
Expected: PASS


### Task 6: Refactor Worker Executor to Use Notebook as Its Work Log

**Files:**
- Modify: `babybot/agent_kernel/executor.py`
- Modify: `babybot/resource_subagent_runtime.py`
- Modify: `babybot/agent_kernel/dag_ports.py`
- Test: `tests/test_agent_kernel_executor.py`

- [ ] **Step 1: Write failing tests for executor notebook logging and no-progress semantics**

Cover:
- each model turn/tool call/tool result creates notebook events
- no-progress uses notebook advancement, not only “tool-only turns”
- artifact detection updates notebook artifact manifest
- worker result writes structured step summary back into notebook

Run: `pytest -q tests/test_agent_kernel_executor.py`
Expected: FAIL

- [ ] **Step 2: Add per-worker notebook node binding**

When a child task starts:
- bind it to a notebook node id
- attach node id to `ExecutionContext`
- log turn starts, tool decisions, failures, retries, and completion to notebook

- [ ] **Step 3: Replace pure turn-count no-progress checks with notebook advancement checks**

A turn counts as progress only when it adds at least one of:
- verified observation
- decision
- artifact
- completed substep
- blocker transition

- [ ] **Step 4: Update bridge enrichment to use notebook-derived upstream context**

Replace raw `upstream_results` string appends with context projection built from:
- direct dependency outputs
- current node objective
- latest relevant decisions

- [ ] **Step 5: Run tests**

Run: `pytest -q tests/test_agent_kernel_executor.py tests/test_dynamic_orchestrator.py -k 'executor or upstream or no_progress'`
Expected: PASS


### Task 7: Add Repair, Checkpoint, and Human-Feedback Integration

**Files:**
- Modify: `babybot/agent_kernel/plan_notebook.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/feedback_events.py`
- Modify: `babybot/orchestrator.py`
- Test: `tests/test_feedback_events.py`
- Test: `tests/test_dynamic_orchestrator.py`

- [ ] **Step 1: Write failing tests for notebook checkpoints**

Cover:
- `needs_repair`
- `needs_human_input`
- `verification_failed`
- `ready_to_finalize`

Run: `pytest -q tests/test_feedback_events.py tests/test_dynamic_orchestrator.py -k 'checkpoint or repair or feedback'`
Expected: FAIL

- [ ] **Step 2: Add checkpoint transitions and repair branches**

Notebook should support:
- promoting a failed node into repair
- preserving lineage to the original failed node
- pausing on human-input checkpoints without losing frontier state

- [ ] **Step 3: Make feedback rendering notebook-aware**

Render from notebook state:
- current phase
- current owner
- completed step labels
- blockers
- next action

Do not render full internal prompts or raw descriptions.

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_feedback_events.py tests/test_dynamic_orchestrator.py`
Expected: PASS


### Task 8: Persist Completion Summaries and Queryable Context for Future Tasks

**Files:**
- Modify: `babybot/orchestrator.py`
- Modify: `babybot/context_views.py`
- Modify: `babybot/memory_store.py`
- Modify: `babybot/agent_kernel/plan_notebook_store.py`
- Test: `tests/test_context_views.py`
- Test: `tests/test_memory_store.py`

- [ ] **Step 1: Write failing tests for post-completion summary reuse**

Cover:
- completion writes compact notebook summary + index rows
- later related tasks load summary layer by default
- raw notebook events remain queryable on demand
- context views stay concise under large-task history

Run: `pytest -q tests/test_context_views.py tests/test_memory_store.py`
Expected: FAIL

- [ ] **Step 2: Write completion summary pipeline**

At task completion:
- derive completion view from notebook
- persist summary and searchable references
- emit compact memory observations instead of raw step spam

- [ ] **Step 3: Update future context retrieval**

Later tasks should retrieve:
- compact completion summary first
- node-level summary second
- raw full-text excerpts only when query relevance requires them

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_context_views.py tests/test_memory_store.py`
Expected: PASS


### Task 9: Compatibility Layer and Incremental Migration Cleanup

**Files:**
- Modify: `babybot/agent_kernel/types.py`
- Modify: `babybot/agent_kernel/dag_ports.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`
- Modify: `babybot/agent_kernel/executor.py`
- Test: `tests/test_dynamic_orchestrator.py`
- Test: `tests/test_agent_kernel_executor.py`

- [ ] **Step 1: Add temporary compatibility shims**

Support existing consumers of:
- `context.state["upstream_results"]`
- `context.state["policy_hints"]`
- `context.state["pending_runtime_hints"]`
- `task_state_snapshot`

by deriving them from notebook state until all call sites are migrated.

- [ ] **Step 2: Remove duplicated runtime state writes where notebook is canonical**

Keep only compatibility mirrors that are still required by callers / tests.

- [ ] **Step 3: Run targeted regression suite**

Run: `pytest -q tests/test_agent_kernel_executor.py tests/test_dynamic_orchestrator.py tests/test_orchestrator_routing.py`
Expected: PASS


### Task 10: End-to-End Verification and Performance Guardrails

**Files:**
- Modify: `tests/test_dynamic_orchestrator.py`
- Modify: `tests/test_agent_kernel_executor.py`
- Modify: `tests/test_orchestrator_routing.py`
- Create: `tests/test_plan_notebook_integration.py`

- [ ] **Step 1: Add integration tests for long multi-step notebook workflows**

Cover:
- complex maintenance goal with repo reference + local repair + verification
- multi-step task with one failure, one repair, one final summary
- large notebook with compact prompt context and full raw retrieval

- [ ] **Step 2: Add token-budget / context-size assertions**

Verify that notebook-derived prompt views are smaller than raw full-log injection and remain bounded as task length grows.

- [ ] **Step 3: Run full focused verification**

Run: `pytest -q tests/test_plan_notebook.py tests/test_plan_notebook_context.py tests/test_plan_notebook_store.py tests/test_plan_notebook_integration.py tests/test_agent_kernel_executor.py tests/test_dynamic_orchestrator.py tests/test_orchestrator_routing.py tests/test_execution_plan.py tests/test_context_views.py tests/test_memory_store.py`
Expected: PASS

- [ ] **Step 4: Run broader regression checkpoint**

Run: `pytest -q`
Expected: PASS or a short explicit list of unrelated pre-existing failures.


### Design Notes

- The notebook is the canonical runtime state. `ExecutionPlan` remains the entry contract, not the long-lived execution source of truth.
- Prompt construction must always come from projection builders with explicit budgets; raw notebook logs are never injected wholesale.
- Durable storage must preserve both:
  - compact summary/index views for default retrieval
  - raw append-only full text for traceability and on-demand lookup
- Repair and feedback logic should be modeled as notebook checkpoints instead of ad-hoc strings in `context.state`.
- Compatibility shims are temporary and should be removed only after the notebook path is stable and covered by tests.
