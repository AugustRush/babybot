# Large File Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split 6 overweight files (1200–2000 lines each) into focused modules without breaking any public API.

**Architecture:** Pure extraction — move cohesive chunks (free functions, utility methods, schema constants) into dedicated sub-modules, then re-import and re-export everything from the original file so all existing callers and tests continue to work unchanged. No class renames, no protocol changes.

**Tech Stack:** Python 3.12, pytest, asyncio, SQLite (sqlite3), lark-oapi (Feishu SDK)

---

## Ground Rules

1. **Backward compat is non-negotiable.** Every symbol that was importable from the original file must remain importable from the original file after the split, even if its definition has moved.
2. **Run `pytest tests/ -q --tb=short` after every task.** All 849 tests must pass before committing.
3. **One commit per task.** Keep the diff reviewable.
4. **No logic changes.** Pure moves only. If a method needs a small signature change to become a free function (e.g. removing unused `self`), that is allowed, but the class wrapper must keep identical behavior.

---

## File Map (after all tasks)

| New file | Content | Source |
|---|---|---|
| `babybot/channels/feishu_content_extractor.py` | 4 pure content-extraction functions | `feishu.py:47–243` |
| `babybot/channels/feishu_card_builder.py` | Card/markdown formatting helpers | `feishu.py:803–1036` |
| `babybot/agent_kernel/executor_history.py` | History-building free functions | `executor.py:38–71, 1665–1915` |
| `babybot/policy_store_utils.py` | Pure utility functions for policy store | `orchestration_policy_store.py:1156–1197` |
| `babybot/resource_path_utils.py` | Path-detection and artifact-collection free functions | `resource.py:82–215` |
| `babybot/resource_callable_tool.py` | `CallableTool` class + `override_current_write_root` context manager | `resource.py:218–355` |
| `babybot/agent_kernel/dynamic_orchestrator_tools.py` | Orchestration tool schema constants | `dynamic_orchestrator.py:78–291` |
| `babybot/agent_kernel/dynamic_orchestrator_prompt.py` | `_build_initial_messages` and resource catalog helpers | `dynamic_orchestrator.py:818–940`, free functions |
| `babybot/orchestrator_routing_cascade.py` | Full routing cascade methods extracted from `OrchestratorAgent` | `orchestrator.py:687–938` |

---

## Task 1: feishu.py → feishu_content_extractor.py

Extract 4 pure module-level functions that parse inbound Feishu message payloads. They have zero dependency on `FeishuChannel` state.

**Files:**
- Create: `babybot/channels/feishu_content_extractor.py`
- Modify: `babybot/channels/feishu.py`

- [ ] **Step 1: Read the source lines**

  Read `feishu.py` lines 47–243 to confirm exact function signatures and imports needed:
  ```
  _extract_share_card_content(content_json, msg_type) -> str
  _extract_interactive_content(content_json) -> str
  _extract_element_content(el) -> str
  _extract_post_content(content_json) -> tuple[str, list[str]]
  ```

- [ ] **Step 2: Create `feishu_content_extractor.py`**

  New file contains only those 4 functions + their docstrings + the `from __future__ import annotations` header. No imports needed beyond stdlib.

  ```python
  """Pure content-extraction helpers for inbound Feishu messages."""

  from __future__ import annotations

  from typing import Any

  # ── paste the 4 functions verbatim here ──
  ```

- [ ] **Step 3: Update `feishu.py`**

  Replace the function bodies in `feishu.py` lines 47–243 with an import + re-export block:

  ```python
  from .feishu_content_extractor import (
      _extract_element_content,
      _extract_interactive_content,
      _extract_post_content,
      _extract_share_card_content,
  )

  __all__ = [
      # ... existing __all__ if present, or just add these so tests that do
      # `from babybot.channels.feishu import _extract_interactive_content` keep working
  ]
  ```

  Note: the test file `tests/test_feishu_format.py` imports `_extract_interactive_content` directly from `babybot.channels.feishu`, so the re-import at module level is sufficient — no `__all__` entry needed.

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/test_feishu_format.py tests/test_feishu_streaming.py -q --tb=short
  ```
  Expected: all pass.

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```
  Expected: 849 passed, 3 skipped (same as baseline).

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/channels/feishu_content_extractor.py babybot/channels/feishu.py
  git commit -m "refactor: extract feishu content-extraction helpers to feishu_content_extractor.py"
  ```

---

## Task 2: feishu.py → feishu_card_builder.py

Extract the markdown-to-card conversion logic from `FeishuChannel`. These methods only use `self` to access class-level constants and call each other — they can become module-level functions.

**Files:**
- Create: `babybot/channels/feishu_card_builder.py`
- Modify: `babybot/channels/feishu.py`

Functions/methods to extract (convert from instance methods to module functions):
- `_detect_msg_format(text: str) -> str`
- `_normalize_markdown_images(text: str) -> str`
- `_markdown_to_post(text: str) -> dict`
- `_parse_md_table(lines: list[str]) -> list | None`
- `_build_card_elements(text: str) -> list`
- `_split_elements_by_table_limit(elements: list) -> list[list]`
- `_split_headings(elements: list) -> list[list]`
- `_build_single_stream_card(text: str) -> dict`

Constants to move with them: `_MAX_CARD_TEXT_LEN`, `_MAX_TABLE_COLS`, `_MAX_TABLE_ROWS`, and any regex patterns used only by these functions.

- [ ] **Step 1: Read the source lines**

  Read `feishu.py` lines 803–1036 to identify:
  - Exact method signatures
  - All class-level constants referenced
  - Any calls to `self._method()` that need to become plain function calls

- [ ] **Step 2: Create `feishu_card_builder.py`**

  ```python
  """Feishu outbound card and markdown formatting helpers."""

  from __future__ import annotations

  import re
  from typing import Any

  # ── move constants here ──
  _MAX_CARD_TEXT_LEN = ...
  ...

  # ── functions (signatures identical to original methods, minus `self`) ──
  def _detect_msg_format(text: str) -> str: ...
  def _normalize_markdown_images(text: str) -> str: ...
  # etc.
  ```

  Any `self._method(...)` call inside the extracted functions becomes a direct function call to the sibling function.

- [ ] **Step 3: Update `FeishuChannel` methods in `feishu.py`**

  Replace each method body with a one-liner delegation:

  ```python
  from .feishu_card_builder import (
      _build_card_elements,
      _build_single_stream_card,
      _detect_msg_format,
      _markdown_to_post,
      _normalize_markdown_images,
      _parse_md_table,
      _split_elements_by_table_limit,
      _split_headings,
  )

  class FeishuChannel(BaseChannel):
      ...
      @staticmethod
      def _normalize_markdown_images(text: str) -> str:
          return _normalize_markdown_images(text)   # delegate to module fn

      def _detect_msg_format(self, text: str) -> str:
          return _detect_msg_format(text)
      # etc.
  ```

  Keep the class methods as thin wrappers so that `FeishuChannel._normalize_markdown_images` still works (test accesses it as a class method).

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_feishu_format.py tests/test_feishu_streaming.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/channels/feishu_card_builder.py babybot/channels/feishu.py
  git commit -m "refactor: extract feishu card/markdown builder to feishu_card_builder.py"
  ```

---

## Task 3: executor.py → executor_history.py

Extract 5 module-level free functions that build the LLM conversation history from the tape. These have no dependency on `SingleAgentExecutor` state.

**Files:**
- Create: `babybot/agent_kernel/executor_history.py`
- Modify: `babybot/agent_kernel/executor.py`

Functions to move:
- `_estimate_token_count(text) -> int` (line 38)
- `_extract_keywords(text) -> list[str]` (line 43)
- `_build_context_view_messages(memory_store, chat_id, query) -> list[ModelMessage]` (line 56)
- `_history_entry_text(entry) -> str` (line ~1665, ~50 lines)
- `_build_history_messages(...)` (line ~1717, ~200 lines) — the largest single function

- [ ] **Step 1: Read source lines**

  Read `executor.py` lines 38–72 and lines 1663–1916 to get exact signatures and all imports these functions need.

- [ ] **Step 2: Create `executor_history.py`**

  ```python
  """History-building helpers for SingleAgentExecutor.

  Extracted from executor.py to keep the executor module focused on the
  execution loop. All symbols re-exported from executor.py for backward compat.
  """

  from __future__ import annotations

  # imports needed by _build_history_messages (BM25 recall, tape types, etc.)
  from typing import TYPE_CHECKING, Any
  ...

  def _estimate_token_count(text: str) -> int: ...
  def _extract_keywords(text: str) -> list[str]: ...
  def _build_context_view_messages(...) -> list[ModelMessage]: ...
  def _history_entry_text(entry: Any) -> str: ...
  def _build_history_messages(...) -> list[ModelMessage]: ...
  ```

  `_build_history_messages` uses BM25 and tape types — carry over all necessary imports verbatim.

- [ ] **Step 3: Update `executor.py`**

  Replace the 5 function definitions with imports at the top of the file:

  ```python
  from .executor_history import (
      _build_context_view_messages,
      _build_history_messages,
      _estimate_token_count,
      _extract_keywords,
      _history_entry_text,
  )
  ```

  Remove the now-redundant function bodies.

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_agent_kernel_executor.py tests/test_worker.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/agent_kernel/executor_history.py babybot/agent_kernel/executor.py
  git commit -m "refactor: extract executor history-building helpers to executor_history.py"
  ```

---

## Task 4: orchestration_policy_store.py → policy_store_utils.py

Extract 5 tiny utility methods that are pure functions (no DB access, no `self` state needed). Making them module-level functions also allows the analytics and recommender code to use them without going through `self`.

**Files:**
- Create: `babybot/policy_store_utils.py`
- Modify: `babybot/orchestration_policy_store.py`

Methods to convert to free functions:
- `_utc_now() -> str` (reads `datetime.now(timezone.utc)`)
- `_decay_weight(created_at_iso, half_life_days) -> float`
- `_parse_time(ts) -> datetime | None`
- `_is_recent(ts, window_days) -> bool`
- `_decode_json_object(s) -> dict`

The class constants `_OUTCOME_HALF_LIFE_DAYS`, `_FEEDBACK_HALF_LIFE_DAYS`, etc. stay on the class — the free functions take explicit `half_life_days` arguments.

- [ ] **Step 1: Read source lines**

  Read `orchestration_policy_store.py` lines 1154–1197.

- [ ] **Step 2: Create `policy_store_utils.py`**

  ```python
  """Pure utility functions shared by OrchestrationPolicyStore and its helpers."""

  from __future__ import annotations

  import json
  from datetime import datetime, timezone
  from typing import Any


  def utc_now() -> str:
      """Return current UTC time as ISO string."""
      return datetime.now(timezone.utc).isoformat()


  def decay_weight(created_at_iso: str, half_life_days: float) -> float:
      """Exponential decay weight for a record given its creation time."""
      ...

  def parse_time(ts: str | None) -> datetime | None: ...
  def is_recent(ts: str | None, window_days: float) -> bool: ...
  def decode_json_object(s: str | None) -> dict[str, Any]: ...
  ```

  Note: rename to public names (drop leading `_`) since these are now a proper module — but keep private aliases in the store for compat:

  ```python
  # backward-compat aliases used inside the class methods
  _utc_now = utc_now
  _decay_weight = decay_weight
  _parse_time = parse_time
  _is_recent = is_recent
  _decode_json_object = decode_json_object
  ```

- [ ] **Step 3: Update `orchestration_policy_store.py`**

  Add import at top:

  ```python
  from .policy_store_utils import (
      decay_weight as _decay_weight,
      decode_json_object as _decode_json_object,
      is_recent as _is_recent,
      parse_time as _parse_time,
      utc_now as _utc_now,
  )
  ```

  Remove the 5 method definitions from `OrchestrationPolicyStore` (since they're now module-level functions imported as `_` aliases, all internal call sites work unchanged).

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_orchestration_policy_store.py tests/test_orchestration_policy_capture.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/policy_store_utils.py babybot/orchestration_policy_store.py
  git commit -m "refactor: extract policy store pure utilities to policy_store_utils.py"
  ```

---

## Task 5: resource.py → resource_path_utils.py

Extract 3 free functions that perform path detection and artifact collection. These are already module-level functions in `resource.py`; they just need a new home.

**Files:**
- Create: `babybot/resource_path_utils.py`
- Modify: `babybot/resource.py`

Functions to move:
- `_looks_like_path_candidate(value: Any) -> bool` (lines 82–117, ~36 lines)
- `_normalize_artifact_path_for_manager(path_str, output_dir) -> str` (lines 120–152, ~33 lines)
- `_collect_artifact_paths(value, output_dir) -> list[str]` (lines 155–215, ~61 lines)

- [ ] **Step 1: Read source lines**

  Read `resource.py` lines 82–215.

- [ ] **Step 2: Create `resource_path_utils.py`**

  ```python
  """Path-detection and artifact-collection helpers.

  Used by ResourceManager and CallableTool to identify and normalise file
  paths embedded in tool return values.
  """

  from __future__ import annotations

  import shutil
  from pathlib import Path
  from typing import Any


  def _looks_like_path_candidate(value: Any) -> bool: ...
  def _normalize_artifact_path_for_manager(path_str: str, output_dir: Path | None) -> str: ...
  def _collect_artifact_paths(value: Any, output_dir: Path | None) -> list[str]: ...
  ```

- [ ] **Step 3: Update `resource.py`**

  Replace function bodies with import:

  ```python
  from .resource_path_utils import (
      _collect_artifact_paths,
      _looks_like_path_candidate,
      _normalize_artifact_path_for_manager,
  )
  ```

  The rest of `resource.py` calls these functions by name, so nothing else changes.

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_runtime_refactor_resource_manager.py tests/test_resource_module_exports.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/resource_path_utils.py babybot/resource.py
  git commit -m "refactor: extract resource path utilities to resource_path_utils.py"
  ```

---

## Task 6: resource.py → resource_callable_tool.py

Extract `CallableTool` (the class that wraps Python callables into the tool protocol) plus its associated context-manager helpers. These have a defined interface; moving them keeps `resource.py` focused on the `ResourceManager` facade.

**Files:**
- Create: `babybot/resource_callable_tool.py`
- Modify: `babybot/resource.py`

Symbols to move:
- `CallableTool` class (lines ~218–335, ~120 lines)
- `override_current_write_root` context manager (lines ~338–345)
- `get_current_write_root` function (lines ~348–355)
- The two `contextvars.ContextVar` module-level vars that back them

Note: `test_resource_module_exports.py` imports `CallableTool` from `babybot.resource`, so it must be re-exported.

- [ ] **Step 1: Read source lines**

  Read `resource.py` lines 218–360 to capture exact code and the two ContextVar declarations.

- [ ] **Step 2: Create `resource_callable_tool.py`**

  ```python
  """CallableTool — wraps Python callables into the kernel Tool protocol."""

  from __future__ import annotations

  import contextvars
  import inspect
  from contextlib import contextmanager
  from pathlib import Path
  from typing import Any, TYPE_CHECKING

  from .resource_path_utils import _collect_artifact_paths, _normalize_artifact_path_for_manager

  if TYPE_CHECKING:
      from .resource import ResourceManager

  _CURRENT_CALLABLE_TOOL_WRITE_ROOT: contextvars.ContextVar[Path | None] = ...
  _CURRENT_DEFAULT_WRITE_ROOT: contextvars.ContextVar[Path | None] = ...


  class CallableTool:
      ...

  @contextmanager
  def override_current_write_root(path: Path): ...

  def get_current_write_root() -> Path | None: ...
  ```

  `CallableTool.__init__` takes an optional `manager: ResourceManager | None` — use `TYPE_CHECKING` import to avoid circular dependency.

- [ ] **Step 3: Update `resource.py`**

  Replace the `CallableTool` class and the two functions with:

  ```python
  from .resource_callable_tool import (
      CallableTool,
      get_current_write_root,
      override_current_write_root,
  )
  ```

  `resource.py` must still export `CallableTool` at module level — the import statement above achieves that.

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_resource_module_exports.py tests/test_resource_skills.py tests/test_runtime_refactor_resource_manager.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/resource_callable_tool.py babybot/resource.py
  git commit -m "refactor: extract CallableTool and write-root helpers to resource_callable_tool.py"
  ```

---

## Task 7: dynamic_orchestrator.py → dynamic_orchestrator_tools.py

The orchestration tool schemas (`_ORCHESTRATION_TOOLS` tuple, ~185 lines of JSON-like dicts) and the backward-compat `_ORCHESTRATION_TOOL_BY_NAME` dict are pure data — they never change at runtime and have no dependencies on the `DynamicOrchestrator` class.

**Files:**
- Create: `babybot/agent_kernel/dynamic_orchestrator_tools.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`

Symbols to move:
- `_ORCHESTRATION_TOOLS` tuple (lines 78–263)
- `_ORCHESTRATION_TOOL_BY_NAME` dict (lines 265–267)

- [ ] **Step 1: Read source lines**

  Read `dynamic_orchestrator.py` lines 78–270.

- [ ] **Step 2: Create `dynamic_orchestrator_tools.py`**

  ```python
  """Static tool schemas for the DynamicOrchestrator.

  These definitions describe the 5 orchestration tools (dispatch_task,
  wait_for_tasks, get_task_result, reply_to_user, dispatch_team) that the
  orchestrator model may call.
  """

  from __future__ import annotations

  from typing import Any

  _ORCHESTRATION_TOOLS: tuple[dict[str, Any], ...] = (
      # ... verbatim copy of the tuple ...
  )

  _ORCHESTRATION_TOOL_BY_NAME: dict[str, dict[str, Any]] = {
      t["function"]["name"]: t for t in _ORCHESTRATION_TOOLS
  }
  ```

- [ ] **Step 3: Update `dynamic_orchestrator.py`**

  Replace the definitions (lines 78–267) with:

  ```python
  from .dynamic_orchestrator_tools import (
      _ORCHESTRATION_TOOL_BY_NAME,
      _ORCHESTRATION_TOOLS,
  )
  ```

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_dynamic_orchestrator.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/agent_kernel/dynamic_orchestrator_tools.py babybot/agent_kernel/dynamic_orchestrator.py
  git commit -m "refactor: extract orchestration tool schemas to dynamic_orchestrator_tools.py"
  ```

---

## Task 8: dynamic_orchestrator.py → dynamic_orchestrator_prompt.py

Extract the free functions involved in building the orchestrator system prompt and resource catalog. These are all module-level functions with clear inputs and outputs — no `DynamicOrchestrator` state needed.

**Files:**
- Create: `babybot/agent_kernel/dynamic_orchestrator_prompt.py`
- Modify: `babybot/agent_kernel/dynamic_orchestrator.py`

Functions to move:
- `_build_resource_catalog(resource_briefs) -> str` (lines ~293–352)
- `_needs_deferred_task_guidance(goal) -> bool` (lines ~355–362)
- `_normalize_recommended_resource_ids(recs) -> list[str]` (lines ~365–376)
- `_provider_policy_hints(ctx, provider) -> tuple[str, str]` (lines ~379–431)
- `_goal_has_explicit_parallel_intent(goal) -> bool` (lines ~434–441)
- `_is_maintenance_goal(goal) -> bool` (lines ~444–454)
- `_emit_policy_decision(ctx, ...)` (lines ~457–469)
- String constants: default fallback prompts, `_FORCE_CONVERGE_TOOL_NAMES`

Also move `_EMPTY_TOKEN_USAGE` and `_DEFAULT_FALLBACK_REPLY` string constants if present.

- [ ] **Step 1: Read source lines**

  Read `dynamic_orchestrator.py` lines 53–470 (after the imports).

- [ ] **Step 2: Create `dynamic_orchestrator_prompt.py`**

  ```python
  """Prompt-assembly and policy-hint helpers for DynamicOrchestrator."""

  from __future__ import annotations

  from typing import TYPE_CHECKING, Any

  if TYPE_CHECKING:
      from ..resource import ResourceManager

  # ── verbatim copies of all extracted free functions ──
  def _build_resource_catalog(...) -> str: ...
  def _needs_deferred_task_guidance(goal: str) -> bool: ...
  # etc.
  ```

- [ ] **Step 3: Update `dynamic_orchestrator.py`**

  Replace the function definitions with:

  ```python
  from .dynamic_orchestrator_prompt import (
      _build_resource_catalog,
      _emit_policy_decision,
      _goal_has_explicit_parallel_intent,
      _is_maintenance_goal,
      _needs_deferred_task_guidance,
      _normalize_recommended_resource_ids,
      _provider_policy_hints,
  )
  ```

- [ ] **Step 4: Run targeted tests**

  ```bash
  pytest tests/test_dynamic_orchestrator.py tests/test_orchestrator_routing.py -q --tb=short
  ```

- [ ] **Step 5: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add babybot/agent_kernel/dynamic_orchestrator_prompt.py babybot/agent_kernel/dynamic_orchestrator.py
  git commit -m "refactor: extract orchestrator prompt-assembly helpers to dynamic_orchestrator_prompt.py"
  ```

---

## Task 9: orchestrator.py → orchestrator_routing_cascade.py

Extract the routing cascade methods from `OrchestratorAgent`. These form a self-contained subdomain (~400 lines): they read from `self` but only access policy engine, reflection store, and intent cache — all injectable.

**Strategy:** Extract as a standalone class `RoutingCascade` that receives the necessary service objects via `__init__`, or as module-level functions that take explicit parameters. The module-function approach is cleaner given the existing pattern in the codebase.

**Files:**
- Create: `babybot/orchestrator_routing_cascade.py`
- Modify: `babybot/orchestrator.py`

Methods to extract (become free functions or a `RoutingCascade` class):
- `_resolve_routing(self, ...)` — main cascade (~250 lines)
- `_routing_decision_from_reflection(self, payload) -> RoutingDecision`
- `_routing_decision_from_intent_cache(self, payload) -> RoutingDecision`
- `_maybe_override_policy_from_reflection(self, policy, reflection)`
- `_maybe_soften_policy_from_guardrail(self, policy, guardrail)`
- `_routing_policy_hints(self, decision) -> list[str]`

These methods access `self._policy_engine`, `self._policy_store`, and `self._model_gateway` — these become constructor parameters for `RoutingCascade`.

- [ ] **Step 1: Read source lines**

  Read `orchestrator.py` lines 687–938 to get exact method signatures and all `self.*` references.

- [ ] **Step 2: Identify all `self.*` dependencies**

  For each extracted method, list every attribute accessed through `self`. These become constructor parameters of `RoutingCascade`:
  - `self._policy_engine`
  - `self._policy_store`
  - `self._model_gateway`
  - `self._logger` (or use module-level logger)
  - Any config fields

- [ ] **Step 3: Create `orchestrator_routing_cascade.py`**

  ```python
  """Routing cascade for OrchestratorAgent.

  Extracted from orchestrator.py to separate the multi-stage routing
  decision logic from the orchestrator's main process_task flow.
  """

  from __future__ import annotations

  import logging
  from typing import TYPE_CHECKING, Any

  if TYPE_CHECKING:
      from .model_gateway import OpenAICompatibleGateway
      from .orchestration_policy_store import OrchestrationPolicyStore
      from .orchestrator_policy_engine import PolicyEngine

  logger = logging.getLogger(__name__)


  class RoutingCascade:
      def __init__(
          self,
          policy_engine: PolicyEngine,
          policy_store: OrchestrationPolicyStore,
          model_gateway: OpenAICompatibleGateway,
      ) -> None:
          self._policy_engine = policy_engine
          self._policy_store = policy_store
          self._model_gateway = model_gateway

      async def resolve_routing(self, ...) -> RoutingResult: ...
      def routing_policy_hints(self, decision: RoutingDecision) -> list[str]: ...
      # etc.
  ```

- [ ] **Step 4: Update `OrchestratorAgent` in `orchestrator.py`**

  - In `__init__`, create `self._routing_cascade = RoutingCascade(self._policy_engine, self._policy_store, self._model_gateway)`
  - Replace each extracted method with a one-liner delegation:
    ```python
    async def _resolve_routing(self, *args, **kwargs):
        return await self._routing_cascade.resolve_routing(*args, **kwargs)
    ```
  - Add import at top of file:
    ```python
    from .orchestrator_routing_cascade import RoutingCascade
    ```

- [ ] **Step 5: Run targeted tests**

  ```bash
  pytest tests/test_orchestrator_routing.py tests/test_orchestrator_interactive_sessions.py -q --tb=short
  ```

- [ ] **Step 6: Full suite check**

  ```bash
  pytest tests/ -q --tb=short
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add babybot/orchestrator_routing_cascade.py babybot/orchestrator.py
  git commit -m "refactor: extract routing cascade to orchestrator_routing_cascade.py"
  ```

---

## Verification

After all tasks, run the full suite one final time and check line counts:

```bash
pytest tests/ -q --tb=short
wc -l babybot/orchestrator.py babybot/agent_kernel/executor.py babybot/resource.py \
       babybot/agent_kernel/dynamic_orchestrator.py babybot/channels/feishu.py \
       babybot/orchestration_policy_store.py
```

Expected outcome: every file above is below 1000 lines, all tests pass.

---

## Expected Line Count Reduction

| File | Before | Target | Extracted to |
|---|---|---|---|
| `orchestrator.py` | ~1993 | ~1550 | `orchestrator_routing_cascade.py` (~400 lines) |
| `executor.py` | ~1915 | ~1660 | `executor_history.py` (~255 lines) |
| `resource.py` | ~1680 | ~1420 | `resource_path_utils.py` (~140) + `resource_callable_tool.py` (~120) |
| `dynamic_orchestrator.py` | ~1537 | ~1100 | `dynamic_orchestrator_tools.py` (~200) + `dynamic_orchestrator_prompt.py` (~240) |
| `channels/feishu.py` | ~1268 | ~820 | `feishu_content_extractor.py` (~200) + `feishu_card_builder.py` (~250) |
| `orchestration_policy_store.py` | ~1197 | ~1140 | `policy_store_utils.py` (~60) |

The high-ROI tasks are 1–3 and 7–8 (pure free-function extractions, near-zero risk).
Tasks 6 and 9 require more care (class extraction) but produce the biggest reductions.
