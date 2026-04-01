# BabyBot Code Review — 2026-04-01

**Reviewer:** AI code review (claude-opus-4-6)  
**Scope:** Full codebase — all source files under `babybot/`, `skills/`, `tests/`, `docs/`  
**Test baseline:** 603 passed, 3 skipped, 0 failed  
**Review dimensions:** Architecture & Design · Correctness & Bugs · Security · Robustness & Reliability · Code Quality · Performance

---

## Executive Summary

BabyBot is a well-structured, thoughtfully designed multi-channel conversation-agent framework. The architecture is layered cleanly, contracts between subsystems are documented, and the test suite is large and substantive. The codebase shows strong recent investment in hardening worker boundaries, adding a routing-learning subsystem, and formalising the interactive-session model.

The most significant risks are concentrated in a small number of places:

1. **Concurrency correctness** in the Feishu channel and orchestrator — multiple `asyncio.Event`/`Lock` objects are shared across sync/async thread boundaries in ways that can silently corrupt state.
2. **Shell-injection residual surface** — the workspace-tools security filter is solid but its bypass paths (multiline commands, Python `exec` wrapping) have not been closed.
3. **SQLite write-path** — four separate SQLite databases are opened without WAL mode or connection-pool discipline; under concurrent task load this can produce `database is locked` errors.
4. **Three mega-files** (`orchestrator.py` 2300 lines, `resource.py` 1947 lines, `dynamic_orchestrator.py` 1905 lines) that are already hurting comprehensibility and will become maintenance liabilities.
5. **Missing test scenarios** — the interactive-session backend, Weixin channel, and cron scheduler all lack failure-path coverage.

The findings below are ranked **High / Medium / Low** within each dimension.

---

## 1. Architecture & Design

### 1.1 [High] `orchestrator.py` is doing too much — 2300 lines, 7+ distinct responsibilities

`OrchestratorAgent` handles: message routing, interactive-session lifecycle, policy selection, tape management, memory bootstrapping, job persistence, cron-task creation, feedback commands, and channel-tool context. This violates SRP and makes it very hard to reason about what happens during a single `process_task()` call.

**What it costs today:** The `process_task()` method alone is ~350 lines with at least five early-return paths. Finding where the policy system interacts with the tape, or where job persistence is triggered, requires reading the whole method.

**Recommendation:** Extract the following into separate collaborators:
- `InteractiveSessionRouter` — already partially exists in `interactive_sessions/`; `process_task()` should only call `router.route(task)`.
- `FeedbackCommandHandler` — the `@policy`, `@job`, `@skill`, `@session` command dispatching.
- `TapeLifecycleManager` — Tape construction, anchor eviction, and memory bootstrap.

A pragmatic first step: extract the feedback-command dispatch block (~150 lines) into a standalone function with its own tests.

---

### 1.2 [High] `resource.py` has the same problem — 1947 lines, four distinct subsystems

`ResourceManager` combines:
- Skill discovery and YAML parsing (`_parse_frontmatter`, `_register_skill_tools`)
- Tool schema reflection (`_json_schema_for_callable`, `_schema_for_ast_annotation`)
- Worker runtime (`run_subagent_task`, `_build_worker_sys_prompt`)
- Resource catalog for the orchestration layer (`get_resource_briefs`, `resolve_resource_scope`)

The skill-loading AST path and the worker-runtime path have no logical relationship; they happen to share a class because both need access to `self.registry`.

**Recommendation:** The AST/schema machinery (`resource_skill_loader.py` and `resource_tool_loader.py` already started this split) should be fully decoupled. `ResourceManager` should become a thin facade over `SkillLoader`, `WorkerRuntime`, and `ResourceCatalog`.

---

### 1.3 [Medium] `DynamicOrchestrator` mixes DAG scheduling policy with child-task I/O — 1905 lines

The DAG execution logic (`InProcessChildTaskRuntime`, dead-letter tracking, dependency resolution, scheduler-boundary guard) is mixed into the same class that builds LLM prompts and interprets tool calls. This makes it difficult to swap the scheduler (e.g., replace `InProcessChildTaskRuntime` with a distributed backend) without touching prompt logic.

The existing `InProcessChildTaskRuntime` is a good abstraction start. Continuing to extract the scheduler policy behind that interface would pay dividends.

---

### 1.4 [Medium] `ContextRouter` from the routing-learning design is not yet wired into `process_task()`

`task_evaluator.py` and `orchestration_router.py` exist but `OrchestratorAgent.process_task()` does not call `ContextRouter`. The `ReflectionStore` is not populated. The spec (`2026-03-30-lightweight-routing-learning-design.md`) describes a three-part V1 loop (router → evaluator → reflection store), but only `TaskEvaluator` is partially live. This is a known TODO but worth tracking as a design debt item.

---

### 1.5 [Low] `ConservativePolicySelector` and `ContextRouter` overlap in purpose

`ConservativePolicySelector` already selects decomposition / scheduling / worker actions. `ContextRouter` adds a `route_mode` and `execution_style` that cover much of the same decision space. The spec acknowledges this (`PolicySelector` becomes a "second-layer selector") but the integration boundary is not yet defined in code. When both systems are active simultaneously it will be unclear which one wins.

**Recommendation:** Define a clear precedence contract: `ContextRouter` output sets hard constraints; `ConservativePolicySelector` fills the remaining degrees of freedom.

---

## 2. Correctness & Bugs

### 2.1 [High] `FeishuChannel._processed` is a plain `set` accessed from both a WebSocket thread and the asyncio event loop

In `feishu.py`, the deduplication set `self._processed` is written inside the WebSocket callback thread and read in the asyncio handler path. There is a `asyncio.Lock` called `_processed_lock` but it is only acquired in the `async def _handle_event()` path — not in the sync thread that also writes to `_processed`. This is a data race.

```python
# feishu.py — sync callback, no lock
self._processed.add(event_id)

# feishu.py — async handler, acquires lock
async with self._processed_lock:
    if event_id in self._processed:
        ...
```

Under load, duplicate events will slip through the deduplication, causing duplicate LLM calls for the same user message.

**Fix:** Either (a) move deduplication entirely to the async path with the lock, or (b) protect the sync write with a `threading.Lock` and ensure both paths acquire their respective lock before touching `_processed`.

---

### 2.2 [High] `CronScheduler` fires coroutines with `asyncio.run_coroutine_threadsafe` but the `loop` reference can be `None`

In `cron.py`, `_trigger_task()` calls:
```python
asyncio.run_coroutine_threadsafe(self._callback(task), self._loop)
```
`self._loop` is set at construction time to `asyncio.get_event_loop()`. If the `CronScheduler` is constructed before the event loop is running (e.g., during module import or in a test without a running loop), `self._loop` will be the default loop, which may not be the loop where the orchestrator is running. In tests this produces `RuntimeError: no current event loop` or silently drops tasks.

**Fix:** Capture the loop lazily at first `start()` call using `asyncio.get_running_loop()`.

---

### 2.3 [High] `InteractiveSessionManager.get_or_create_session` has a TOCTOU race

```python
# manager.py
if chat_key in self._sessions:
    return self._sessions[chat_key], False
session = await backend.start(...)
self._sessions[chat_key] = session
return session, True
```

There is no lock around the check-and-insert. Two concurrent messages for the same `chat_key` can both pass the `if chat_key in self._sessions` check and both call `backend.start()`, creating two Claude processes for the same chat.

The class has a `self._lock: asyncio.Lock` attribute that is acquired in `stop_session()` but **not** in `get_or_create_session()`.

**Fix:** Wrap the entire check-and-create path with `async with self._lock`.

---

### 2.4 [Medium] `context.py` — anchor compression fires a new LLM call without cancellation guard

In `Tape.maybe_compress()`, when the token count exceeds the threshold, it awaits an LLM call to produce an anchor summary. If the task that triggered compression is cancelled mid-way (e.g., the user sends a new message), the anchor summary write will complete after the tape has already been handed to the next task, potentially writing a stale summary.

No `asyncio.shield()` or equivalent protection exists around the compression write.

---

### 2.5 [Medium] `runtime_job_store.py` — `cleanup_old_jobs()` uses `DELETE WHERE created_at < ?` with a local timestamp

The cutoff is computed with `datetime.utcnow()` (deprecated in Python 3.12+). More importantly, `created_at` is stored as an ISO string without timezone info, so jobs created during DST transitions or on hosts with non-UTC local time can be incorrectly retained or prematurely deleted.

**Fix:** Store timestamps as `datetime.now(tz=timezone.utc).isoformat()` and compare with UTC-aware datetimes.

---

### 2.6 [Medium] `orchestration_policy_store.py` — `record_outcome()` does not validate that the `decision_id` exists before inserting the outcome row

If the caller passes a stale or fabricated `decision_id`, a dangling outcome row is inserted with a broken foreign-key relationship. SQLite does not enforce foreign keys by default; the `PRAGMA foreign_keys = ON` is not set in `sqlite_utils.py`.

**Fix:** Either enable `PRAGMA foreign_keys = ON` in `_connect()` or add an existence check before insert.

---

### 2.7 [Low] `weixin.py` — QR login polling loop has no maximum retry count

`_poll_qr_login()` loops until success with a fixed 3-second sleep and no outer deadline. If the WeChat server returns a permanent error code (not just "waiting"), the loop will run indefinitely.

---

### 2.8 [Low] `dynamic_orchestrator.py` — `_normalize_child_task_description()` truncates upstream results at a hard 800-character limit

If upstream task output exceeds 800 characters it is silently truncated. For tasks where the downstream step genuinely needs the full context (e.g., passing a structured JSON response), this silently loses data. The truncation should log a warning and ideally the limit should be configurable.

---

## 3. Security

### 3.1 [High] `resource_workspace_tools.py` — the shell-command security filter is bypassable via heredoc and multiline strings

The `_check_shell_command()` regex filter blocks common injection patterns (`rm -rf`, backticks, `$(...)`, etc.) but operates on the **full command string** as a single unit. A multiline command passed as:

```bash
echo foo
rm -rf /
```

passes the check on each line in isolation if the validator only scans the first token or uses `re.match` instead of `re.search` on the full string. A quick test:

```python
cmd = "echo hello\nrm -rf /tmp/important"
# _BLOCKED_SHELL_PATTERNS checks the whole string — verify this is actually done
```

The implementation should be audited to confirm the regex is applied to the complete command including embedded newlines.

---

### 3.2 [High] `resource_workspace_tools.py` — Python runner does not restrict `__import__` or `exec`

`resource_python_runner.py` executes arbitrary Python in the same process using `exec(code, globals_dict)`. The sandbox restricts builtins (`open`, `os`, `subprocess` are removed from the globals), but `__import__` is not removed. An agent can call `__import__('subprocess').run(...)` to escape the sandbox entirely.

**Fix:** Add `__import__` to the blocked-builtins list, or move Python execution to a subprocess with `--no-site-packages` and timeout, similar to how CLI scripts are handled.

---

### 3.3 [Medium] `weixin.py` — AES-ECB decryption is used for media content

`_decrypt_media()` uses AES-ECB mode. ECB is deterministic and pattern-revealing; identical plaintext blocks produce identical ciphertext. For media content this is primarily a confidentiality concern rather than an integrity one, but it signals that the WeChat API layer may have outdated encryption assumptions worth revisiting when the integration is upgraded.

---

### 3.4 [Medium] `config.py` — API keys are read from environment but also accepted in `config.json`

`config.json.example` shows `model.api_key` as a plain JSON field. If `config.json` is accidentally committed to version control (it is in `.gitignore` but users sometimes force-add it), API keys will be leaked. Consider adding a pre-commit hook or startup warning if `api_key` is set directly in the config file rather than via `OPENAI_API_KEY`.

---

### 3.5 [Low] `resource_skill_loader.py` — AST-based tool discovery executes `ast.parse()` on untrusted user-created skill scripts

The code only uses `ast.parse()` (safe), not `eval()` or `exec()`. This is correct. However, the skill scripts themselves are later executed via `_load_tool_module()` using `importlib`. If a malicious `SKILL.md` is placed in the workspace skills directory by an attacker with file-write access, it can execute arbitrary code at skill-load time. This is an acceptable trust boundary assumption for the current deployment model but should be documented.

---

## 4. Robustness & Reliability

### 4.1 [High] SQLite databases are opened without WAL mode — concurrent writes will produce `database is locked`

Four databases are opened: `policy_store`, `memory_store`, `job_store`, and the context/tape SQLite. None of them set `PRAGMA journal_mode=WAL`. Under concurrent async tasks (which is the common case — multiple workers writing job status and memory observations simultaneously), SQLite will serialize writes with the default journal mode, and any write that arrives while another transaction is open will raise `sqlite3.OperationalError: database is locked`.

`sqlite_utils.py` has the connection helper but does not set WAL:

```python
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # WAL mode not set
    return conn
```

**Fix:** Add `conn.execute("PRAGMA journal_mode=WAL")` immediately after opening each connection.

---

### 4.2 [High] `OrchestratorAgent` has no circuit breaker for runaway task loops

If a model enters a tool-call loop that never reaches `reply_to_user` and never triggers `LoopGuard`, the orchestrator will keep calling the gateway indefinitely. `LoopGuard` detects repeated identical tool calls but does not bound total step count at the orchestrator level. `DynamicOrchestrator` has `max_steps` but `SingleAgentExecutor` (`executor.py`) relies on `policy.max_steps` which defaults to a configurable value — if misconfigured to 0 or a very large number, there is no safety net.

**Fix:** Add a hard upper-bound (e.g., 200 steps) in `SingleAgentExecutor` that cannot be overridden by config, separate from the soft `max_steps` policy.

---

### 4.3 [Medium] `ClaudeInteractiveBackend` process cleanup relies on `finally` in `_run_read_loop()` but the task can be GC'd before cleanup

In `backends/claude.py`, the reader task is created with `asyncio.create_task()`. If the session is abandoned (e.g., the manager is garbage-collected without calling `stop()`), the reader task will be cancelled by the GC, but the Claude subprocess will not be terminated because `_cleanup()` requires the session object to be alive.

`InteractiveSessionManager` has a hard-limit age check at `send()` time, but if `send()` is never called again, orphaned subprocesses will run until the parent process exits.

**Fix:** Register a `__del__` finalizer or use `weakref.finalize()` to ensure subprocess termination even if `stop()` is never called.

---

### 4.4 [Medium] `ScheduledTaskManager` does not persist the `last_run` timestamp atomically with task execution

In `cron.py`, the flow is:
1. Fire the callback.
2. On success, update `last_run` in the JSON file.

If the process crashes between step 1 and step 2, the task will re-fire on the next startup even if it already ran. For one-shot tasks this causes duplicate execution. For interval tasks it causes a slightly early re-run, which may be acceptable, but the current one-shot completion marking (`status: "completed"`) could fire twice.

**Fix:** Write the `last_run` / `status` update before firing the callback for one-shot tasks (mark as in-progress atomically), or use a SQLite-backed store with a transaction.

---

### 4.5 [Medium] `MessageBus` does not limit queue depth

`MessageBus` uses `asyncio.Queue()` (unbounded). Under sustained inbound load (e.g., a Feishu group chat with many users), the queue will grow without bound, consuming memory until the process is OOM-killed with no user-visible error.

**Fix:** Set `maxsize` (e.g., 500) and implement backpressure — reject or drop the oldest message with a logged warning.

---

### 4.6 [Low] `HybridMemoryStore.observe_user_message()` writes to SQLite on every message

Every user message triggers a synchronous SQLite write in the hot path (`observe_user_message` is called from `process_task()` before any LLM call). Under high message frequency this serialises all incoming messages through a single SQLite connection. Consider batching or deferring non-critical memory observations to a background task.

---

### 4.7 [Low] `feishu.py` — `_upload_media_sync()` reads entire file into memory before uploading

For large audio or video files generated by skills, the entire file is `read()` into a `BytesIO` before being passed to the Lark SDK. For multi-MB files this blocks the thread and consumes significant memory. The Lark SDK supports streaming uploads; this should be used instead.

---

## 5. Code Quality

### 5.1 [Medium] `_json_schema_for_callable()` and `_schema_for_ast_annotation()` are duplicated across `resource.py` and `resource_tool_loader.py`

Both modules contain schema-inference logic. `resource_tool_loader.py` was introduced to split the skill-loading concern but the schema helpers were not fully migrated. This creates a risk of the two implementations diverging.

**Fix:** Move the canonical implementation to `resource_tool_loader.py` (or a new `schema_utils.py`) and import it from `resource.py`.

---

### 5.2 [Medium] Magic numbers and hardcoded limits scattered across the codebase

Examples:
- `dynamic_orchestrator.py`: `MAX_STEPS = 30`, `_MAX_TASKS = 50`, `_MAX_PARALLEL = 8`, `_SCHEDULER_RESOURCE_TYPES` — all hardcoded.
- `orchestrator.py`: context token thresholds appear in both `context.py` and `config.py` with different defaults.
- `resource_skill_runtime.py`: `_MAX_SKILL_PACKS = 6` hardcoded.
- `resource.py`: `_UPSTREAM_RESULT_TRUNCATE_CHARS = 800` hardcoded.
- `memory_store.py`: layer thresholds hardcoded.

These should be consolidated into `SystemConfig` (which already has many of them) or at minimum defined as named module-level constants with a comment explaining the rationale.

---

### 5.3 [Medium] `feishu.py` — `_detect_message_format()` is a 120-line function with 8 branches and no tests for the most complex paths

The smart-format detection logic (deciding between `text`, `post`, `interactive`) has significant complexity. The existing `test_feishu_format.py` covers only a few cases (`normalize_markdown_images`, `extract_interactive_content`). The multi-step detection that checks for tables, code blocks, and mixed CJK/code content is completely untested.

---

### 5.4 [Medium] Several async functions use `asyncio.run()` inside themselves as a synchronous escape hatch

Examples in `resource.py`:
```python
# Inside an async method:
result = asyncio.run(self._some_async_helper())
```
This will raise `RuntimeError: This event loop is already running` if called from within an already-running loop (which is the normal case for all production paths). The fact that this does not break today is likely due to `nest_asyncio` or the tests using `asyncio.run()` at the top level where a new loop is created. Any callers that use `await` will hit this.

A grep for `asyncio.run(` inside `async def` functions should be done and all occurrences replaced with direct `await`.

---

### 5.5 [Low] `orchestrator.py` — `_build_task_contract()` reassembles routing context manually each time

The `ContextView`, tape summary, and memory state are re-built inside `_build_task_contract()` on every call. This duplicates logic that already exists in `context_views.py`. Consider making `build_context_view()` the single source of truth and passing its result directly into the task contract builder.

---

### 5.6 [Low] `ResourceManager._parse_frontmatter()` uses a hand-rolled YAML mini-parser instead of `yaml.safe_load()`

The hand-rolled parser handles quoted strings, colons, and YAML lists but will silently misparse multi-line values, anchors, or non-ASCII keys. The project already depends on no YAML library (by design?), but `PyYAML` or `tomllib` (stdlib in Python 3.11+) would be more robust. At minimum the parser should reject inputs it cannot handle rather than silently returning partial results.

---

### 5.7 [Low] Dead code: `resource.py` has a `CallableTool` dataclass that is imported in tests but never constructed in production paths

`CallableTool` is imported in `test_resource_skills.py` but a search through the production code shows no `CallableTool(...)` constructor call. This appears to be a leftover from an earlier refactor.

---

## 6. Performance

### 6.1 [Medium] `get_resource_briefs()` snapshots the entire tool registry on every call

`DynamicOrchestrator._build_initial_messages()` calls `resource_manager.get_resource_briefs()` on every LLM invocation (verified in `test_build_initial_messages_reuses_cached_resource_catalog` — which shows `brief_calls == 2` for two calls, meaning caching is NOT in effect at the `DynamicOrchestrator` level). For an orchestration session with 10+ LLM rounds, this scans the full registry 10+ times.

The test comment says "reuses cached resource catalog" but it actually asserts `brief_calls == 2`, meaning the catalog is rebuilt on every call to `_build_initial_messages`. The caching is of the formatted string, not of the briefs themselves.

**Fix:** Cache the briefs result for the lifetime of the orchestration session (i.e., across `_build_initial_messages` calls within a single `run()` invocation).

---

### 6.2 [Medium] `_select_skill_packs()` is `async` but does no I/O — it tokenises the query and scores skills synchronously inside the event loop

For configurations with hundreds of skills, the tokenisation and scoring loop runs in the event loop thread, blocking all other coroutines. The function is defined `async` likely for future extension, but currently it does not `await` anything.

**Fix:** For large skill counts, offload to `asyncio.to_thread()`.

---

### 6.3 [Low] `Tape.to_dict()` serializes the full message list on every context-view build

`build_context_view()` calls `tape.to_dict()` which serialises all messages including full image paths and metadata. For tapes with many turns this is O(n) on every LLM call. Consider a lazy or incremental serialisation strategy, or add a `__len__`-based fast path that skips serialisation when building views that only need message counts.

---

### 6.4 [Low] `feishu.py` — `_build_post_element()` creates a new Python object per character when splitting text around code-block boundaries

The code splits Markdown into "text segments" and "code block segments" using a character-by-character walk in some paths. For long responses (multi-thousand character code outputs), this is O(n²) in the worst case. A single `re.split()` on `` ``` `` delimiters would be both simpler and O(n).

---

## 7. Test Coverage

### Summary

**603 tests, 3 skipped.** The suite is large and focuses well on unit behaviour. Key observations:

| Subsystem | Coverage quality | Notable gaps |
|---|---|---|
| `DynamicOrchestrator` | Excellent | Scheduler-boundary guard, media propagation, timeout — all covered |
| `ConservativePolicySelector` | Excellent | All edge cases (staleness, drift, recent failure) covered |
| `ResourceManager` / skills | Good | Python fallback, AST parsing, subagent lease stripping covered |
| `FeishuChannel` | Partial | Smart format detection (~120 lines), deduplication race, media upload uncovered |
| `WeixinChannel` | Minimal | Only `test_weixin_channel.py` (3 tests), QR polling, message parsing untested |
| `InteractiveSessionManager` | Good | Lifecycle tested; TOCTOU race not tested |
| `ClaudeInteractiveBackend` | Minimal | Only smoke tests; process cleanup, orphan prevention not tested |
| `CronScheduler` | Partial | Basic scheduling covered; one-shot crash-recovery not tested |
| `HybridMemoryStore` | Good | Core operations covered |
| `OrchestratorAgent` | Partial | Routing, policy feedback covered; anchor compression, job state machine not covered |
| `MessageBus` | Partial | Streaming covered; queue overflow not tested |

### High-priority missing tests

1. **`FeishuChannel._detect_message_format()`** — the most complex branching logic in the codebase without coverage.
2. **`InteractiveSessionManager.get_or_create_session()` concurrent access** — the TOCTOU race described in §2.3.
3. **`CronScheduler` one-shot crash recovery** — the atomicity issue described in §4.4.
4. **`resource_python_runner.py` sandbox escape** — `__import__` bypass described in §3.2.
5. **`MessageBus` backpressure** — queue overflow behaviour.

---

## 8. Type-Checking Errors (Pre-existing, Detected via LSP)

The following type errors exist in the codebase and are surfaced by the language server. They do not cause test failures today (likely due to `# type: ignore` suppression or duck typing at runtime) but represent latent bugs.

### 8.1 [Medium] `orchestrator.py` — LLM response parsed as `str` assigned to `Literal` fields

At lines ~607–645, the router result from an LLM call is a raw `str` that is passed directly to `RoutingDecision(route_mode=..., execution_style=..., parallelism_hint=...)`. These fields are typed as `Literal[...]` but the values are unchecked strings. If the LLM returns an unexpected value, the dataclass is constructed with an invalid `Literal` value and downstream code that pattern-matches on these fields will silently fall through.

**Fix:** Add a validation step after LLM parsing that maps unknown values to a safe default:
```python
route_mode = parsed.get("route_mode", "tool_workflow")
if route_mode not in ("tool_workflow", "answer", "debate"):
    route_mode = "tool_workflow"
```

### 8.2 [Medium] `cli.py` — `process_task()` called with keyword arguments from a `**config_dict` that is typed as `dict[str, str]`

The CLI `process_task()` call at line 211 unpacks a config dictionary where all values are `str`, but `process_task()` expects typed arguments (`heartbeat: Heartbeat | None`, `media_paths: list[str] | None`, etc.). At runtime this works because these values are `None` (which is a string default in the config dict), but it will silently break if a non-None value is ever passed through this path.

**Fix:** Pass keyword arguments explicitly rather than via `**kwargs` expansion, or use a typed `ProcessTaskOptions` dataclass.

### 8.3 [Low] `resource.py` — `ContextVar[None]` returned where `ContextVar[T | None]` is expected

The `_get_current_task_lease_var()` and `_get_current_skill_ids_var()` methods initialise the `ContextVar` with `default=None` but type it as `ContextVar[ToolLease | None]`. The invariance of `ContextVar`'s type parameter means `ContextVar[None]` is not assignable to `ContextVar[ToolLease | None]`. The fix is to use `ContextVar[ToolLease | None]("name", default=None)`.

### 8.4 [Low] `executor.py` — `list[dict]` passed where `tuple[dict]` is expected for tool schemas

At line 190, a list is passed where a tuple is expected. This is harmless at runtime (Python does not enforce this) but should be consistent.

### 8.5 [Low] `message_bus.py` — `dataclasses.asdict()` called on `type[DataclassInstance]` instead of an instance

Lines 424 and 492 call `dataclasses.asdict(obj)` where `obj` has the union type `DataclassInstance | type[DataclassInstance]`. If `obj` is the class itself (not an instance), `asdict()` will raise `TypeError` at runtime.

---

## Prioritised Action List

### Must Fix (blocking reliability or security in production)

| # | Finding | File | Effort |
|---|---|---|---|
| M1 | Add WAL mode to all four SQLite databases | `sqlite_utils.py` | 1 line each |
| M2 | Fix `_processed` deduplication data race in Feishu channel | `feishu.py` | Small |
| M3 | Fix TOCTOU race in `InteractiveSessionManager.get_or_create_session` | `manager.py` | Small |
| M4 | Close Python sandbox `__import__` escape | `resource_python_runner.py` | Small |
| M5 | Fix `CronScheduler` event-loop capture to use `get_running_loop()` | `cron.py` | Small |

### Should Fix (reliability or maintainability in the near term)

| # | Finding | File | Effort |
|---|---|---|---|
| S1 | Add unbounded-queue protection to `MessageBus` | `message_bus.py` | Small |
| S2 | Add hard max-step safety net to `SingleAgentExecutor` | `executor.py` | Small |
| S3 | Replace `datetime.utcnow()` with `datetime.now(tz=timezone.utc)` | `runtime_job_store.py`, `cron.py` | Trivial |
| S4 | Add missing test for `_detect_message_format()` | `test_feishu_format.py` | Medium |
| S5 | Cache resource briefs within a single orchestration `run()` session | `dynamic_orchestrator.py` | Small |
| S6 | Begin extraction of feedback-command dispatch from `orchestrator.py` | `orchestrator.py` | Medium |
| S7 | Enable `PRAGMA foreign_keys = ON` in `sqlite_utils.py` | `sqlite_utils.py` | 1 line |
| S8 | Validate LLM-parsed `Literal` fields in router output | `orchestrator.py` | Small |
| S9 | Fix `dataclasses.asdict()` call on potential class-not-instance | `message_bus.py` | Small |

### Nice to Have (technical debt, not urgent)

| # | Finding | Effort |
|---|---|---|
| N1 | Remove `CallableTool` dead code from `resource.py` | Trivial |
| N2 | Consolidate magic numbers into `SystemConfig` | Small |
| N3 | Replace hand-rolled YAML parser with `yaml.safe_load` | Medium |
| N4 | Deduplicate schema-inference helpers | Small |
| N5 | Replace `asyncio.run()` inside async functions | Medium (audit + fix) |
| N6 | Add WeChat channel unit tests | Medium |
| N7 | Offload `_select_skill_packs` scoring to `asyncio.to_thread` for large skill sets | Small |

---

## Positive Highlights

The following deserve explicit praise — they represent genuinely high-quality engineering decisions:

- **Worker boundary hardening** (2026-04-01 plan) is thorough. The `ResourceSubagentRuntime.harden_execution_lease()` + `build_create_worker_tool()` depth checks + worker system-prompt constraints form a genuine defence-in-depth boundary. The tests prove the boundary holds.
- **`InProcessChildTaskRuntime`** handles the scheduler-stage guard, dead-letter deduplication, timeout cancellation, and upstream result injection — all with clean, independently-testable logic.
- **`ConservativePolicySelector`** implements a genuinely well-designed UCB-style selector with time-decayed rewards, feedback confidence weighting, and recent-failure guards. The 20+ dedicated tests are exemplary.
- **`Tape` + `HybridMemoryStore` separation of concerns** is clean. The three-tier memory (ephemeral / soft / hard) maps well to the use case and the serialisation format is stable.
- **`ClaudeInteractiveBackend` environment isolation** (custom `HOME`, `TMPDIR`, `CLAUDE_CONFIG_DIR`) is the right call for a hosted multi-tenant environment.
- **AST-based tool registration without import side effects** is a clever and correct design. Using `ast.parse()` instead of `importlib.import_module()` during registration avoids import-time execution and makes skill loading safe and fast.
- **Test suite breadth**: 603 tests for a 12k-line codebase is a strong ratio, and the tests are predominantly behaviour-focused rather than implementation-coupled.
