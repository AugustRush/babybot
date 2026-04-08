"""Dynamic orchestration loop — main agent dispatches sub-agents via tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import uuid
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .child_task_events import (
    ChildTaskEvent,
    ChildTaskView,
    InMemoryChildTaskBus,
)
from .child_task_runtime import InProcessChildTaskRuntime
from .context import ContextManager
from .dag_ports import ResourceBridgeExecutor, build_history_summary
from .orchestrator_child_tasks import ChildTaskRuntimeHelper, dispatch_resource_ids
from .execution_constraints import format_execution_constraints_for_prompt
from .model import ModelMessage, ModelRequest, ModelResponse, ModelToolCall
from .orchestrator_config import OrchestratorConfig
from .orchestrator_notebook import NotebookRuntimeHelper
from .plan_notebook import PlanNotebook
from .plan_notebook_context import (
    build_completion_context_view,
    build_orchestrator_context_view,
)
from .prompt_assembly import add_list_section, add_text_section, dedupe_prompt_items
from .runtime_state import RuntimeState
from .team_runtime import TeamDispatchRuntime
from .dynamic_orchestrator_prompt import (
    _DEFERRED_TASK_GUIDANCE,
    _DEFERRED_TASK_PATTERNS,
    _MULTI_STEP_TOKENS,
    _PARALLEL_TOKENS,
    _SYSTEM_PROMPT_ROLE,
    _build_resource_catalog,
    _emit_policy_decision,
    _goal_has_explicit_parallel_intent,
    _is_maintenance_goal,
    _needs_deferred_task_guidance,
    _normalize_recommended_resource_ids,
    _provider_policy_hints,
)
from .dynamic_orchestrator_tools import (
    _ORCHESTRATION_TOOL_BY_NAME,
    _ORCHESTRATION_TOOLS,
)
from .types import (
    ExecutionContext,
    FinalResult,
    SystemPromptBuilder,
    TaskContract,
    TaskResult,
    ToolLease,
)

if TYPE_CHECKING:
    from ..heartbeat import TaskHeartbeatRegistry
    from ..model_gateway import OpenAICompatibleGateway
    from ..resource import ResourceManager
    from .protocols import ExecutorPort

logger = logging.getLogger(__name__)

_FORCE_CONVERGE_TOOL_NAMES = frozenset(
    {"wait_for_tasks", "get_task_result", "reply_to_user"}
)


def _resolve_orchestrator_config(
    config: OrchestratorConfig | None,
) -> OrchestratorConfig:
    if config is not None:
        return config
    try:
        from ..orchestrator_prompts import build_orchestrator_config

        resolved = build_orchestrator_config()
        if isinstance(resolved, OrchestratorConfig):
            return resolved
    except Exception:
        logger.exception(
            "Failed to load application orchestrator config; falling back to generic defaults"
        )
    return OrchestratorConfig()


# ── DynamicOrchestrator ──────────────────────────────────────────────────


class DynamicOrchestrator:
    """Dynamic orchestration loop driven by model tool calls."""

    MAX_STEPS = 30
    MAX_TASKS = 20

    def __init__(
        self,
        resource_manager: "ResourceManager",
        gateway: "OpenAICompatibleGateway",
        child_task_bus: InMemoryChildTaskBus | None = None,
        task_heartbeat_registry: "TaskHeartbeatRegistry | None" = None,
        task_stale_after_s: float | None = None,
        max_steps: int | None = None,
        default_task_timeout_s: float | None = 300.0,
        executor_registry: "ExecutorPort | None" = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        from ..heartbeat import TaskHeartbeatRegistry

        self._rm = resource_manager
        self._gateway = gateway
        self._bridge = ResourceBridgeExecutor(resource_manager, gateway)
        self._executor_registry = executor_registry
        self._child_task_bus = child_task_bus or InMemoryChildTaskBus()
        self._task_heartbeat_registry = (
            task_heartbeat_registry or TaskHeartbeatRegistry()
        )
        self._task_stale_after_s = task_stale_after_s
        self._max_steps = max(1, int(max_steps or self.MAX_STEPS))
        self._default_task_timeout_s = default_task_timeout_s
        self._resource_catalog_cache_key: tuple[str, ...] | None = None
        self._resource_catalog_cache_value = ""
        self._config = _resolve_orchestrator_config(config)
        self._child_task_runtime = ChildTaskRuntimeHelper(
            self._config,
            is_maintenance_goal=_is_maintenance_goal,
            has_parallel_intent=_goal_has_explicit_parallel_intent,
        )
        self._notebook_runtime = NotebookRuntimeHelper(self._config)
        self._team_runtime = TeamDispatchRuntime(
            resource_manager=self._rm,
            gateway=self._gateway,
            executor=self._executor,
            config=self._config,
            notebook_runtime=self._notebook_runtime,
        )
        self._orchestration_tools = self._build_orchestration_tools()

    @property
    def _executor(self) -> ExecutorPort:
        """Return the executor registry if provided, otherwise fall back to the bridge."""
        if self._executor_registry is not None:
            return self._executor_registry
        return self._bridge

    def _build_orchestration_tools(self) -> tuple[dict[str, Any], ...]:
        """Merge config-supplied descriptions into the static tool schemas."""
        cfg = self._config
        td = cfg.tool_descriptions
        pd = cfg.tool_param_descriptions

        def _desc(tool_name: str, fallback: str) -> str:
            return td.get(tool_name) or fallback

        def _pdesc(tool_name: str, param: str, fallback: str) -> str:
            return (pd.get(tool_name) or {}).get(param) or fallback

        tools = list(_ORCHESTRATION_TOOLS)
        result: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool["function"]
            name = fn["name"]
            patched_fn: dict[str, Any] = dict(fn)
            patched_fn["description"] = _desc(name, fn.get("description", ""))
            # Patch parameter descriptions if provided
            orig_params: dict[str, Any] = fn.get("parameters", {})
            if pd.get(name):
                patched_props = {}
                for pname, pschema in orig_params.get("properties", {}).items():
                    patched_prop = dict(pschema)
                    override = _pdesc(name, pname, "")
                    if override:
                        patched_prop["description"] = override
                    patched_props[pname] = patched_prop
                patched_params = dict(orig_params)
                patched_params["properties"] = patched_props
                patched_fn["parameters"] = patched_params
            result.append({"type": "function", "function": patched_fn})
        return tuple(result)

    def _augment_dispatch_resources(
        self,
        dispatch_args: dict[str, Any],
        *,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        resource_ids = list(dispatch_resource_ids(dispatch_args))
        original_goal = str(context.state.get("original_goal", "") or "").strip()
        description = str(dispatch_args.get("description", "") or "").strip()
        signal_text = "\n".join(part for part in (original_goal, description) if part)
        if not signal_text:
            return dispatch_args

        recommender = getattr(self._rm, "recommend_resources", None)
        if not callable(recommender):
            recommender = getattr(self._rm, "_recommend_resources", None)
        if not callable(recommender):
            return dispatch_args

        try:
            recommendation = recommender(signal_text, limit=4)
        except Exception:
            logger.exception("resource recommendation raised; using raw dispatch args")
            return dispatch_args

        primary_ids = list(
            _normalize_recommended_resource_ids(
                recommendation,
                "primary_resource_ids",
            )
        )
        supporting_ids = list(
            _normalize_recommended_resource_ids(
                recommendation,
                "supporting_resource_ids",
            )
        )
        recommended_ids = list(
            _normalize_recommended_resource_ids(recommendation, "resource_ids")
        )
        if not primary_ids and not supporting_ids and not recommended_ids:
            return dispatch_args

        additions: list[str] = []
        if not resource_ids and recommended_ids:
            additions.extend(recommended_ids)
        elif primary_ids and not set(resource_ids) & set(primary_ids):
            additions.append(primary_ids[0])
        for resource_id in supporting_ids:
            if resource_id not in resource_ids and resource_id not in additions:
                additions.append(resource_id)

        if not additions:
            return dispatch_args

        merged_resource_ids = list(dict.fromkeys([*resource_ids, *additions]))
        augmented = dict(dispatch_args)
        if len(merged_resource_ids) == 1:
            augmented["resource_id"] = merged_resource_ids[0]
            augmented.pop("resource_ids", None)
            return augmented

        augmented["resource_ids"] = merged_resource_ids
        augmented.pop("resource_id", None)
        return augmented

    async def run(self, goal: str, context: ExecutionContext) -> FinalResult:
        task_counter = 0
        context.state.setdefault("original_goal", goal)
        self._ensure_plan_notebook(goal, context)
        context.state.setdefault(
            "original_request_header",
            self._config.original_request_header,
        )
        context.state.setdefault(
            "upstream_results_header",
            self._config.upstream_results_header,
        )
        heartbeat = context.state.get("heartbeat")
        reply_text: str | None = None
        scheduler_handoff_created = False
        flow_id = context.session_id or "orchestrator"
        runtime = InProcessChildTaskRuntime(
            flow_id=flow_id,
            resource_manager=self._rm,
            bridge=self._executor,
            child_task_bus=self._child_task_bus,
            task_heartbeat_registry=self._task_heartbeat_registry,
            max_parallel=4,
            max_tasks=self.MAX_TASKS,
            default_timeout_s=self._default_task_timeout_s,
            stale_after_s=self._task_stale_after_s,
            plan_step_id=(
                context.state.get("execution_plan").steps[0].step_id
                if getattr(context.state.get("execution_plan"), "steps", None)
                else ""
            ),
            task_title_builder=self._task_title_from_description,
        )
        runtime_event_callback = context.state.get("runtime_event_callback")
        forwarder_task: asyncio.Task[None] | None = None
        if runtime_event_callback is not None:
            forwarder_task = asyncio.create_task(
                self._forward_runtime_events(flow_id, runtime_event_callback)
            )

        messages = self._build_initial_messages(goal, context)
        logger.info("DynamicOrchestrator: initial messages built, entering main loop")
        try:
            for step in range(self._max_steps):
                if heartbeat is not None:
                    heartbeat.beat()

                messages = self._prune_stale_wait_history(messages)
                messages = self._prune_messages_by_count(messages)
                force_converge_reason = self._refresh_force_converge_state(
                    runtime=runtime,
                    context=context,
                )
                if force_converge_reason:
                    messages.append(
                        ModelMessage(
                            role="system",
                            content=(
                                "Runtime update:\n"
                                f"{force_converge_reason}\n"
                                "Only use get_task_result, wait_for_tasks, or "
                                "reply_to_user until this stage converges."
                            ),
                        )
                    )
                logger.debug("DynamicOrchestrator: step=%d calling model", step)
                response = await self._call_model(messages, context, step=step)
                response = ModelResponse(
                    text=response.text,
                    tool_calls=self._merge_dispatch_calls_for_maintenance(
                        response.tool_calls,
                        goal=str(context.state.get("original_goal", goal) or goal),
                    ),
                    finish_reason=response.finish_reason,
                    metadata=dict(response.metadata),
                )

                # Model responded with plain text (no tool calls)
                if not response.tool_calls:
                    return FinalResult(
                        conclusion=response.text,
                        task_results=runtime.results,
                    )

                reply_call_count = sum(
                    1
                    for tool_call in response.tool_calls
                    if tool_call.name == "reply_to_user"
                )
                if reply_call_count and len(response.tool_calls) > 1:
                    response = ModelResponse(
                        text=response.text,
                        tool_calls=tuple(
                            tool_call
                            for tool_call in response.tool_calls
                            if tool_call.name != "reply_to_user"
                        )
                        + (
                            ModelToolCall(
                                call_id="reply_guard",
                                name="reply_to_user",
                                arguments={
                                    "text": (
                                        "error: reply_to_user must be the only tool call in its turn"
                                    )
                                },
                            ),
                        ),
                        finish_reason="tool_calls",
                    )

                # Append assistant message with all tool_calls
                messages.append(
                    ModelMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )

                # Process each tool call
                scheduler_dispatch_succeeded_this_turn = False
                prior_live_task_ids_this_turn: list[str] = []
                scheduler_dispatch_present_this_turn = any(
                    tc.name == "dispatch_task"
                    and tc.arguments.get("resource_id") == "group.scheduler"
                    for tc in response.tool_calls
                )
                for tc in response.tool_calls:
                    result_text = await self._dispatch_tool(
                        tc,
                        runtime,
                        context,
                        task_counter,
                        scheduler_handoff_created=scheduler_handoff_created,
                        scheduler_dispatch_present_this_turn=scheduler_dispatch_present_this_turn,
                        scheduler_dispatch_seen_before_call=scheduler_dispatch_succeeded_this_turn,
                        prior_live_task_ids_this_turn=tuple(
                            prior_live_task_ids_this_turn
                        ),
                    )
                    if tc.name == "dispatch_task" and not result_text.startswith(
                        "error:"
                    ):
                        task_counter += 1
                        resource_id = tc.arguments.get("resource_id")
                        if resource_id == "group.scheduler":
                            scheduler_dispatch_succeeded_this_turn = True
                        else:
                            prior_live_task_ids_this_turn.append(result_text)
                    messages.append(
                        ModelMessage(
                            role="tool",
                            content=result_text,
                            tool_call_id=tc.call_id,
                        )
                    )
                    if tc.name == "reply_to_user":
                        if result_text.startswith("error:"):
                            continue
                        reply_text = tc.arguments.get("text", result_text)

                if scheduler_dispatch_succeeded_this_turn:
                    scheduler_handoff_created = True

                if reply_text is not None:
                    await runtime.cancel_all()
                    return FinalResult(
                        conclusion=reply_text, task_results=runtime.results
                    )

            await runtime.cancel_all(grace_period_s=5.0)
            return self._build_fallback_result(
                goal,
                runtime.results,
                notebook=context.state.get("plan_notebook"),
            )
        finally:
            if forwarder_task is not None:
                forwarder_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder_task
            self._child_task_bus.clear_flow(flow_id)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_initial_messages(
        self,
        goal: str,
        context: ExecutionContext,
    ) -> list[ModelMessage]:
        state_view = RuntimeState(context)
        briefs = self._rm.get_resource_briefs()
        tape = state_view.get("tape")
        memory_store = state_view.get("memory_store")
        history = build_history_summary(tape, memory_store=memory_store, query=goal)
        media_paths = state_view.media_paths()
        execution_constraints = state_view.get("execution_constraints")
        deduped_policy_hints = dedupe_prompt_items(
            [
                *state_view.policy_hints(),
                *_provider_policy_hints(self._rm, goal, self._config),
            ],
            limit=20,
        )
        state_view.set_policy_hints(deduped_policy_hints)

        builder = SystemPromptBuilder()
        add_text_section(
            builder,
            "identity",
            self._config.system_prompt or _SYSTEM_PROMPT_ROLE,
            priority=0,
            cacheable=True,
        )
        add_text_section(
            builder,
            "resource_catalog",
            self._resource_catalog_text(briefs),
            priority=10,
        )
        if _needs_deferred_task_guidance(goal, self._config):
            add_text_section(
                builder,
                "deferred_task_guidance",
                self._config.deferred_task_guidance or _DEFERRED_TASK_GUIDANCE,
                priority=5,
            )
        addendum_builder = getattr(
            self._config, "build_resource_selection_addendum", None
        )
        if callable(addendum_builder):
            addendum = addendum_builder(briefs)
            if addendum:
                add_text_section(
                    builder,
                    "resource_selection_addendum",
                    addendum,
                    priority=15,
                )
        notebook_binding = state_view.notebook_binding()
        if notebook_binding.active:
            notebook_context_budget = state_view.notebook_context_budget(default=2400)
            notebook_view = build_orchestrator_context_view(
                notebook_binding.notebook,
                token_budget=notebook_context_budget,
                current_node_id=notebook_binding.node_id,
            )
            if notebook_view.text:
                add_text_section(
                    builder,
                    "notebook_context",
                    notebook_view.text,
                    priority=20,
                    header="\n[Notebook Context]\n",
                )
        if history:
            add_text_section(builder, "history", history, priority=30, header="\n")
        if deduped_policy_hints:
            add_list_section(
                builder,
                "policy_hints",
                deduped_policy_hints,
                priority=40,
                header=self._config.policy_hints_header or "\nPolicy hints:\n",
            )

        # execution_constraints is dynamic and request-scoped — inject it as a
        # dedicated context block immediately before the user message so it is
        # clearly separated from the static role/catalog and from conversation
        # history, making it easier for the model to locate and respect.
        user_context_prefix = ""
        if execution_constraints:
            constraints_text = format_execution_constraints_for_prompt(
                execution_constraints
            )
            wrapper = self._config.execution_constraints_wrapper
            if wrapper and "{constraints}" in wrapper and "{goal}" in wrapper:
                user_context_prefix = wrapper.split("{goal}")[0].replace(
                    "{constraints}", constraints_text
                )
            else:
                user_context_prefix = (
                    f"[Constraints]\n{constraints_text}\n\n[Request]\n"
                )

        return [
            ModelMessage(role="system", content=builder.build()),
            ModelMessage(
                role="user",
                content=user_context_prefix + goal,
                images=tuple(media_paths),
            ),
        ]

    def _resource_catalog_text(self, briefs: list[dict[str, Any]]) -> str:
        cache_key = tuple(
            json.dumps(brief, ensure_ascii=False, sort_keys=True) for brief in briefs
        )
        if cache_key != self._resource_catalog_cache_key:
            self._resource_catalog_cache_key = cache_key
            self._resource_catalog_cache_value = _build_resource_catalog(
                briefs, self._config
            )
        return self._resource_catalog_cache_value

    def _ensure_plan_notebook(
        self,
        goal: str,
        context: ExecutionContext,
    ) -> PlanNotebook:
        return self._notebook_runtime.ensure_plan_notebook(goal, context)

    @staticmethod
    def _notebook_task_map(context: ExecutionContext) -> dict[str, str]:
        return NotebookRuntimeHelper.notebook_task_map(context)

    @staticmethod
    def _task_title_from_description(description: str) -> str:
        return NotebookRuntimeHelper.task_title_from_description(description)

    @staticmethod
    def _current_notebook_node(
        notebook: PlanNotebook,
        context: ExecutionContext,
    ) -> Any:
        return NotebookRuntimeHelper.current_notebook_node(notebook, context)

    @staticmethod
    def _child_notebook_nodes(notebook: PlanNotebook, parent_id: str) -> list[Any]:
        return NotebookRuntimeHelper.child_notebook_nodes(notebook, parent_id)

    def _complete_converged_execution_nodes(self, notebook: PlanNotebook) -> None:
        self._notebook_runtime.complete_converged_execution_nodes(notebook)

    def _update_notebook_from_task_payloads(
        self,
        *,
        context: ExecutionContext,
        runtime: InProcessChildTaskRuntime,
        task_ids: list[str],
    ) -> None:
        self._notebook_runtime.update_notebook_from_task_payloads(
            context=context,
            runtime=runtime,
            task_ids=task_ids,
        )

    def _notebook_blocking_node_ids(
        self,
        notebook: PlanNotebook,
    ) -> list[str]:
        return self._notebook_runtime.notebook_blocking_node_ids(notebook)

    @staticmethod
    def _notebook_reply_blocking_checkpoints(
        notebook: PlanNotebook,
    ) -> list[str]:
        return NotebookRuntimeHelper.notebook_reply_blocking_checkpoints(notebook)

    def _force_converge_state(
        self,
        *,
        runtime: InProcessChildTaskRuntime,
        context: ExecutionContext,
    ) -> tuple[str, dict[str, Any]] | None:
        return self._notebook_runtime.force_converge_state(
            runtime=runtime,
            context=context,
        )

    def _refresh_force_converge_state(
        self,
        *,
        runtime: InProcessChildTaskRuntime,
        context: ExecutionContext,
    ) -> str | None:
        return self._notebook_runtime.refresh_force_converge_state(
            runtime=runtime,
            context=context,
        )

    def _finalize_notebook_for_reply(
        self,
        *,
        context: ExecutionContext,
        reply_text: str,
    ) -> None:
        self._notebook_runtime.finalize_notebook_for_reply(
            context=context,
            reply_text=reply_text,
        )

    def _merge_dispatch_calls_for_maintenance(
        self,
        tool_calls: tuple[ModelToolCall, ...],
        *,
        goal: str,
    ) -> tuple[ModelToolCall, ...]:
        return self._child_task_runtime.merge_dispatch_calls_for_maintenance(
            tool_calls,
            goal=goal,
        )

    def _maintenance_serial_dependency_ids(
        self,
        runtime: InProcessChildTaskRuntime,
        *,
        prior_live_task_ids_this_turn: tuple[str, ...],
    ) -> list[str]:
        return self._child_task_runtime.maintenance_serial_dependency_ids(
            runtime,
            prior_live_task_ids_this_turn=prior_live_task_ids_this_turn,
        )

    def _normalize_child_task_description(
        self,
        *,
        description: str,
        resource_ids: tuple[str, ...],
        context: ExecutionContext,
        upstream_results: dict[str, "TaskResult"] | None = None,
    ) -> str:
        return self._child_task_runtime.normalize_child_task_description(
            description=description,
            resource_ids=resource_ids,
            context=context,
            upstream_results=upstream_results,
        )

    def _build_child_task_feedback_label(
        self,
        *,
        description: str,
        resource_ids: tuple[str, ...],
        context: ExecutionContext,
    ) -> str:
        return self._child_task_runtime.build_child_task_feedback_label(
            description=description,
            resource_ids=resource_ids,
            context=context,
        )

    @staticmethod
    def _recent_successful_upstream_results(
        results: dict[str, "TaskResult"],
        *,
        limit: int = 1,
    ) -> dict[str, "TaskResult"] | None:
        return ChildTaskRuntimeHelper.recent_successful_upstream_results(
            results,
            limit=limit,
        )

    @staticmethod
    def _prune_stale_wait_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        wait_call_ids: list[str] = []
        for message in messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if tool_call.name == "wait_for_tasks" and tool_call.call_id:
                    wait_call_ids.append(tool_call.call_id)

        if len(wait_call_ids) <= 1:
            return messages

        keep_wait_call_id = wait_call_ids[-1]
        stale_wait_call_ids = set(wait_call_ids[:-1])
        pruned: list[ModelMessage] = []

        for message in messages:
            if message.role == "tool" and message.tool_call_id in stale_wait_call_ids:
                continue
            if message.role == "assistant" and message.tool_calls:
                filtered_tool_calls = tuple(
                    tool_call
                    for tool_call in message.tool_calls
                    if tool_call.call_id not in stale_wait_call_ids
                )
                if filtered_tool_calls != message.tool_calls:
                    if not filtered_tool_calls and not message.content.strip():
                        continue
                    pruned.append(
                        ModelMessage(
                            role=message.role,
                            content=message.content,
                            name=message.name,
                            tool_call_id=message.tool_call_id,
                            tool_calls=filtered_tool_calls,
                            images=message.images,
                        )
                    )
                    continue
            pruned.append(message)

        tool_ids_present = {
            message.tool_call_id
            for message in pruned
            if message.role == "tool" and message.tool_call_id
        }
        if keep_wait_call_id not in tool_ids_present:
            return messages
        return pruned

    @staticmethod
    def _prune_messages_by_count(
        messages: list[ModelMessage],
        *,
        max_non_system: int = 80,
        keep_recent: int = 60,
    ) -> list[ModelMessage]:
        """Fallback count-based pruning to prevent unbounded message growth.

        When the total number of non-system messages exceeds ``max_non_system``,
        only the system message (index 0) plus the most recent ``keep_recent``
        non-system messages are retained.  This is a safety net — it does NOT
        do token-based compaction, but it bounds worst-case prompt size in long
        task chains.
        """
        if not messages:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= max_non_system:
            return messages

        logger.warning(
            "DynamicOrchestrator: message count exceeded %d (actual=%d); "
            "pruning to %d most recent non-system messages",
            max_non_system,
            len(non_system),
            keep_recent,
        )
        return system_msgs + non_system[-keep_recent:]

    async def _call_model(
        self,
        messages: list[ModelMessage],
        context: ExecutionContext,
        *,
        step: int = 0,
    ) -> ModelResponse:
        heartbeat = context.state.get("heartbeat")
        stream_callback = context.state.get("stream_callback")
        request = ModelRequest(
            messages=tuple(messages),
            tools=self._tool_schemas_for_context(context),
        )
        state = {
            k: v
            for k, v in [
                ("heartbeat", heartbeat),
                ("stream_callback", stream_callback),
            ]
            if v is not None
        }
        ctx = ExecutionContext(state=state)
        logger.info("DynamicOrchestrator: step=%d LLM call start", step)
        if heartbeat is not None:
            async with heartbeat.keep_alive():
                response = await self._gateway.generate(request, ctx)
        else:
            response = await self._gateway.generate(request, ctx)
        logger.info(
            "DynamicOrchestrator: step=%d LLM call done finish_reason=%s tool_calls=%d",
            step,
            response.finish_reason,
            len(response.tool_calls),
        )
        return response

    def _tool_schemas_for_context(
        self,
        context: ExecutionContext,
    ) -> tuple[dict[str, Any], ...]:
        execution_plan = context.state.get("execution_plan")
        allowed = ()
        if getattr(execution_plan, "steps", None):
            payload = getattr(execution_plan.steps[0], "payload", {}) or {}
            allowed = tuple(payload.get("allowed_tools") or ())
        if not allowed:
            task_contract = context.state.get("task_contract")
            allowed = tuple(getattr(task_contract, "allowed_tools", ()) or ())
        if not allowed:
            selected = self._orchestration_tools
        else:
            tool_by_name = {t["function"]["name"]: t for t in self._orchestration_tools}
            filtered = tuple(
                tool_by_name[name] for name in allowed if name in tool_by_name
            )
            selected = filtered or self._orchestration_tools
        if str(context.state.get("orchestrator_force_converge", "") or "").strip():
            convergence_tools = tuple(
                tool
                for tool in selected
                if tool["function"]["name"] in _FORCE_CONVERGE_TOOL_NAMES
            )
            if convergence_tools:
                return convergence_tools
            return tuple(
                tool
                for tool in self._orchestration_tools
                if tool["function"]["name"] in _FORCE_CONVERGE_TOOL_NAMES
            )
        return selected

    async def _dispatch_tool(
        self,
        tool_call: ModelToolCall,
        runtime: InProcessChildTaskRuntime,
        context: ExecutionContext,
        task_counter: int,
        *,
        scheduler_handoff_created: bool,
        scheduler_dispatch_present_this_turn: bool,
        scheduler_dispatch_seen_before_call: bool,
        prior_live_task_ids_this_turn: tuple[str, ...],
    ) -> str:
        name = tool_call.name
        args = tool_call.arguments

        if args.get("__tool_argument_parse_error__"):
            return f"error: invalid arguments: {args.get('__raw_arguments__', '')}"

        force_converge_reason = str(
            context.state.get("orchestrator_force_converge", "") or ""
        ).strip()
        if force_converge_reason and name in {"dispatch_task", "dispatch_team"}:
            return (
                "error: convergence required: "
                f"{force_converge_reason} "
                "Do not dispatch new child tasks; use get_task_result, "
                "wait_for_tasks, or reply_to_user."
            )

        if name == "dispatch_task":
            dispatch_args = self._augment_dispatch_resources(
                dict(args),
                context=context,
            )
            resource_ids = dispatch_resource_ids(dispatch_args)
            raw_description = str(dispatch_args.get("description", "") or "")
            feedback_label = self._build_child_task_feedback_label(
                description=raw_description,
                resource_ids=resource_ids,
                context=context,
            )
            deps = list(dispatch_args.get("deps", []) or [])
            active_in_flight = len(runtime.in_flight)
            scheduler_dispatch = "group.scheduler" in resource_ids
            maintenance_single_owner = (
                not scheduler_dispatch
                and _is_maintenance_goal(
                    str(context.state.get("original_goal", "") or ""),
                    self._config,
                )
                and not _goal_has_explicit_parallel_intent(
                    str(context.state.get("original_goal", "") or ""),
                    self._config,
                )
            )
            if maintenance_single_owner and not deps:
                serialized_deps = self._maintenance_serial_dependency_ids(
                    runtime,
                    prior_live_task_ids_this_turn=prior_live_task_ids_this_turn,
                )
                if serialized_deps:
                    dispatch_args["deps"] = serialized_deps
                    deps = list(serialized_deps)
            if (
                not deps
                and not scheduler_dispatch
                and prior_live_task_ids_this_turn
                and not _goal_has_explicit_parallel_intent(
                    str(context.state.get("original_goal", "") or ""),
                    self._config,
                )
            ):
                current_resources = {rid for rid in resource_ids if rid}
                for prior_task_id in prior_live_task_ids_this_turn:
                    prior_state = runtime.task_state_snapshot(prior_task_id)
                    prior_resources = {
                        str(item).strip()
                        for item in (prior_state.get("resource_ids") or [])
                        if str(item).strip()
                    }
                    if not current_resources or not prior_resources:
                        continue
                    if current_resources & prior_resources and (
                        len(current_resources) > 1 or len(prior_resources) > 1
                    ):
                        return (
                            "error: overlapping sibling dispatches in the same turn "
                            "must be merged into one dispatch_task with resource_ids, "
                            "or the later task must declare deps for serial execution"
                        )
            if "group.scheduler" not in resource_ids:
                # Collect completed upstream results for dep tasks so the child task
                # knows what its dependencies produced (break the black-box problem).
                upstream_results = {
                    tid: runtime.results[tid] for tid in deps if tid in runtime.results
                } or None
                if (
                    upstream_results is None
                    and not deps
                    and not prior_live_task_ids_this_turn
                    and active_in_flight == 0
                ):
                    upstream_results = self._recent_successful_upstream_results(
                        runtime.results
                    )
                # Populate the shared upstream_results bucket so
                # _enrich_with_upstream in dag_ports.py can inject them
                # into the worker prompt (single injection point).
                if upstream_results:
                    bucket = context.state.setdefault("upstream_results", {})
                    for tid, result in upstream_results.items():
                        output = getattr(result, "output", result)
                        bucket[tid] = str(output or "")
                dispatch_args["__public_description"] = raw_description
                dispatch_args["__feedback_label"] = feedback_label
                dispatch_args["description"] = self._normalize_child_task_description(
                    description=raw_description,
                    resource_ids=resource_ids,
                    context=context,
                    upstream_results=upstream_results,
                )
            scheduling_action = (
                "parallel_dispatch"
                if not deps and (prior_live_task_ids_this_turn or active_in_flight > 0)
                else "serial_dispatch"
            )
            _emit_policy_decision(
                context,
                decision_kind="scheduling",
                action_name=scheduling_action,
                state_features={
                    "deps_count": len(deps),
                    "active_in_flight": active_in_flight,
                    "resource_count": len(resource_ids),
                },
            )
            if (
                scheduler_dispatch
                and not dispatch_args.get("deps")
                and prior_live_task_ids_this_turn
            ):
                dispatch_args["deps"] = list(prior_live_task_ids_this_turn)
            if scheduler_handoff_created and not scheduler_dispatch:
                return (
                    "error: scheduled handoff already created in this flow; "
                    "do not dispatch additional live tasks after scheduling future work"
                )
            if scheduler_dispatch_seen_before_call and not scheduler_dispatch:
                return (
                    "error: scheduled handoff was already created earlier in this turn; "
                    "do not dispatch additional live tasks after it"
                )
            if (
                runtime.results
                and scheduler_dispatch_present_this_turn
                and not scheduler_dispatch
            ):
                return (
                    "error: scheduled handoff is being created for a later stage; "
                    "do not mix new live tasks into the same turn"
                )
            runtime_dispatch_args = dict(dispatch_args)
            runtime_dispatch_args["__task_kind"] = (
                "scheduled_task" if scheduler_dispatch else "child_task"
            )
            task_id = await runtime.dispatch(
                runtime_dispatch_args, task_counter=task_counter, context=context
            )
            return task_id
        if name == "wait_for_tasks":
            _emit_policy_decision(
                context,
                decision_kind="scheduling",
                action_name="wait_barrier",
                state_features={
                    "task_count": len(args.get("task_ids", []) or []),
                },
            )
            task_ids = list(args.get("task_ids", []) or [])
            payload = await runtime.wait_for_tasks(task_ids)
            self._update_notebook_from_task_payloads(
                context=context,
                runtime=runtime,
                task_ids=task_ids,
            )
            return payload
        if name == "get_task_result":
            task_id = args.get("task_id", "")
            payload = runtime.get_task_result(task_id)
            self._update_notebook_from_task_payloads(
                context=context,
                runtime=runtime,
                task_ids=[task_id],
            )
            return payload
        if name == "reply_to_user":
            pending = runtime.pending_reply_blocking_task_ids()
            if pending:
                return (
                    "error: reply_to_user called before child tasks finished; "
                    f"pending tasks: {', '.join(pending)}"
                )
            self._update_notebook_from_task_payloads(
                context=context,
                runtime=runtime,
                task_ids=list(runtime.results.keys()),
            )
            notebook = self._ensure_plan_notebook(
                str(context.state.get("original_goal", "") or ""),
                context,
            )
            self._complete_converged_execution_nodes(notebook)
            all_dead, dead_ids = runtime.all_tasks_dead_lettered_with_no_success()
            blocking_nodes = self._notebook_blocking_node_ids(notebook)
            if blocking_nodes:
                return (
                    "error: reply_to_user called before notebook frontier converged; "
                    f"blocking notebook nodes: {', '.join(blocking_nodes)}"
                )
            open_checkpoints = self._notebook_reply_blocking_checkpoints(notebook)
            if open_checkpoints:
                return (
                    "error: reply_to_user called before notebook checkpoints were resolved; "
                    f"open notebook checkpoints: {', '.join(open_checkpoints)}"
                )
            if all_dead:
                errors = "; ".join(
                    str(runtime.results[tid].error or "unknown error")[:120]
                    for tid in dead_ids
                )
                logger.warning(
                    "reply_to_user after all tasks dead-lettered (%s); "
                    "allowing direct failure reply. errors: %s",
                    ", ".join(dead_ids),
                    errors,
                )
            self._finalize_notebook_for_reply(
                context=context,
                reply_text=str(args.get("text", "") or ""),
            )
            return args.get("text", "")
        if name == "dispatch_team":
            return await self._run_team(args, context)
        return f"error: unknown tool: {name}"

    async def _run_team(self, args: dict[str, Any], context: ExecutionContext) -> str:
        return await self._team_runtime.run(args, context)

    # DAG task events that carry meaningful user-facing progress.
    # "started" is included so the progress card updates when a subtask
    # begins executing, giving the user visible activity during long tasks.
    _FORWARD_EVENT_ALLOWLIST: frozenset[str] = frozenset(
        {
            "queued",
            "started",
            "succeeded",
            "failed",
            "dead_lettered",
            "stalled",
            "cancelled",
        }
    )

    async def _forward_runtime_events(
        self,
        flow_id: str,
        callback: Callable[[dict[str, Any]], Any],
    ) -> None:
        async for event in self._child_task_bus.subscribe(flow_id):
            if event.event not in self._FORWARD_EVENT_ALLOWLIST:
                continue
            result = callback(dataclasses.asdict(event))
            if inspect.isawaitable(result):
                await result

    def _build_fallback_result(
        self,
        goal: str,
        results: dict[str, TaskResult],
        *,
        notebook: PlanNotebook | None = None,
    ) -> FinalResult:
        del goal
        parts = [self._config.step_budget_exhausted_header]
        if notebook is not None:
            completion_view = build_completion_context_view(notebook, token_budget=1200)
            if completion_view.text:
                parts.append(completion_view.text)
        for task_id, r in results.items():
            if r.status == "succeeded":
                parts.append(
                    self._config.step_budget_succeeded_line.format(
                        task_id=task_id, output=r.output or "done"
                    )
                )
            else:
                parts.append(
                    self._config.step_budget_failed_line.format(
                        task_id=task_id, error=r.error or ""
                    )
                )
        return FinalResult(conclusion="\n".join(parts), task_results=results)
