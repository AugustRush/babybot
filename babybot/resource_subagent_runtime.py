from __future__ import annotations

import contextvars
import logging
import re
import time
from pathlib import Path
from typing import Any

from .agent_kernel import (
    ExecutionContext,
    SkillPack,
    TaskContract,
    TaskResult,
    ToolLease,
)
from .agent_kernel.lease_utils import filter_tool_lease, merge_tool_leases
from .resource_protocols import SubagentHost

logger = logging.getLogger(__name__)

# Matches absolute paths (starting with /) in prose text.
# No extension allow-list — any file that actually exists on disk qualifies.
_ABS_PATH_RE = re.compile(r"(?:^|[\s'\"`(:：=])((?:/[\w.+\-]+)+)")


class ResourceSubagentRuntime:
    _FORBIDDEN_CHILD_GROUPS = frozenset({"worker_control"})
    _FORBIDDEN_CHILD_TOOLS = frozenset(
        {
            "create_worker",
            "dispatch_workers",
            "send_text",
            "send_image",
            "send_file",
            "send_audio",
            "send_card",
            "add_reaction",
        }
    )

    def __init__(self, owner: SubagentHost) -> None:
        self._owner = owner

    @classmethod
    def harden_execution_lease(cls, lease: ToolLease) -> ToolLease:
        return filter_tool_lease(
            lease,
            drop_groups=cls._FORBIDDEN_CHILD_GROUPS,
            drop_group_prefixes=("channel_",),
            drop_tools=cls._FORBIDDEN_CHILD_TOOLS,
            extra_exclude_tools=cls._FORBIDDEN_CHILD_TOOLS,
        )

    @staticmethod
    def merge_skill_leases(
        base_lease: ToolLease,
        skill_packs: list[SkillPack],
    ) -> ToolLease:
        return merge_tool_leases(
            base_lease,
            *(skill.tool_lease for skill in skill_packs),
        )

    @staticmethod
    def executor_skill_packs(skill_packs: list[SkillPack]) -> list[SkillPack]:
        return [
            SkillPack(name=pack.name, system_prompt=pack.system_prompt)
            for pack in skill_packs
        ]

    def _make_after_tool_call_hook(self, write_root: Path) -> Any:
        """Create an after_tool_call hook that extracts file paths from tool outputs.

        Workspace tools (``_workspace_write_text_file``,
        ``_workspace_execute_shell_command``, etc.) have
        ``collect_artifacts=False``, so files they produce are not
        automatically tracked.  This hook scans tool output text for
        **any** absolute file path that:

        1. Actually exists on disk, and
        2. Lives inside the *write_root* directory.

        This is extension-agnostic — no regex extension list to maintain.
        Only files the tool explicitly mentioned in its output are collected,
        so intermediate/temp files the tool didn't report are excluded.
        """
        owner = self._owner
        resolved_root = write_root.resolve()

        def _after_tool_call(
            tool_name: str,
            args: Any,
            result: Any,
            context: ExecutionContext,
        ) -> None:
            if result is None:
                return
            if not getattr(result, "ok", False):
                return
            # If the tool already returned artifacts, the executor's
            # _collect_media already handled them — skip.
            if getattr(result, "artifacts", None):
                return
            content = getattr(result, "content", "") or ""
            if not content:
                return

            # First try the owner's existing regex-based extraction
            # (handles known extensions like .pdf, .png etc.)
            found: list[str] = []
            try:
                found = owner._extract_media_from_text(content)
            except Exception:
                pass

            # Then scan for any absolute path that exists on disk
            # inside write_root — extension-agnostic.
            for match in _ABS_PATH_RE.finditer(content):
                candidate = match.group(1).rstrip(".,;:!?)")
                try:
                    resolved = Path(candidate).resolve()
                except (OSError, ValueError):
                    continue
                try:
                    if not resolved.is_file():
                        continue
                    if not str(resolved).startswith(str(resolved_root)):
                        continue
                except OSError:
                    continue
                path_str = str(resolved)
                if path_str not in found:
                    found.append(path_str)

            if not found:
                return
            bucket: list[str] = context.state.setdefault("media_paths_collected", [])
            existing = set(bucket)
            for path in found:
                if path and path not in existing:
                    bucket.append(path)
                    existing.add(path)

        return _after_tool_call

    def build_execution_context(
        self,
        agent_name: str,
        *,
        heartbeat: Any = None,
        tape: Any = None,
        tape_store: Any = None,
        memory_store: Any = None,
        media_paths: list[str] | None = None,
        channel_context: Any = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> ExecutionContext:
        return ExecutionContext(
            session_id=agent_name,
            state={
                key: value
                for key, value in [
                    ("heartbeat", heartbeat),
                    ("tape", tape),
                    ("tape_store", tape_store),
                    ("memory_store", memory_store),
                    (
                        "context_history_tokens",
                        self._owner.config.system.context_history_tokens,
                    ),
                    ("media_paths", media_paths),
                    ("channel_context", channel_context),
                    ("plan_notebook", plan_notebook),
                    (
                        "plan_notebook_id",
                        getattr(plan_notebook, "notebook_id", "")
                        if plan_notebook is not None
                        else None,
                    ),
                    ("current_notebook_node_id", notebook_node_id),
                ]
                if value is not None
            },
            after_tool_call=self._make_after_tool_call_hook(
                self._owner._get_output_dir()
            ),
        )

    async def run_subagent_task(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: Any = None,
        tape_store: Any = None,
        memory_store: Any = None,
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> tuple[str, list[str]]:
        result = await self.run_subagent_task_result(
            task_description=task_description,
            lease=lease,
            agent_name=agent_name,
            tape=tape,
            tape_store=tape_store,
            memory_store=memory_store,
            heartbeat=heartbeat,
            media_paths=media_paths,
            skill_ids=skill_ids,
            plan_notebook=plan_notebook,
            notebook_node_id=notebook_node_id,
        )
        text = result.output if result.status == "succeeded" else result.error
        return (
            text.strip() or "任务完成但没有文本输出。",
            list(result.artifacts or ()),
        )

    async def run_subagent_task_result(
        self,
        task_description: str,
        lease: dict[str, Any] | None = None,
        agent_name: str = "Worker",
        tape: Any = None,
        tape_store: Any = None,
        memory_store: Any = None,
        heartbeat: Any = None,
        media_paths: list[str] | None = None,
        skill_ids: list[str] | None = None,
        plan_notebook: Any = None,
        notebook_node_id: str = "",
    ) -> TaskResult:
        from .channels.tools import ChannelToolContext

        channel_context = ChannelToolContext.get_current()
        write_root = self._owner._get_output_dir()
        started = time.perf_counter()
        scope_token: contextvars.Token[ToolLease | None] | None = None
        skill_ids_token: contextvars.Token[tuple[str, ...] | None] | None = None
        worker_depth_token: contextvars.Token[int] | None = None
        with self._owner._override_current_write_root(write_root):
            try:
                merged_lease = self._owner._build_task_lease(lease or {})
                skill_packs = await self._owner._select_skill_packs(
                    task_description,
                    skill_ids=skill_ids,
                )
                merged_lease = self.merge_skill_leases(merged_lease, skill_packs)
                merged_lease = self.harden_execution_lease(merged_lease)
                scope_token = self._owner._get_current_task_lease_var().set(
                    merged_lease
                )
                skill_ids_token = self._owner._get_current_skill_ids_var().set(
                    tuple(skill_ids) if skill_ids is not None else None
                )
                parent_depth = self._owner._get_current_worker_depth_var().get()
                worker_depth_token = self._owner._get_current_worker_depth_var().set(
                    int(parent_depth) + 1
                )
                tools_text = (
                    ", ".join(
                        sorted(
                            registered.tool.name
                            for registered in self._owner.registry.list(merged_lease)
                        )
                    )
                    or "无"
                )
                logger.info(
                    "Run subagent agent=%s write_root=%s selected_skills=%s tools=%s include_groups=%s include_tools=%s exclude_tools=%s",
                    agent_name,
                    write_root,
                    [skill.name for skill in skill_packs],
                    tools_text,
                    list(merged_lease.include_groups),
                    list(merged_lease.include_tools),
                    list(merged_lease.exclude_tools),
                )
                sys_prompt = self._owner._build_worker_sys_prompt(
                    agent_name=agent_name,
                    task_description=task_description,
                    tools_text=tools_text,
                    selected_skill_packs=skill_packs,
                    merged_lease=merged_lease,
                )
                executor = self._owner._create_worker_executor(
                    config=self._owner.config,
                    tools=self._owner.registry,
                    sys_prompt=sys_prompt,
                    skill_packs=self.executor_skill_packs(skill_packs),
                    gateway=self._owner._get_shared_gateway(),
                )
                exec_context = self.build_execution_context(
                    agent_name,
                    heartbeat=heartbeat,
                    tape=tape,
                    tape_store=tape_store,
                    memory_store=memory_store,
                    media_paths=media_paths,
                    channel_context=channel_context,
                    plan_notebook=plan_notebook,
                    notebook_node_id=notebook_node_id,
                )
                result = await executor.execute(
                    TaskContract(
                        task_id=agent_name,
                        description=task_description,
                        lease=merged_lease,
                        retries=0,
                    ),
                    exec_context,
                )
                text = result.output if result.status == "succeeded" else result.error
                if result.status != "succeeded":
                    logger.error(
                        "Subagent failed agent=%s status=%s error=%s metadata=%s",
                        agent_name,
                        result.status,
                        result.error,
                        (result.metadata or {}),
                    )
                logger.info(
                    "Run subagent done agent=%s status=%s elapsed=%.2fs output_len=%d",
                    agent_name,
                    result.status,
                    time.perf_counter() - started,
                    len(text or ""),
                )
                collected_media = list(
                    exec_context.state.get("media_paths_collected", [])
                )
                fallback_media = self._owner._extract_media_from_text(text)
                merged_media = tuple(
                    dict.fromkeys(
                        [
                            *tuple(result.artifacts or ()),
                            *collected_media,
                            *fallback_media,
                        ]
                    )
                )
                return TaskResult(
                    task_id=result.task_id,
                    status=result.status,
                    output=result.output,
                    error=result.error,
                    artifacts=merged_media,
                    attempts=result.attempts,
                    metadata=dict(result.metadata or {}),
                )
            except Exception:
                logger.exception("Run subagent crashed agent=%s", agent_name)
                raise
            finally:
                if worker_depth_token is not None:
                    self._owner._get_current_worker_depth_var().reset(
                        worker_depth_token
                    )
                if skill_ids_token is not None:
                    self._owner._get_current_skill_ids_var().reset(skill_ids_token)
                if scope_token is not None:
                    self._owner._get_current_task_lease_var().reset(scope_token)
