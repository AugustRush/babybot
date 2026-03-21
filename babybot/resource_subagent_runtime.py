from __future__ import annotations

import contextvars
import logging
import time
from typing import Any

from .agent_kernel import ExecutionContext, SkillPack, TaskContract, ToolLease

logger = logging.getLogger(__name__)


class ResourceSubagentRuntime:
    def __init__(self, owner: Any) -> None:
        self._owner = owner

    @staticmethod
    def merge_skill_leases(
        base_lease: ToolLease,
        skill_packs: list[SkillPack],
    ) -> ToolLease:
        merged_lease = base_lease
        for skill in skill_packs:
            merged_lease = ToolLease(
                include_groups=tuple(
                    sorted(
                        set(merged_lease.include_groups)
                        | set(skill.tool_lease.include_groups)
                    )
                ),
                include_tools=tuple(
                    sorted(
                        set(merged_lease.include_tools)
                        | set(skill.tool_lease.include_tools)
                    )
                ),
                exclude_tools=tuple(
                    sorted(
                        set(merged_lease.exclude_tools)
                        | set(skill.tool_lease.exclude_tools)
                    )
                ),
            )
        return merged_lease

    @staticmethod
    def executor_skill_packs(skill_packs: list[SkillPack]) -> list[SkillPack]:
        return [
            SkillPack(name=pack.name, system_prompt=pack.system_prompt)
            for pack in skill_packs
        ]

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
                ]
                if value is not None
            },
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
    ) -> tuple[str, list[str]]:
        from .channels.tools import ChannelToolContext

        write_root = self._owner._get_output_dir()
        token = self._owner._active_write_root.set(str(write_root))
        started = time.perf_counter()
        scope_token: contextvars.Token[ToolLease | None] | None = None
        skill_ids_token: contextvars.Token[tuple[str, ...] | None] | None = None
        try:
            merged_lease = self._owner._build_task_lease(lease or {})
            skill_packs = await self._owner._select_skill_packs(
                task_description,
                skill_ids=skill_ids,
            )
            merged_lease = self.merge_skill_leases(merged_lease, skill_packs)
            scope_token = self._owner._get_current_task_lease_var().set(merged_lease)
            skill_ids_token = self._owner._get_current_skill_ids_var().set(
                tuple(skill_ids) if skill_ids is not None else None
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
                channel_context=ChannelToolContext.get_current(),
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
            collected_media = list(exec_context.state.get("media_paths_collected", []))
            fallback_media = self._owner._extract_media_from_text(text)
            merged_media = list(dict.fromkeys(collected_media + fallback_media))
            return text.strip() or "任务完成但没有文本输出。", merged_media
        except Exception:
            logger.exception("Run subagent crashed agent=%s", agent_name)
            raise
        finally:
            if skill_ids_token is not None:
                self._owner._get_current_skill_ids_var().reset(skill_ids_token)
            if scope_token is not None:
                self._owner._get_current_task_lease_var().reset(scope_token)
            self._owner._active_write_root.reset(token)
