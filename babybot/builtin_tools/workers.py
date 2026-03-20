from __future__ import annotations

import asyncio
import json
from typing import Any


def build_create_worker_tool(owner: Any) -> Any:
    async def create_worker(
        task_description: str,
        lease: dict[str, Any] | None = None,
        skill_ids: list[str] | None = None,
    ) -> str:
        inherited_lease = lease
        if inherited_lease is None:
            current_lease = owner._get_current_task_lease_var().get()
            if current_lease is not None:
                inherited_lease = owner._lease_to_dict(current_lease)
        inherited_skill_ids = skill_ids
        if inherited_skill_ids is None:
            current_skill_ids = owner._get_current_skill_ids_var().get()
            if current_skill_ids is not None:
                inherited_skill_ids = list(current_skill_ids)
        text, _ = await owner.run_subagent_task(
            task_description,
            lease=inherited_lease,
            skill_ids=inherited_skill_ids,
        )
        return text

    return create_worker


def build_dispatch_workers_tool(owner: Any) -> Any:
    async def dispatch_workers(
        tasks: list[str],
        max_concurrency: int = 3,
        lease: dict[str, Any] | None = None,
        skill_ids: list[str] | None = None,
    ) -> str:
        normalized = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
        if not normalized:
            return "No valid tasks were provided."
        limit = max(1, min(int(max_concurrency), len(normalized), 8))
        semaphore = asyncio.Semaphore(limit)
        inherited_lease = lease
        if inherited_lease is None:
            current_lease = owner._get_current_task_lease_var().get()
            if current_lease is not None:
                inherited_lease = owner._lease_to_dict(current_lease)
        inherited_skill_ids = skill_ids
        if inherited_skill_ids is None:
            current_skill_ids = owner._get_current_skill_ids_var().get()
            if current_skill_ids is not None:
                inherited_skill_ids = list(current_skill_ids)

        async def run_one(index: int, task: str) -> dict[str, Any]:
            async with semaphore:
                try:
                    text, _ = await owner.run_subagent_task(
                        task_description=task,
                        lease=inherited_lease,
                        agent_name=f"Worker-{index}",
                        skill_ids=inherited_skill_ids,
                    )
                    return {"index": index, "task": task, "result": text}
                except Exception as exc:
                    return {"index": index, "task": task, "error": str(exc)}

        results = await asyncio.gather(
            *(run_one(i, task) for i, task in enumerate(normalized, start=1))
        )
        return json.dumps(
            {"max_concurrency": limit, "results": results},
            ensure_ascii=False,
            indent=2,
        )

    return dispatch_workers


def iter_worker_tool_registrations(owner: Any) -> tuple[tuple[Any, str], ...]:
    return (
        (build_create_worker_tool(owner), "basic"),
        (build_dispatch_workers_tool(owner), "basic"),
    )
