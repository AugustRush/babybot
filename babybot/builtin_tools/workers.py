from __future__ import annotations

import asyncio
import json
from typing import Any


def _worker_policy_features(
    *,
    task_description: str = "",
    task_count: int = 1,
) -> dict[str, Any]:
    text = str(task_description or "").strip()
    return {
        "task_shape": "multi_step"
        if task_count > 1 or any(token in text for token in ("然后", "再", "并且", "同时", "先"))
        else "single_step",
        "task_count": max(1, int(task_count)),
        "independent_subtasks": max(1, int(task_count)),
        "input_length": len(text),
    }


def _normalize_policy_action(payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        action_name = str(payload.get("action_name") or payload.get("name") or "").strip()
        hint = str(payload.get("hint") or "").strip()
        return action_name, hint
    action_name = str(
        getattr(payload, "action_name", "") or getattr(payload, "name", "") or ""
    ).strip()
    hint = str(getattr(payload, "hint", "") or "").strip()
    return action_name, hint


def _policy_denial_message(
    owner: Any,
    *,
    task_description: str,
    task_count: int,
    operation: str,
) -> str | None:
    provider = getattr(owner, "_observability_provider", None)
    chooser = getattr(provider, "choose_worker_policy", None)
    if not callable(chooser):
        return None
    payload = chooser(
        features=_worker_policy_features(
            task_description=task_description,
            task_count=task_count,
        )
    )
    action_name, hint = _normalize_policy_action(payload)
    if action_name != "deny_worker":
        return None
    suffix = f": {hint}" if hint else "."
    return f"Policy denied {operation}{suffix}"


def build_create_worker_tool(owner: Any) -> Any:
    async def create_worker(
        task_description: str,
        lease: dict[str, Any] | None = None,
        skill_ids: list[str] | None = None,
    ) -> str:
        system_conf = getattr(getattr(owner, "config", None), "system", None)
        max_depth = max(1, int(getattr(system_conf, "worker_max_depth", 3) or 3))
        depth_var = getattr(owner, "_get_current_worker_depth_var", None)
        current_depth = int(depth_var().get()) if callable(depth_var) else 0
        if current_depth > 0:
            return (
                "Normal child workers cannot create nested workers. "
                "Return the current task result to the main orchestrator instead."
            )
        if current_depth >= max_depth:
            return (
                f"Max worker depth reached ({current_depth}/{max_depth}). "
                "Finish in the current worker instead of creating another nested worker."
            )
        denial = _policy_denial_message(
            owner,
            task_description=task_description,
            task_count=1,
            operation="worker creation",
        )
        if denial is not None:
            return denial
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
        timeout_s: float | None = None,
    ) -> str:
        normalized = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
        if not normalized:
            return "No valid tasks were provided."
        depth_var = getattr(owner, "_get_current_worker_depth_var", None)
        current_depth = int(depth_var().get()) if callable(depth_var) else 0
        if current_depth > 0:
            return (
                "Normal child workers cannot dispatch nested workers. "
                "Return the current task result to the main orchestrator instead."
            )
        denial = _policy_denial_message(
            owner,
            task_description="\n".join(normalized),
            task_count=len(normalized),
            operation="worker dispatch",
        )
        if denial is not None:
            return denial
        limit = max(1, min(int(max_concurrency), len(normalized), 8))
        system_conf = getattr(getattr(owner, "config", None), "system", None)
        effective_timeout = timeout_s
        if effective_timeout is None:
            effective_timeout = getattr(system_conf, "worker_subtask_timeout", None)
        if effective_timeout is not None:
            try:
                effective_timeout = float(effective_timeout)
            except (TypeError, ValueError):
                effective_timeout = None
            if effective_timeout is not None and effective_timeout <= 0:
                effective_timeout = None
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
                    coro = owner.run_subagent_task(
                        task_description=task,
                        lease=inherited_lease,
                        agent_name=f"Worker-{index}",
                        skill_ids=inherited_skill_ids,
                    )
                    if effective_timeout is not None:
                        text, _ = await asyncio.wait_for(coro, timeout=effective_timeout)
                    else:
                        text, _ = await coro
                    return {"index": index, "task": task, "result": text}
                except asyncio.TimeoutError:
                    return {
                        "index": index,
                        "task": task,
                        "error": f"Timeout after {effective_timeout:.2f}s",
                    }
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
        (build_create_worker_tool(owner), "worker_control"),
        (build_dispatch_workers_tool(owner), "worker_control"),
    )
