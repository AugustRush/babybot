"""Reference single-agent executor built on model/tools/skills ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .model import ModelMessage, ModelProvider, ModelRequest, ModelResponse, ModelToolCall
from .skills import SkillPack, merge_leases, merge_prompts
from .tools import ToolContext, ToolRegistry
from .types import ExecutionContext, TaskContract, TaskResult, ToolLease


@dataclass
class ExecutorPolicy:
    """Policy for one single-agent execution loop."""

    max_steps: int = 8


@dataclass
class SingleAgentExecutor:
    """ExecutorPort implementation.

    It runs a compact agent loop:
    model -> tool_calls -> tool_results -> model ... -> final text
    """

    model: ModelProvider
    tools: ToolRegistry
    skill_resolver: Callable[[TaskContract, ExecutionContext], SkillPack | Iterable[SkillPack] | None] | None = None
    policy: ExecutorPolicy = field(default_factory=ExecutorPolicy)

    async def execute(self, task: TaskContract, context: ExecutionContext) -> TaskResult:
        skills = self._resolve_skills(task, context)
        base_lease = task.lease
        for skill in skills:
            base_lease = merge_leases(base_lease, skill.tool_lease)

        system_prompt = merge_prompts(skills)
        messages: list[ModelMessage] = []
        if system_prompt:
            messages.append(ModelMessage(role="system", content=system_prompt))
        messages.append(ModelMessage(role="user", content=task.description))

        tool_context = ToolContext(session_id=context.session_id, state=context.state)
        heartbeat = context.state.get("heartbeat")
        for step in range(1, max(1, self.policy.max_steps) + 1):
            if heartbeat is not None:
                heartbeat.beat()
            context.emit("executor.step", task_id=task.task_id, step=step)
            response = await self.model.generate(
                ModelRequest(
                    messages=tuple(messages),
                    tools=self.tools.tool_schemas(base_lease),
                    metadata={"task_id": task.task_id, "step": step},
                ),
                context,
            )

            if response.tool_calls:
                messages.append(
                    ModelMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )
                for tool_call in response.tool_calls:
                    registered = self.tools.get(tool_call.name)
                    if not registered or not self._tool_allowed(registered.group, tool_call.name, base_lease):
                        tool_output = f"Tool unavailable: {tool_call.name}"
                    else:
                        result = await registered.tool.invoke(tool_call.arguments, tool_context)
                        tool_output = result.content if result.ok else f"Tool error: {result.error}"
                    messages.append(
                        ModelMessage(
                            role="tool",
                            name=tool_call.name,
                            content=tool_output,
                            tool_call_id=tool_call.call_id,
                        )
                    )
                continue

            text = response.text.strip()
            if text:
                return TaskResult(task_id=task.task_id, status="succeeded", output=text)

        return TaskResult(
            task_id=task.task_id,
            status="failed",
            error=f"No terminal answer within {self.policy.max_steps} steps.",
            metadata={"history": [self._dump_message(message) for message in messages]},
        )

    def _resolve_skills(self, task: TaskContract, context: ExecutionContext) -> list[SkillPack]:
        resolver = self.skill_resolver
        if resolver is None:
            return []
        result = resolver(task, context)
        if not result:
            return []
        if isinstance(result, SkillPack):
            return [result]
        return [item for item in result if isinstance(item, SkillPack)]

    @staticmethod
    def _tool_allowed(group: str, name: str, lease: ToolLease) -> bool:
        include_groups = set(lease.include_groups)
        include_tools = set(lease.include_tools)
        exclude_tools = set(lease.exclude_tools)
        if name in exclude_tools:
            return False
        if include_tools and name not in include_tools:
            return False
        if include_groups and group not in include_groups:
            return False
        return True

    @staticmethod
    def _dump_message(message: ModelMessage) -> dict[str, str]:
        payload = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.call_id, "name": tc.name} for tc in message.tool_calls
            ]
        return payload


class EchoModelProvider:
    """Tiny debug model provider for local wiring tests."""

    async def generate(self, request: ModelRequest, context: ExecutionContext) -> ModelResponse:
        last = request.messages[-1] if request.messages else ModelMessage(role="assistant", content="")
        if last.role == "tool":
            return ModelResponse(text=f"Observed: {last.content}")
        if request.tools:
            fn = request.tools[0]["function"]["name"]
            return ModelResponse(
                text="",
                tool_calls=(
                    ModelToolCall(
                        call_id="c1",
                        name=fn,
                        arguments={"input": last.content},
                    ),
                ),
            )
        return ModelResponse(text=last.content)
