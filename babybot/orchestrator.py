"""Orchestrator built on lightweight kernel — single agent mode."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import Config
from .model_gateway import OpenAICompatibleGateway
from .resource import ResourceManager

if TYPE_CHECKING:
    from .heartbeat import Heartbeat

logger = logging.getLogger(__name__)


@dataclass
class TaskResponse:
    """Structured response from process_task with text and optional media."""

    text: str = ""
    media_paths: list[str] = field(default_factory=list)


class OrchestratorAgent:
    """Orchestrator — all tasks go through a single agent that can self-escalate."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.model.validate()
        self.resource_manager = ResourceManager(self.config)
        self.gateway = OpenAICompatibleGateway(self.config)
        self._initialized = False
        self._collected_media: list[str] = []

    def _get_direct_prompt(self) -> str:
        return """你是高效助手。对任务直接回答：
- 简洁准确
- 不虚构工具执行结果
- 如需外部信息必须明确指出并建议调用工具"""

    async def _answer_direct(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> str:
        logger.info("_answer_direct calling run_subagent_task...")
        text, media = await self.resource_manager.run_subagent_task(
            task_description=user_input,
            lease={},
            agent_name="DirectAssistant",
            heartbeat=heartbeat,
        )
        logger.info("_answer_direct subagent done text_len=%d media=%d", len(text or ""), len(media or []))
        if media:
            self._collected_media.extend(media)
        if text.strip():
            return text
        return await self.gateway.complete(
            self._get_direct_prompt(), user_input, heartbeat=heartbeat
        )

    async def process_task(
        self, user_input: str, heartbeat: Heartbeat | None = None
    ) -> TaskResponse:
        if not self._initialized:
            logger.info("Initializing resource manager...")
            await self.resource_manager.initialize_async()
            self._initialized = True
            logger.info("Resource manager initialized")

        self._collected_media.clear()
        if heartbeat is not None:
            heartbeat.beat()

        try:
            text = await self._answer_direct(user_input, heartbeat)
            if heartbeat is not None:
                heartbeat.beat()
            return TaskResponse(text=text, media_paths=list(self._collected_media))
        except Exception as exc:
            logger.exception("Error processing task")
            return TaskResponse(text=f"处理任务时出错：{exc}")

    def reset(self) -> None:
        self.resource_manager.reset()
        self._initialized = False

    def get_status(self) -> dict[str, Any]:
        return {
            "resource_manager": "initialized",
            "available_tools": len(self.resource_manager.get_available_tools()),
            "resources": self.resource_manager.search_resources(),
        }
