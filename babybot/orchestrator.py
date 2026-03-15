"""Orchestrator built on lightweight kernel — single agent mode."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import Config
from .context import Tape, TapeStore
from .model_gateway import OpenAICompatibleGateway
from .resource import ResourceManager

if TYPE_CHECKING:
    from .heartbeat import Heartbeat

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = (
    "请用中文将以下对话历史浓缩为一段简短摘要（不超过200字），"
    "保留关键事实、用户意图和已完成的操作，省略冗余细节：\n\n"
)


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
        self.tape_store = TapeStore(
            db_path=self.config.home_dir / "context.db",
            max_chats=self.config.system.context_max_chats,
        )
        self._initialized = False
        self._collected_media: list[str] = []

    def _get_direct_prompt(self) -> str:
        return """你是高效助手。对任务直接回答：
- 简洁准确
- 不虚构工具执行结果
- 如需外部信息必须明确指出并建议调用工具"""

    async def _answer_direct(
        self, user_input: str, tape: Tape | None = None,
        heartbeat: Heartbeat | None = None,
    ) -> str:
        logger.info("_answer_direct calling run_subagent_task...")
        text, media = await self.resource_manager.run_subagent_task(
            task_description=user_input,
            lease={},
            agent_name="DirectAssistant",
            tape=tape,
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
        self, user_input: str, chat_key: str = "",
        heartbeat: Heartbeat | None = None,
    ) -> TaskResponse:
        if not self._initialized:
            logger.info("Initializing resource manager...")
            await self.resource_manager.initialize_async()
            self._initialized = True
            logger.info("Resource manager initialized")

        self._collected_media.clear()
        if heartbeat is not None:
            heartbeat.beat()

        # --- Tape context ---
        tape: Tape | None = None
        if chat_key:
            tape = self.tape_store.get_or_create(chat_key)
            # Ensure bootstrap anchor exists
            if tape.last_anchor() is None:
                anchor = tape.append("anchor", {"name": "session/start", "state": {}})
                self.tape_store.save_entry(chat_key, anchor)
            # Append user message
            user_entry = tape.append("message", {"role": "user", "content": user_input})
            self.tape_store.save_entry(chat_key, user_entry)

        try:
            text = await self._answer_direct(user_input, tape=tape, heartbeat=heartbeat)
            if heartbeat is not None:
                heartbeat.beat()

            # Append assistant response
            if tape and chat_key:
                asst_entry = tape.append("message", {"role": "assistant", "content": text})
                self.tape_store.save_entry(chat_key, asst_entry)
                # Fire-and-forget async handoff check
                asyncio.create_task(self._maybe_handoff(tape, chat_key))

            return TaskResponse(text=text, media_paths=list(self._collected_media))
        except Exception as exc:
            logger.exception("Error processing task")
            return TaskResponse(text=f"处理任务时出错：{exc}")

    async def _maybe_handoff(self, tape: Tape, chat_key: str) -> None:
        """Check if entries since last anchor exceed threshold; if so, create a new anchor."""
        try:
            threshold = self.config.system.context_compact_threshold
            if tape.total_tokens_since_anchor() <= threshold:
                return

            # Collect entries to summarize
            old_entries = tape.entries_since_anchor()
            if not old_entries:
                return

            # Build text to summarize
            lines: list[str] = []
            for e in old_entries:
                if e.kind == "message":
                    role = e.payload.get("role", "?")
                    content = e.payload.get("content", "")
                    lines.append(f"{role}: {content}")
            if not lines:
                return

            history_text = "\n".join(lines)
            summary = await self.gateway.complete(
                _SUMMARIZE_PROMPT, history_text
            )

            source_ids = [e.entry_id for e in old_entries]
            anchor = tape.append("anchor", {
                "name": f"compact/{tape.turn_count()}",
                "state": {
                    "summary": summary.strip(),
                    "phase": "continuation",
                    "source_ids": source_ids,
                    "turn_count": tape.turn_count(),
                },
            })
            self.tape_store.save_entry(chat_key, anchor)
            tape.compact_entries()
            logger.info(
                "Handoff created anchor chat_key=%s entry_id=%d summarized=%d entries",
                chat_key, anchor.entry_id, len(source_ids),
            )
        except Exception:
            logger.exception("Error in _maybe_handoff for chat_key=%s", chat_key)

    def reset(self) -> None:
        self.resource_manager.reset()
        self.tape_store.clear()
        self._initialized = False

    def get_status(self) -> dict[str, Any]:
        return {
            "resource_manager": "initialized",
            "available_tools": len(self.resource_manager.get_available_tools()),
            "resources": self.resource_manager.search_resources(),
        }
