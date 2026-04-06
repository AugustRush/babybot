"""Interactive session support for orchestrator chat flows."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from .interactive_sessions.types import InteractiveOutputCallback, InteractiveRequest
from .task_contract import build_task_contract


class OrchestratorInteractiveSessionSupport:
    """Handles @session commands and active interactive session message routing."""

    def __init__(
        self,
        *,
        session_manager: Any,
        invoke_callback: Callable[[Callable[[Any], Awaitable[None] | None] | None, Any], Awaitable[None]]
        | None = None,
        prepare_tape: Callable[..., Any] | None = None,
        tape_store: Any = None,
        response_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._manager = session_manager
        self._invoke_callback = invoke_callback
        self._prepare_tape = prepare_tape
        self._tape_store = tape_store
        self._response_factory = response_factory or (lambda **kwargs: SimpleNamespace(**kwargs))

    @staticmethod
    def parse_command(user_input: str) -> dict[str, str] | None:
        text = (user_input or "").strip()
        if not text.lower().startswith("@session"):
            return None
        parts = text.split()
        action = parts[1].lower() if len(parts) >= 2 else "status"
        backend_name = parts[2].lower() if len(parts) >= 3 else ""
        return {"action": action, "backend_name": backend_name}

    async def handle_command(
        self,
        *,
        chat_key: str,
        control: dict[str, str],
    ) -> Any:
        action = control.get("action", "")
        backend_name = control.get("backend_name", "")

        if action == "start":
            if not backend_name:
                return self._response_factory(text="用法：@session start <backend>")
            session = await self._manager.start(
                chat_key=chat_key, backend_name=backend_name
            )
            label = session.backend_name.capitalize()
            return self._response_factory(
                text=(
                    f"{label} 会话已启动（session_id={session.session_id}）。"
                    "后续消息将直接发送到该交互会话。"
                )
            )
        if action == "stop":
            stopped = await self._manager.stop(chat_key, reason="user_stop")
            return self._response_factory(
                text="交互会话已关闭。" if stopped else "当前没有活动中的交互会话。"
            )
        if action == "status":
            status = self._manager.status(chat_key)
            if status is None:
                return self._response_factory(text="当前没有活动中的交互会话。")
            backend_bits: list[str] = []
            backend_status = dict(status.backend_status or {})
            mode = str(backend_status.get("mode", "") or status.mode).strip()
            if mode:
                backend_bits.append(f"mode={mode}")
            pid = backend_status.get("pid", status.process_pid)
            if pid:
                backend_bits.append(f"pid={pid}")
            alive = backend_status.get("alive")
            if alive is not None:
                backend_bits.append(f"alive={bool(alive)}")
            return self._response_factory(
                text=(
                    f"当前交互会话：{status.backend_name} "
                    f"(session_id={status.session_id}"
                    + (f", {', '.join(backend_bits)}" if backend_bits else "")
                    + ")"
                )
            )
        return self._response_factory(text="支持的命令：@session start <backend> / status / stop")

    async def handle_message(
        self,
        *,
        chat_key: str,
        user_input: str,
        media_paths: list[str] | None = None,
        heartbeat: Any = None,
        runtime_event_callback: Callable[[Any], Awaitable[None] | None] | None = None,
        interactive_output_callback: InteractiveOutputCallback | None = None,
    ) -> Any | None:
        task_contract = build_task_contract(
            user_input=user_input,
            chat_key=chat_key,
        )
        request = InteractiveRequest(
            text=user_input,
            media_paths=tuple(media_paths or ()),
            contract_mode=task_contract.mode,
        )
        await self._emit_runtime_event(
            runtime_event_callback,
            {
                "event": "running",
                "task_id": "interactive_session",
                "flow_id": f"interactive:{chat_key}",
                "payload": {
                    "stage": "interactive_session",
                    "state": "running",
                    "message": "交互会话处理中",
                },
            },
        )
        try:
            if heartbeat is not None:
                async with heartbeat.keep_alive():
                    reply = await self._manager.send(
                        chat_key,
                        request,
                        output_event_callback=interactive_output_callback,
                    )
            else:
                reply = await self._manager.send(
                    chat_key,
                    request,
                    output_event_callback=interactive_output_callback,
                )
        except RuntimeError:
            with contextlib.suppress(Exception):
                await self._manager.stop(chat_key, reason="backend_failed")
            return None
        if reply.expired:
            return None
        if self._prepare_tape is not None and self._tape_store is not None:
            tape = self._prepare_tape(
                chat_key=chat_key,
                user_input=user_input,
                media_paths=media_paths,
            )
            if tape is not None:
                assistant_entry = tape.append(
                    "message", {"role": "assistant", "content": reply.text}
                )
                self._tape_store.save_entries(chat_key, [assistant_entry])
        await self._emit_runtime_event(
            runtime_event_callback,
            {
                "event": "completed",
                "task_id": "interactive_session",
                "flow_id": f"interactive:{chat_key}",
                "payload": {
                    "stage": "interactive_session",
                    "state": "completed",
                    "message": "交互会话完成",
                },
            },
        )
        return self._response_factory(
            text=reply.text,
            media_paths=list(reply.media_paths or []),
        )

    async def _emit_runtime_event(
        self,
        callback: Callable[[Any], Awaitable[None] | None] | None,
        payload: dict[str, Any],
    ) -> None:
        if self._invoke_callback is not None:
            await self._invoke_callback(callback, payload)
            return
        if callback is None:
            return
        maybe = callback(payload)
        if hasattr(maybe, "__await__"):
            await maybe


__all__ = ["OrchestratorInteractiveSessionSupport"]
