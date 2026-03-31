import asyncio
import inspect
import logging
from .config import Config
from .orchestrator import OrchestratorAgent
from .channels import ChannelManager

_CLI_CHAT_KEY = "cli:local"


def _setup_logging(console_output: bool) -> None:
    level = logging.INFO if console_output else logging.WARNING
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)
    logging.getLogger("babybot").setLevel(level)


def _init_orchestrator(*, gateway_mode: bool = False) -> tuple[Config, OrchestratorAgent]:
    """Shared initialisation for both CLI and gateway entry points."""
    config = Config()
    _setup_logging(console_output=gateway_mode or config.system.console_output)
    if config.is_bootstrapped:
        raise SystemExit(
            f"\n已初始化配置文件：{config.config_file}\n"
            f"工作目录：{config.workspace_dir}\n"
            "请先编辑配置文件后再运行。"
        )
    if not config.system.console_output:
        logging.getLogger("openai").setLevel(logging.WARNING)
    orchestrator = OrchestratorAgent(config)
    return config, orchestrator


def run_gateway():
    """Start the channel gateway — all enabled channels run automatically."""
    print("Initializing Gateway...")
    try:
        config, orchestrator = _init_orchestrator(gateway_mode=True)
    except ValueError as e:
        print(f"Error: {e}")
        return

    manager = ChannelManager(config, orchestrator)
    if not manager.channels:
        print("No channels enabled. Enable at least one channel in config.json.")
        return

    print(f"Starting channels: {', '.join(manager.channels)}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(manager.start_all())
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nStopping channels...")
    finally:
        loop.run_until_complete(manager.stop_all())
        loop.close()


def run():
    """Run the interactive CLI."""
    print("Initializing Orchestrator...")
    try:
        config, orchestrator = _init_orchestrator()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set your API key:")
        print("  1. Create a .env file with: OPENAI_API_KEY=your_key")
        print("  2. Or export it: export OPENAI_API_KEY=your_key")
        return

    print("\n" + "=" * 60)
    print("BabyBot")
    print("=" * 60)
    print("命令：status (状态) | reset (重置) | quit (退出)")
    print("=" * 60 + "\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                user_input = input("任务：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "退出"]:
                print("\n再见！")
                break

            if user_input.lower() == "reset":
                orchestrator.reset()
                print("\n已重置所有状态。\n")
                continue

            if user_input.lower() == "status":
                status = orchestrator.get_status()
                policy_telemetry = dict(status.get("policy_telemetry") or {})
                interactive = dict(status.get("interactive_sessions") or {})
                active_count = int(interactive.get("active_count") or 0)
                chat_keys = ", ".join(interactive.get("chat_keys") or [])
                skip_breakdown = dict(policy_telemetry.get("skip_breakdown") or {})
                skip_breakdown_text = ", ".join(
                    f"{str(reason).strip() or 'unknown'}:{int(count or 0)}"
                    for reason, count in sorted(
                        (
                            (reason, count)
                            for reason, count in skip_breakdown.items()
                        ),
                        key=lambda item: (-int(item[1] or 0), str(item[0])),
                    )
                )
                session_lines: list[str] = []
                for item in interactive.get("sessions") or []:
                    if not isinstance(item, dict):
                        continue
                    backend_name = str(item.get("backend_name", "") or "").strip()
                    chat_key = str(item.get("chat_key", "") or "").strip()
                    backend_status = dict(item.get("backend_status") or {})
                    mode = str(backend_status.get("mode", "") or "").strip()
                    pid = backend_status.get("pid")
                    session_lines.append(
                        f"{chat_key or '-'} [{backend_name or '-'}] mode={mode or '-'} pid={pid or '-'}"
                    )
                print(
                    f"\nAvailable Tools: {status.get('available_tools', 0)}\n"
                    + (
                        f"Policy Telemetry Runs: {int(policy_telemetry.get('runs', 0) or 0)}\n"
                        f"Avg Execution Elapsed Ms: {float(policy_telemetry.get('avg_execution_elapsed_ms', 0.0) or 0.0):.2f}\n"
                        f"Avg Tool Call Count: {float(policy_telemetry.get('avg_tool_call_count', 0.0) or 0.0):.2f}\n"
                        f"Tool Failure Rate: {float(policy_telemetry.get('tool_failure_rate', 0.0) or 0.0):.2f}\n"
                        f"Loop Guard Block Rate: {float(policy_telemetry.get('loop_guard_block_rate', 0.0) or 0.0):.2f}\n"
                        f"Max Step Exhausted Rate: {float(policy_telemetry.get('max_step_exhausted_rate', 0.0) or 0.0):.2f}\n"
                        f"Fallback Rate: {float(policy_telemetry.get('fallback_rate', 0.0) or 0.0):.2f}\n"
                        f"Skipped Rate: {float(policy_telemetry.get('skipped_rate', 0.0) or 0.0):.2f}\n"
                        f"Model Route Rate: {float(policy_telemetry.get('model_route_rate', 0.0) or 0.0):.2f}\n"
                        + (
                            f"Skip Breakdown: {skip_breakdown_text}\n"
                            if skip_breakdown_text
                            else ""
                        )
                        if policy_telemetry
                        else ""
                    )
                    + f"Interactive Sessions: {active_count}\n"
                    + f"Interactive Chats: {chat_keys or '-'}\n"
                    + (
                        "Interactive Details:\n" + "\n".join(session_lines) + "\n"
                        if session_lines
                        else ""
                    )
                )
                continue

            try:
                print("\n正在处理...")
                interactive_stream_seen = False
                interactive_stream_final_text = ""
                interactive_stream_printed = False

                async def _interactive_output_callback(event: object) -> None:
                    nonlocal interactive_stream_seen
                    nonlocal interactive_stream_final_text
                    nonlocal interactive_stream_printed
                    event_name = str(getattr(event, "event", "") or "")
                    text = str(getattr(event, "text", "") or "")
                    delta = str(getattr(event, "delta", "") or "")
                    if event_name == "message_start":
                        interactive_stream_seen = True
                        return
                    if event_name == "message_delta" and delta:
                        interactive_stream_seen = True
                        interactive_stream_printed = True
                        interactive_stream_final_text = text or interactive_stream_final_text
                        print(delta, end="", flush=True)
                        return
                    if event_name == "message_complete":
                        interactive_stream_seen = True
                        interactive_stream_final_text = text or interactive_stream_final_text
                        if interactive_stream_printed:
                            print()

                process_kwargs = {
                    "chat_key": _CLI_CHAT_KEY,
                }
                try:
                    supports_interactive_output = (
                        "interactive_output_callback"
                        in inspect.signature(orchestrator.process_task).parameters
                    )
                except (TypeError, ValueError):
                    supports_interactive_output = False
                if supports_interactive_output:
                    process_kwargs["interactive_output_callback"] = _interactive_output_callback
                response = loop.run_until_complete(
                    asyncio.wait_for(
                        orchestrator.process_task(
                            user_input,
                            **process_kwargs,
                        ),
                        timeout=float(config.system.timeout),
                    )
                )
                should_echo_final = not (
                    interactive_stream_seen
                    and interactive_stream_printed
                    and response.text.strip()
                    and response.text.strip() == interactive_stream_final_text.strip()
                )
                if should_echo_final:
                    print(f"\n结果:\n{response.text}\n")
                if response.media_paths:
                    print(f"生成的文件: {', '.join(response.media_paths)}\n")
            except asyncio.TimeoutError:
                print(
                    f"\n任务执行超时（{config.system.timeout} 秒），请重试或简化任务。\n"
                )
            except Exception as e:
                print(f"\nError: {e}\n")
    finally:
        loop.close()
