import asyncio
import logging
from .config import Config
from .orchestrator import OrchestratorAgent
from .channels import ChannelManager


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
                print(
                    f"\nAvailable Tools: {status.get('available_tools', 0)}\n"
                )
                continue

            try:
                print("\n正在处理...")
                response = loop.run_until_complete(
                    asyncio.wait_for(
                        orchestrator.process_task(user_input),
                        timeout=float(config.system.timeout),
                    )
                )
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
