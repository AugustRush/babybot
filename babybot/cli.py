import asyncio
import argparse
import logging
from .config import Config
from .orchestrator import OrchestratorAgent
from .channels import ChannelManager


def _setup_logging(channel_mode: bool, console_output: bool) -> None:
    level = logging.INFO if (channel_mode or console_output) else logging.WARNING
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)
    logging.getLogger("babybot").setLevel(level)


def run():
    """Run the multi-agent orchestrator CLI."""
    parser = argparse.ArgumentParser(description="BabyBot")
    parser.add_argument(
        "--channel",
        default="cli",
        help="Run mode: 'cli' for interactive terminal, or channel name (e.g. 'feishu')",
    )
    args = parser.parse_args()

    print("Initializing Orchestrator...")

    try:
        config = Config()
        _setup_logging(channel_mode=(args.channel != "cli"), console_output=config.system.console_output)
        if config.is_bootstrapped:
            print(f"\n已初始化配置文件：{config.config_file}")
            print(f"工作目录：{config.workspace_dir}")
            print("请先编辑配置文件后再运行 babybot。")
            return
        if config.system.tracing_endpoint:
            print(
                "Warning: tracing_endpoint is configured but tracing integration "
                "is not enabled in the lightweight kernel runtime."
            )
        if not config.system.console_output:
            logging.getLogger("openai").setLevel(logging.WARNING)
        orchestrator = OrchestratorAgent(config)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set your API key:")
        print("  1. Create a .env file with: OPENAI_API_KEY=your_key")
        print("  2. Or export it: export OPENAI_API_KEY=your_key")
        return

    if args.channel != "cli":
        # Channel mode — use ChannelManager
        manager = ChannelManager(config, orchestrator)
        if args.channel != "all":
            # If a specific channel name is given, check it was enabled
            if args.channel not in manager.channels:
                ch_config = config.get_channel_config(args.channel)
                if ch_config is None:
                    print(f"Unknown channel: {args.channel}")
                else:
                    print(
                        f"Channel '{args.channel}' is disabled. "
                        f"Set channels.{args.channel}.enabled=true in config."
                    )
                return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(manager.start_all())
            # Keep the process alive while channels run
            loop.run_forever()
        except KeyboardInterrupt:
            print("\nStopping channels...")
        finally:
            loop.run_until_complete(manager.stop_all())
            loop.close()
        return

    print("\n" + "=" * 60)
    print("🤖 BabyBot 多 Agent 协同系统")
    print("=" * 60)
    print("命令：status (状态) | reset (重置) | quit (退出)")
    print("=" * 60 + "\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                user_input = input("📝 任务：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "退出"]:
                print("\n👋 再见！")
                break

            if user_input.lower() == "reset":
                orchestrator.reset()
                print("\n✅ 已重置所有状态。\n")
                continue

            if user_input.lower() == "status":
                status = orchestrator.get_status()
                scheduler = status.get("scheduler", {})
                scheduler_status = scheduler.get("status", {})
                running = sum(1 for v in scheduler_status.values() if v == "running")
                print(
                    f"\n🤖 Available Tools: {status.get('available_tools', 0)} | "
                    f"Running Tasks: {running}\n"
                )
                continue

            try:
                print("\n⏳ 正在分析任务并创建动态 Agent...")
                response = loop.run_until_complete(
                    asyncio.wait_for(
                        orchestrator.process_task(user_input),
                        timeout=float(config.system.timeout),
                    )
                )
                print(f"\n📋 最终结果:\n{response.text}\n")
                if response.media_paths:
                    print(f"📎 生成的文件: {', '.join(response.media_paths)}\n")
            except asyncio.TimeoutError:
                print(
                    f"\n⏰ 任务执行超时（{config.system.timeout} 秒），请重试或简化任务。\n"
                )
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    finally:
        loop.close()
