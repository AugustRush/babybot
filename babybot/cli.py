import asyncio
import logging
from .config import Config
from .orchestrator import OrchestratorAgent


def run():
    """Run the multi-agent orchestrator CLI."""
    print("Initializing Orchestrator...")

    try:
        config = Config()
        if not config.system.console_output:
            logging.getLogger("agentscope").setLevel(logging.WARNING)
        orchestrator = OrchestratorAgent(config)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set your API key:")
        print("  1. Create a .env file with: OPENAI_API_KEY=your_key")
        print("  2. Or export it: export OPENAI_API_KEY=your_key")
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
                print(f"\n📋 最终结果:\n{response}\n")
            except asyncio.TimeoutError:
                print(
                    f"\n⏰ 任务执行超时（{config.system.timeout} 秒），请重试或简化任务。\n"
                )
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    finally:
        loop.close()
