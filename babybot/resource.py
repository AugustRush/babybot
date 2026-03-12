"""Resource manager based on AgentScope Toolkit."""

import asyncio
import os
from pathlib import Path
from typing import Any

from agentscope.tool import Toolkit, ToolResponse
from agentscope.mcp import HttpStatelessClient, StdIOStatefulClient

from .config import Config


class ResourceManager:
    """Centralized resource manager using Toolkit.

    This is a singleton that manages:
    - Tool functions and tool groups
    - MCP clients and their tools
    - Agent skills

    Configuration is loaded from config.json
    """

    _instance: "ResourceManager | None" = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ResourceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Config | None = None):
        if ResourceManager._initialized:
            return

        self.config = config or Config()
        self.toolkit = Toolkit()
        self.mcp_clients: dict[str, Any] = {}

        # Register create_worker tool
        self.toolkit.register_tool_function(
            self.create_worker_tool(),
            group_name="basic",
        )

        # Load configuration from config.json
        self._load_config()

        ResourceManager._initialized = True

    async def initialize_async(self) -> None:
        """Asynchronously initialize MCP servers."""
        mcp_servers = {
            k: v
            for k, v in self.config.get_mcp_servers().items()
            if not k.startswith("_")
        }
        if mcp_servers:
            await self._register_mcp_servers(mcp_servers)

    def _load_config(self) -> None:
        """Load configuration from Config object."""
        # 1. Setup tool groups first (filter out comment keys)
        tool_groups = {
            k: v
            for k, v in self.config.get_tool_groups().items()
            if not k.startswith("_")
        }
        self._setup_tool_groups(tool_groups)

        # 2. Register agent skills (filter out comment keys)
        agent_skills = {
            k: v
            for k, v in self.config.get_agent_skills().items()
            if not k.startswith("_")
        }
        if agent_skills:
            self._register_agent_skills(agent_skills)

        # 3. Register custom tools (filter out comment keys)
        custom_tools = {
            k: v
            for k, v in self.config.get_custom_tools().items()
            if not k.startswith("_")
        }
        if custom_tools:
            self._register_custom_tools(custom_tools)

    def _setup_tool_groups(self, tool_groups_conf: dict) -> None:
        """Setup tool groups from configuration."""
        # Start with empty default groups
        all_groups = {
            "search": {
                "description": "Search and information retrieval tools",
                "notes": "Use these tools for web search and information gathering.",
                "active": False,
            },
            "code": {
                "description": "Code execution and analysis tools",
                "notes": "Use these tools for code execution and debugging.",
                "active": False,
            },
            "analysis": {
                "description": "Data analysis and visualization tools",
                "notes": "Use these tools for data analysis tasks.",
                "active": False,
            },
            "browser": {
                "description": "Browser automation tools",
                "notes": "Use Playwright for web interaction and automation.",
                "active": False,
            },
        }

        # Merge with config
        for group_name, user_conf in tool_groups_conf.items():
            if group_name in all_groups:
                # Update existing group
                all_groups[group_name]["active"] = user_conf.get(
                    "active", all_groups[group_name]["active"]
                )
                all_groups[group_name]["description"] = user_conf.get(
                    "description", all_groups[group_name]["description"]
                )
                all_groups[group_name]["notes"] = user_conf.get(
                    "notes", all_groups[group_name]["notes"]
                )
            else:
                # Add new group from config
                all_groups[group_name] = {
                    "description": user_conf.get("description", f"{group_name} tools"),
                    "notes": user_conf.get("notes", ""),
                    "active": user_conf.get("active", False),
                }

        # Create all tool groups
        for group_name, group_conf in all_groups.items():
            self.toolkit.create_tool_group(
                group_name=group_name,
                description=group_conf["description"],
                active=group_conf["active"],
                notes=group_conf["notes"],
            )
            print(
                f"✓ Tool group '{group_name}' created (active={group_conf['active']})"
            )

    async def _register_mcp_servers(self, mcp_servers: dict[str, dict]) -> None:
        """Register MCP servers from configuration."""
        for name, server_conf in mcp_servers.items():
            try:
                mcp_type = server_conf.get("type", "http")

                if mcp_type == "stdio":
                    # StdIO MCP server (e.g., Playwright)
                    await self.register_mcp_stdio(
                        name=name,
                        command=server_conf["command"],
                        args=server_conf.get("args", []),
                        group_name=server_conf.get("group_name", name),
                    )
                else:
                    # HTTP MCP server
                    await self.register_mcp(
                        name=name,
                        url=server_conf["url"],
                        transport=server_conf.get("transport", "streamable_http"),
                        group_name=server_conf.get("group_name", name),
                    )

                # Activate if specified
                if server_conf.get("active", False):
                    group_name = server_conf.get("group_name", name)
                    self.activate_groups([group_name])
                    print(
                        f"✓ MCP server '{name}' registered and tool group '{group_name}' activated"
                    )
            except Exception as e:
                print(f"Warning: Failed to register MCP server {name}: {e}")

    def _register_custom_tools(self, custom_tools: dict[str, dict]) -> None:
        """Register custom tools from configuration."""
        for name, tool_conf in custom_tools.items():
            try:
                # Load tool function from module
                module_name = tool_conf["module"]
                func_name = tool_conf.get("function", name)

                # Import module and get tool function
                import importlib

                module = importlib.import_module(module_name)
                tool_func = getattr(module, func_name, None)

                if tool_func:
                    # Process preset kwargs (substitute env variables)
                    preset_kwargs = tool_conf.get("preset_kwargs", {})
                    for key, value in preset_kwargs.items():
                        if (
                            isinstance(value, str)
                            and value.startswith("${")
                            and value.endswith("}")
                        ):
                            env_var = value[2:-1]
                            preset_kwargs[key] = os.getenv(env_var, value)

                    self.register_tool(
                        func=tool_func,
                        group_name=tool_conf.get("group_name", "basic"),
                        preset_kwargs=preset_kwargs,
                    )
            except Exception as e:
                print(f"Warning: Failed to register custom tool {name}: {e}")

    def _register_agent_skills(self, agent_skills: dict[str, dict]) -> None:
        """Register agent skills from configuration."""
        for name, skill_conf in agent_skills.items():
            try:
                self.toolkit.register_agent_skill(
                    skill_conf["directory"],
                )
            except Exception as e:
                print(f"Warning: Failed to register agent skill {name}: {e}")

    def register_tool(
        self,
        func: Any,
        group_name: str = "basic",
        preset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool function.

        Args:
            func: The tool function to register.
            group_name: Group name to organize the tool.
            preset_kwargs: Optional preset keyword arguments.
        """
        self.toolkit.register_tool_function(
            func,
            group_name=group_name,
            preset_kwargs=preset_kwargs or {},
        )

    async def register_mcp(
        self,
        name: str,
        url: str,
        transport: str = "streamable_http",
        group_name: str = "basic",
    ) -> None:
        """Register an HTTP MCP server.

        Args:
            name: The name to identify the MCP server.
            url: The MCP server URL.
            transport: Transport protocol (streamable_http or sse).
            group_name: Group name for the MCP tools.
        """
        client = HttpStatelessClient(
            name=name,
            transport=transport,  # type: ignore
            url=url,
        )

        self.mcp_clients[name] = client

        # Register all tools from MCP server
        await self.toolkit.register_mcp_client(
            client,
            group_name=group_name,
        )

    async def register_mcp_stdio(
        self,
        name: str,
        command: str,
        args: list[str],
        group_name: str = "basic",
    ) -> None:
        """Register a StdIO MCP server."""
        try:
            client = StdIOStatefulClient(
                name=name,
                command=command,
                args=args,
            )

            self.mcp_clients[name] = client

            # Connect to the server with timeout
            print(f"Connecting to MCP server '{name}'...")
            await asyncio.wait_for(client.connect(), timeout=30)
            print(f"✓ MCP server '{name}' connected")

            # Register all tools from MCP server
            await self.toolkit.register_mcp_client(
                client,
                group_name=group_name,
            )
        except asyncio.TimeoutError:
            print(f"Warning: Timeout connecting to MCP server '{name}'")
        except Exception as e:
            print(f"Warning: Failed to connect MCP server '{name}': {e}")

    def activate_groups(self, group_names: list[str]) -> str:
        """Activate tool groups.

        Args:
            group_names: List of group names to activate.

        Returns:
            Notes of activated groups.
        """
        self.toolkit.update_tool_groups(group_names=group_names, active=True)
        return self.toolkit.get_activated_notes()

    def deactivate_groups(self, group_names: list[str]) -> None:
        """Deactivate tool groups.

        Args:
            group_names: List of group names to deactivate.
        """
        self.toolkit.update_tool_groups(group_names=group_names, active=False)

    def get_available_tools(self) -> list[dict]:
        """Get JSON schemas of available tools.

        Returns:
            List of tool JSON schemas.
        """
        return self.toolkit.get_json_schemas()

    def get_tool_prompt(self) -> str:
        """Get prompt for available tools.

        Returns:
            Formatted tool prompt.
        """
        return self.toolkit.get_activated_notes()

    def create_worker_tool(self) -> Any:
        """Create the create_worker tool function.

        Returns:
            A tool function that creates workers with inherited resources.
        """

        async def create_worker(task_description: str) -> ToolResponse:
            """Create a specialized worker to complete a given task.

            Args:
                task_description: Description of the task to be completed.
            """
            # Import here to avoid circular dependency
            from .worker import WorkerAgent

            worker = WorkerAgent(
                config=self.config,
                toolkit=self.toolkit,
            )

            result = await worker.execute(task_description)
            return ToolResponse(content=[{"type": "text", "text": result}])

        return create_worker

    def reset(self) -> None:
        """Reset resource manager to default state."""
        self.toolkit.clear()
        self.mcp_clients.clear()
        ResourceManager._initialized = False
        self._load_config()

    @classmethod
    def get_instance(cls) -> "ResourceManager":
        """Get the singleton instance."""
        if cls._instance is None:
            raise RuntimeError("ResourceManager not initialized yet")
        return cls._instance
