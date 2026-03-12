"""Unified configuration management."""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "deepseek-ai/DeepSeek-V3.2"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "API key is required. Set in ~/.babybot/config.json "
                "or OPENAI_API_KEY environment variable."
            )


@dataclass
class SystemConfig:
    """System configuration."""

    console_output: bool = False
    enable_meta_tool: bool = True
    timeout: int = 60
    tracing_endpoint: str = ""


@dataclass
class FeishuConfig:
    """Feishu channel configuration."""

    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    group_policy: Literal["open", "mention"] = "mention"
    reply_mode: Literal["chat", "p2p"] = "chat"


class Config:
    """Unified configuration manager.

    Loads configuration from config.json with environment variable fallback.
    Priority: config.json > environment variables > defaults
    """

    def __init__(self, config_file: str | None = None):
        """Initialize configuration.

        Args:
            config_file: Path to config file.
                Defaults to `$BABYBOT_CONFIG` or `~/.babybot/config.json`.
        """
        self.home_dir = Path(
            os.getenv("BABYBOT_HOME", "~/.babybot")
        ).expanduser()
        self.workspace_dir = Path(
            os.getenv("BABYBOT_WORKSPACE", str(self.home_dir / "workspace"))
        ).expanduser()
        self.builtin_skills_dir = Path(__file__).resolve().parent.parent / "skills"
        self.workspace_skills_dir = self.workspace_dir / "skills"
        self.workspace_tools_dir = self.workspace_dir / "tools"

        if config_file:
            self.config_file = Path(config_file).expanduser()
        else:
            env_config = os.getenv("BABYBOT_CONFIG", "")
            self.config_file = (
                Path(env_config).expanduser()
                if env_config
                else self.home_dir / "config.json"
            )
        self.raw_config: dict[str, Any] = {}
        self.is_bootstrapped = False

        # Load configuration
        self._load_config()

        # Create model config
        model_conf = self.raw_config.get("model", {})
        self.model = ModelConfig(
            model_name=model_conf.get("model_name", "deepseek-ai/DeepSeek-V3.2"),
            api_key=model_conf.get("api_key", "") or os.getenv("OPENAI_API_KEY", ""),
            api_base=model_conf.get("api_base", "") or os.getenv("OPENAI_API_BASE", ""),
            temperature=model_conf.get("temperature", 0.7),
            max_tokens=model_conf.get("max_tokens", 2048),
        )

        # Create system config
        system_conf = self.raw_config.get("system", {})
        self.system = SystemConfig(
            console_output=system_conf.get("console_output", False),
            enable_meta_tool=system_conf.get("enable_meta_tool", True),
            timeout=system_conf.get("timeout", 60),
            tracing_endpoint=system_conf.get("tracing_endpoint", ""),
        )

        # Resource configuration
        self.resources = self.raw_config.get("resources", {})

        # Channel configuration
        channels_conf = self.raw_config.get("channels", {})
        feishu_conf = channels_conf.get("feishu", {})
        self.feishu = FeishuConfig(
            enabled=feishu_conf.get("enabled", False),
            app_id=feishu_conf.get("app_id", ""),
            app_secret=feishu_conf.get("app_secret", ""),
            encrypt_key=feishu_conf.get("encrypt_key", ""),
            verification_token=feishu_conf.get("verification_token", ""),
            group_policy=feishu_conf.get("group_policy", "mention"),
            reply_mode=feishu_conf.get("reply_mode", "chat"),
        )

    def _load_config(self) -> None:
        """Load configuration from file."""
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_skills_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_tools_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self._bootstrap_config_file()

        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.raw_config = json.load(f)
        else:
            # Use defaults
            self.raw_config = {}

    def _bootstrap_config_file(self) -> None:
        """Create initial config file from example on first run."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        example_candidates = [
            Path(__file__).resolve().parent.parent / "config.json.example",
            Path.cwd() / "config.json.example",
            self.home_dir / "config.json.example",
        ]
        for candidate in example_candidates:
            if candidate.exists():
                shutil.copyfile(candidate, self.config_file)
                self.is_bootstrapped = True
                return

        # Fallback minimal config if example is unavailable.
        fallback = {
            "model": {
                "model_name": "deepseek-ai/DeepSeek-V3.2",
                "api_key": "",
                "api_base": "",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "resources": {
                "mcp_servers": {},
            },
            "system": {
                "console_output": False,
                "enable_meta_tool": True,
                "timeout": 60,
                "tracing_endpoint": "",
            },
            "channels": {
                "feishu": {
                    "enabled": False,
                    "app_id": "",
                    "app_secret": "",
                    "encrypt_key": "",
                    "verification_token": "",
                    "group_policy": "mention",
                    "reply_mode": "chat",
                }
            },
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(fallback, f, ensure_ascii=False, indent=2)
        self.is_bootstrapped = True

    def get_tool_groups(self) -> dict[str, dict]:
        """Get tool group configurations."""
        return self.resources.get("tool_groups", {})

    def get_mcp_servers(self) -> dict[str, dict]:
        """Get MCP server configurations."""
        return self.resources.get("mcp_servers", {})

    def get_custom_tools(self) -> dict[str, dict]:
        """Get custom tool configurations."""
        return self.resources.get("custom_tools", {})

    def get_agent_skills(self) -> dict[str, dict]:
        """Get agent skill configurations."""
        return self.resources.get("agent_skills", {})

    def resolve_workspace_path(self, value: str) -> str:
        """Resolve a path against workspace root if it's relative."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str((self.workspace_dir / path).resolve())

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "model_name": self.model.model_name,
                "api_key": self.model.api_key[:10] + "..."
                if self.model.api_key
                else "",
                "api_base": self.model.api_base,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
            },
            "resources": self.resources,
            "paths": {
                "config_file": str(self.config_file),
                "home_dir": str(self.home_dir),
                "workspace_dir": str(self.workspace_dir),
                "builtin_skills_dir": str(self.builtin_skills_dir),
                "workspace_skills_dir": str(self.workspace_skills_dir),
                "workspace_tools_dir": str(self.workspace_tools_dir),
            },
            "system": {
                "console_output": self.system.console_output,
                "enable_meta_tool": self.system.enable_meta_tool,
                "timeout": self.system.timeout,
                "tracing_endpoint": self.system.tracing_endpoint,
            },
            "channels": {
                "feishu": {
                    "enabled": self.feishu.enabled,
                    "app_id": self.feishu.app_id,
                    "app_secret": "***" if self.feishu.app_secret else "",
                    "encrypt_key": "***" if self.feishu.encrypt_key else "",
                    "verification_token": "***" if self.feishu.verification_token else "",
                    "group_policy": self.feishu.group_policy,
                    "reply_mode": self.feishu.reply_mode,
                }
            },
        }

    def __repr__(self) -> str:
        return (
            f"Config(model={self.model.model_name}, tools={len(self.get_custom_tools())}, "
            f"config_file={self.config_file})"
        )
