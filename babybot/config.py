"""Unified configuration management."""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


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
        errors = []
        if not self.api_key:
            errors.append(
                "API key is required. Set in ~/.babybot/config.json "
                "or OPENAI_API_KEY environment variable."
            )
        if self.temperature < 0 or self.temperature > 2:
            errors.append(
                f"Temperature must be between 0 and 2, got {self.temperature}"
            )
        if self.max_tokens <= 0 or self.max_tokens > 32768:
            errors.append(
                f"max_tokens must be between 1 and 32768, got {self.max_tokens}"
            )
        if errors:
            raise ValueError("; ".join(errors))


@dataclass
class SystemConfig:
    """System configuration."""

    console_output: bool = False
    enable_meta_tool: bool = True
    timeout: int = 600
    subtask_timeout: int = 60
    skill_route_timeout: float = 3.0
    tracing_endpoint: str = ""
    max_parallel: int = 4
    idle_timeout: int = 60
    max_concurrency: int = 8
    scheduled_max_concurrency: int = 2
    message_queue_maxsize: int = 1000
    max_per_chat: int = 1
    send_ack: bool = True
    python_executable: str = ""
    python_fallback_executables: list[str] | tuple[str, ...] = ()
    worker_max_steps: int = 14
    orchestrator_max_steps: int = 30
    context_history_tokens: int = 2000
    context_compact_threshold: int = 3000
    context_max_chats: int = 500
    interactive_session_max_age_seconds: int = 7200
    routing_enabled: bool = True
    routing_model_name: str = ""
    routing_timeout: float = 3.0
    reflection_enabled: bool = True
    reflection_max_hints: int = 3
    debug_runtime_feedback: bool = False
    policy_learning_enabled: bool = True
    policy_learning_min_samples: int = 0
    policy_learning_explore_ratio: float = -1.0


@dataclass
class WeixinConfig:
    """Weixin channel configuration."""

    enabled: bool = False
    base_url: str = "https://ilinkai.weixin.qq.com"
    cdn_base_url: str = "https://novac2c.cdn.weixin.qq.com/c2c"
    token: str = ""
    state_dir: str = ""
    media_dir: str = ""
    poll_timeout: int = 35
    allow_from: list[str] | tuple[str, ...] = ()

    def validate(self) -> None:
        """Validate Weixin configuration when enabled."""
        if not self.enabled:
            return
        if not self.token:
            raise ValueError("token is required when Weixin channel is enabled")


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
    react_emoji: str = "THUMBSUP"
    media_dir: str = ""
    stream_reply: bool = False

    def validate(self) -> None:
        """Validate Feishu configuration when enabled."""
        if not self.enabled:
            return
        errors = []
        if not self.app_id:
            errors.append("app_id is required when Feishu channel is enabled")
        if not self.app_secret:
            errors.append("app_secret is required when Feishu channel is enabled")
        if errors:
            raise ValueError("; ".join(errors))


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
        env_home = os.getenv("BABYBOT_HOME", "")
        env_workspace = os.getenv("BABYBOT_WORKSPACE", "")
        if config_file:
            self.config_file = Path(config_file).expanduser()
        else:
            env_config = os.getenv("BABYBOT_CONFIG", "")
            self.config_file = (
                Path(env_config).expanduser()
                if env_config
                else Path(env_home or "~/.babybot").expanduser() / "config.json"
            )

        if env_home:
            self.home_dir = Path(env_home).expanduser()
        elif config_file:
            self.home_dir = self.config_file.parent
        else:
            self.home_dir = Path("~/.babybot").expanduser()

        self.workspace_dir = Path(
            env_workspace or str(self.home_dir / "workspace")
        ).expanduser()
        self.builtin_skills_dir = Path(__file__).resolve().parent.parent / "skills"
        self.workspace_skills_dir = self.workspace_dir / "skills"
        self.scheduled_tasks_file = self.workspace_dir / "scheduled_tasks.json"
        self.raw_config: dict[str, Any] = {}
        self.is_bootstrapped = False
        self.scheduled_tasks: list[dict[str, Any]] = []

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
            timeout=system_conf.get("timeout", 600),
            subtask_timeout=system_conf.get("subtask_timeout", 60),
            skill_route_timeout=system_conf.get("skill_route_timeout", 3.0),
            tracing_endpoint=system_conf.get("tracing_endpoint", ""),
            max_parallel=system_conf.get("max_parallel", 4),
            idle_timeout=system_conf.get("idle_timeout", 60),
            max_concurrency=system_conf.get("max_concurrency", 8),
            scheduled_max_concurrency=system_conf.get("scheduled_max_concurrency", 2),
            message_queue_maxsize=system_conf.get("message_queue_maxsize", 1000),
            max_per_chat=system_conf.get("max_per_chat", 1),
            send_ack=system_conf.get("send_ack", True),
            python_executable=system_conf.get("python_executable", ""),
            python_fallback_executables=system_conf.get(
                "python_fallback_executables", []
            ),
            worker_max_steps=system_conf.get("worker_max_steps", 14),
            orchestrator_max_steps=system_conf.get("orchestrator_max_steps", 30),
            context_history_tokens=system_conf.get("context_history_tokens", 2000),
            context_compact_threshold=system_conf.get(
                "context_compact_threshold", 3000
            ),
            context_max_chats=system_conf.get("context_max_chats", 500),
            interactive_session_max_age_seconds=system_conf.get(
                "interactive_session_max_age_seconds", 7200
            ),
            routing_enabled=system_conf.get("routing_enabled", True),
            routing_model_name=system_conf.get("routing_model_name", ""),
            routing_timeout=system_conf.get("routing_timeout", 3.0),
            reflection_enabled=system_conf.get("reflection_enabled", True),
            reflection_max_hints=system_conf.get("reflection_max_hints", 3),
            debug_runtime_feedback=system_conf.get("debug_runtime_feedback", False),
            policy_learning_enabled=system_conf.get("policy_learning_enabled", True),
            policy_learning_min_samples=system_conf.get(
                "policy_learning_min_samples", 0
            ),
            policy_learning_explore_ratio=system_conf.get(
                "policy_learning_explore_ratio", -1.0
            ),
        )

        # Resource configuration — support both flat keys and legacy "resources" wrapper
        _res = self.raw_config.get("resources", {})
        self.mcp_servers: dict[str, dict] = (
            self.raw_config.get("mcp_servers") or _res.get("mcp_servers") or {}
        )
        self.tool_groups: dict[str, dict] = (
            self.raw_config.get("tool_groups") or _res.get("tool_groups") or {}
        )
        self.agent_skills: dict[str, dict] = (
            self.raw_config.get("agent_skills") or _res.get("agent_skills") or {}
        )

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
            react_emoji=feishu_conf.get("react_emoji", "THUMBSUP"),
            media_dir=feishu_conf.get("media_dir", ""),
            stream_reply=feishu_conf.get("stream_reply", False),
        )
        weixin_conf = channels_conf.get("weixin", {})
        self.weixin = WeixinConfig(
            enabled=weixin_conf.get("enabled", False),
            base_url=weixin_conf.get("base_url", "https://ilinkai.weixin.qq.com"),
            cdn_base_url=weixin_conf.get(
                "cdn_base_url", "https://novac2c.cdn.weixin.qq.com/c2c"
            ),
            token=weixin_conf.get("token", ""),
            state_dir=weixin_conf.get("state_dir", ""),
            media_dir=weixin_conf.get("media_dir", ""),
            poll_timeout=weixin_conf.get("poll_timeout", 35),
            allow_from=tuple(weixin_conf.get("allow_from", []) or []),
        )

    def _load_config(self) -> None:
        """Load configuration from file."""
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_skills_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self._bootstrap_config_file()

        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.raw_config = json.load(f)
        else:
            self.raw_config = {}
        self._load_scheduled_tasks()

    def _load_scheduled_tasks(self) -> None:
        """Load scheduled tasks from the dedicated workspace file."""
        self.scheduled_tasks_file.parent.mkdir(parents=True, exist_ok=True)

        legacy_tasks = self.raw_config.get("scheduled_tasks")
        if not self.scheduled_tasks_file.exists():
            tasks = legacy_tasks if isinstance(legacy_tasks, list) else []
            self._write_scheduled_tasks(tasks)
            if tasks:
                logger.info(
                    "Migrated %d scheduled task(s) from config.json to %s",
                    len(tasks),
                    self.scheduled_tasks_file,
                )

        try:
            with open(self.scheduled_tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Failed to load scheduled tasks file: {self.scheduled_tasks_file}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Scheduled tasks file must contain a JSON list: {self.scheduled_tasks_file}"
            )
        self.scheduled_tasks = data

    def _write_scheduled_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Persist scheduled tasks to the dedicated workspace file."""
        with open(self.scheduled_tasks_file, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

    def get_scheduled_tasks(self) -> list[dict[str, Any]]:
        """Get scheduled tasks loaded from the workspace file."""
        return list(self.scheduled_tasks)

    def save_scheduled_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Replace the workspace scheduled task definitions."""
        self._write_scheduled_tasks(tasks)
        self.scheduled_tasks = list(tasks)

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
                logger.info(f"Created config file from example: {candidate}")
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
            "mcp_servers": {},
            "system": {
                "console_output": False,
                "enable_meta_tool": True,
                "timeout": 600,
                "subtask_timeout": 60,
                "skill_route_timeout": 3.0,
                "tracing_endpoint": "",
                "max_parallel": 4,
                "idle_timeout": 60,
                "max_concurrency": 8,
                "scheduled_max_concurrency": 2,
                "message_queue_maxsize": 1000,
                "max_per_chat": 1,
                "send_ack": True,
                "python_executable": "",
                "python_fallback_executables": [],
                "worker_max_steps": 14,
                "orchestrator_max_steps": 30,
                "context_history_tokens": 2000,
                "context_compact_threshold": 3000,
                "context_max_chats": 500,
                "interactive_session_max_age_seconds": 7200,
                "routing_enabled": True,
                "routing_model_name": "",
                "routing_timeout": 3.0,
                "reflection_enabled": True,
                "reflection_max_hints": 3,
                "debug_runtime_feedback": False,
                "policy_learning_enabled": True,
                "policy_learning_min_samples": 0,
                "policy_learning_explore_ratio": -1.0,
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
                    "react_emoji": "THUMBSUP",
                    "media_dir": "",
                    "stream_reply": False,
                },
                "weixin": {
                    "enabled": False,
                    "base_url": "https://ilinkai.weixin.qq.com",
                    "cdn_base_url": "https://novac2c.cdn.weixin.qq.com/c2c",
                    "token": "",
                    "state_dir": "",
                    "media_dir": "",
                    "poll_timeout": 35,
                    "allow_from": [],
                },
            },
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(fallback, f, ensure_ascii=False, indent=2)
        self.is_bootstrapped = True
        logger.info("Created default config file (fallback)")

    def get_channel_config(self, name: str) -> Any:
        """Get the config object for a specific channel."""
        if name == "feishu":
            return self.feishu
        if name == "weixin":
            return self.weixin
        return None

    def get_tool_groups(self) -> dict[str, dict]:
        """Get tool group configurations."""
        return self.tool_groups

    def get_mcp_servers(self) -> dict[str, dict]:
        """Get MCP server configurations."""
        return self.mcp_servers

    def get_agent_skills(self) -> dict[str, dict]:
        """Get agent skill configurations."""
        return self.agent_skills

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
            "mcp_servers": self.mcp_servers,
            "paths": {
                "config_file": str(self.config_file),
                "home_dir": str(self.home_dir),
                "workspace_dir": str(self.workspace_dir),
                "scheduled_tasks_file": str(self.scheduled_tasks_file),
                "builtin_skills_dir": str(self.builtin_skills_dir),
                "workspace_skills_dir": str(self.workspace_skills_dir),
            },
            "system": {
                "console_output": self.system.console_output,
                "enable_meta_tool": self.system.enable_meta_tool,
                "timeout": self.system.timeout,
                "subtask_timeout": self.system.subtask_timeout,
                "skill_route_timeout": self.system.skill_route_timeout,
                "tracing_endpoint": self.system.tracing_endpoint,
                "idle_timeout": self.system.idle_timeout,
                "max_concurrency": self.system.max_concurrency,
                "scheduled_max_concurrency": self.system.scheduled_max_concurrency,
                "message_queue_maxsize": self.system.message_queue_maxsize,
                "max_per_chat": self.system.max_per_chat,
                "send_ack": self.system.send_ack,
                "python_executable": self.system.python_executable,
                "python_fallback_executables": list(
                    self.system.python_fallback_executables
                ),
                "worker_max_steps": self.system.worker_max_steps,
                "orchestrator_max_steps": self.system.orchestrator_max_steps,
                "context_history_tokens": self.system.context_history_tokens,
                "context_compact_threshold": self.system.context_compact_threshold,
                "context_max_chats": self.system.context_max_chats,
                "interactive_session_max_age_seconds": self.system.interactive_session_max_age_seconds,
                "routing_enabled": self.system.routing_enabled,
                "routing_model_name": self.system.routing_model_name,
                "routing_timeout": self.system.routing_timeout,
                "reflection_enabled": self.system.reflection_enabled,
                "reflection_max_hints": self.system.reflection_max_hints,
                "debug_runtime_feedback": self.system.debug_runtime_feedback,
                "policy_learning_enabled": self.system.policy_learning_enabled,
                "policy_learning_min_samples": self.system.policy_learning_min_samples,
                "policy_learning_explore_ratio": self.system.policy_learning_explore_ratio,
            },
            "channels": {
                "feishu": {
                    "enabled": self.feishu.enabled,
                    "app_id": self.feishu.app_id,
                    "app_secret": "***" if self.feishu.app_secret else "",
                    "encrypt_key": "***" if self.feishu.encrypt_key else "",
                    "verification_token": "***"
                    if self.feishu.verification_token
                    else "",
                    "group_policy": self.feishu.group_policy,
                    "reply_mode": self.feishu.reply_mode,
                    "react_emoji": self.feishu.react_emoji,
                    "media_dir": self.feishu.media_dir,
                    "stream_reply": self.feishu.stream_reply,
                },
                "weixin": {
                    "enabled": self.weixin.enabled,
                    "base_url": self.weixin.base_url,
                    "cdn_base_url": self.weixin.cdn_base_url,
                    "token": "***" if self.weixin.token else "",
                    "state_dir": self.weixin.state_dir,
                    "media_dir": self.weixin.media_dir,
                    "poll_timeout": self.weixin.poll_timeout,
                    "allow_from": list(self.weixin.allow_from),
                },
            },
            "scheduled_tasks_count": len(self.scheduled_tasks),
        }

    def __repr__(self) -> str:
        return f"Config(model={self.model.model_name}, config_file={self.config_file})"
