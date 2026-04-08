"""Sandbox configuration — security policy as a value object."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field


_IS_DARWIN = platform.system() == "Darwin"


@dataclass(frozen=True)
class SandboxConfig:
    """Configurable security policy for subprocess execution.

    All limits are opt-in via boolean flags so deployments can tune
    the security/usability tradeoff.
    """

    # --- Layer 1: AST analysis ---
    ast_check_enabled: bool = True

    # --- Layer 2: Process-level isolation ---
    resource_limits_enabled: bool = True
    login_shell: bool = False  # default OFF — no -l flag

    # Environment allowlist (only these env vars are passed to children)
    env_allowlist: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "PATH",
                "HOME",
                "USER",
                "LOGNAME",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "LANGUAGE",
                "TERM",
                "TMPDIR",
                "TEMPDIR",
                "TMP",
                "SHELL",  # some tools inspect $SHELL
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONIOENCODING",
                "PYTHONUNBUFFERED",
                # Common build vars
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "PKG_CONFIG_PATH",
            }
        )
    )

    # Patterns in env var names that trigger automatic exclusion
    env_secret_patterns: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "_KEY",
                "_SECRET",
                "_TOKEN",
                "_PASSWORD",
                "_CREDENTIAL",
                "_API_KEY",
                "API_KEY",
                "API_SECRET",
                "AWS_",
                "AZURE_",
                "GCP_",
            }
        )
    )

    # Resource limits (used when resource_limits_enabled=True)
    cpu_limit_seconds: int = 300
    # macOS: RLIMIT_AS doesn't work reliably; use RLIMIT_RSS instead
    memory_limit_mb: int = 512
    max_file_descriptors: int = 256
    max_child_processes: int = 64

    # Legacy regex checks: kept as a *supplementary* fallback alongside AST
    legacy_regex_enabled: bool = True
