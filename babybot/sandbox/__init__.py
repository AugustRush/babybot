"""Sandbox package: defense-in-depth code execution isolation.

Three layers of protection:
  1. AST-level static analysis (allowlist-based, not regex denylist)
  2. Process-level isolation (rlimits, env filtering, process groups)
  3. Configurable security policy
"""

from __future__ import annotations

from .ast_filter import ASTSafetyChecker, SafetyViolation
from .config import SandboxConfig
from .subprocess_sandbox import SubprocessSandbox, SubprocessResult

__all__ = [
    "ASTSafetyChecker",
    "SafetyViolation",
    "SandboxConfig",
    "SubprocessSandbox",
    "SubprocessResult",
]
