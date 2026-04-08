"""Tests for the sandbox package — AST checker, subprocess sandbox, and integration."""

from __future__ import annotations

import asyncio
import os
import platform
import sys

import pytest

from babybot.sandbox import (
    ASTSafetyChecker,
    SandboxConfig,
    SubprocessSandbox,
    SafetyViolation,
)


# ============================================================================
# ASTSafetyChecker tests
# ============================================================================


class TestASTSafetyCheckerSafeCode:
    """Code that SHOULD pass the AST checker."""

    def setup_method(self) -> None:
        self.checker = ASTSafetyChecker()

    def test_basic_arithmetic(self) -> None:
        assert self.checker.check("x = 1 + 2") == []

    def test_allowed_stdlib_imports(self) -> None:
        assert self.checker.check("import json") == []
        assert self.checker.check("import math") == []
        assert self.checker.check("import re") == []
        assert self.checker.check("import os.path") == []
        assert self.checker.check("from collections import defaultdict") == []
        assert self.checker.check("from pathlib import Path") == []
        assert self.checker.check("import datetime") == []

    def test_allowed_third_party_imports(self) -> None:
        assert self.checker.check("import requests") == []
        assert self.checker.check("import httpx") == []
        assert self.checker.check("import numpy") == []
        assert self.checker.check("import pandas") == []
        assert self.checker.check("from pydantic import BaseModel") == []

    def test_os_module_allowed(self) -> None:
        # 'os' is in the allowlist for general filesystem operations
        assert self.checker.check("import os") == []
        assert self.checker.check("from os import path") == []

    def test_future_annotations(self) -> None:
        assert self.checker.check("from __future__ import annotations") == []

    def test_function_definitions(self) -> None:
        code = "def foo(x: int) -> int:\n    return x + 1"
        assert self.checker.check(code) == []

    def test_class_definitions(self) -> None:
        code = "class Foo:\n    def bar(self): pass"
        assert self.checker.check(code) == []

    def test_list_comprehension(self) -> None:
        assert self.checker.check("[x**2 for x in range(10)]") == []

    def test_multiline_code(self) -> None:
        code = (
            "import json\n"
            "import math\n"
            "data = json.dumps({'pi': math.pi})\n"
            "print(data)\n"
        )
        assert self.checker.check(code) == []


class TestASTSafetyCheckerBlockedCode:
    """Code that SHOULD be blocked by the AST checker."""

    def setup_method(self) -> None:
        self.checker = ASTSafetyChecker()

    def test_blocked_import_subprocess(self) -> None:
        violations = self.checker.check("import subprocess")
        assert len(violations) == 1
        assert violations[0].kind == "import"
        assert "subprocess" in violations[0].description

    def test_blocked_import_ctypes(self) -> None:
        violations = self.checker.check("import ctypes")
        assert len(violations) >= 1
        assert "ctypes" in violations[0].description

    def test_blocked_from_import(self) -> None:
        violations = self.checker.check("from subprocess import run")
        assert len(violations) >= 1
        assert "subprocess" in violations[0].description

    def test_blocked_eval_call(self) -> None:
        violations = self.checker.check("eval('1+1')")
        assert len(violations) >= 1
        assert any(v.kind == "builtin_call" for v in violations)

    def test_blocked_exec_call(self) -> None:
        violations = self.checker.check("exec('x = 1')")
        assert len(violations) >= 1
        assert any(v.kind == "builtin_call" for v in violations)

    def test_blocked_dunder_import(self) -> None:
        violations = self.checker.check("__import__('os')")
        assert len(violations) >= 1
        assert any(v.kind == "builtin_call" for v in violations)

    def test_blocked_compile(self) -> None:
        violations = self.checker.check("compile('x=1', '', 'exec')")
        assert len(violations) >= 1

    def test_blocked_subclasses_access(self) -> None:
        violations = self.checker.check("object.__subclasses__()")
        assert len(violations) >= 1
        assert any(v.kind == "attribute_access" for v in violations)

    def test_blocked_bases_access(self) -> None:
        violations = self.checker.check("x.__bases__")
        assert len(violations) >= 1

    def test_multiple_violations(self) -> None:
        code = "import subprocess\neval('bad')\n__import__('ctypes')"
        violations = self.checker.check(code)
        assert len(violations) >= 3

    def test_syntax_error_reported(self) -> None:
        violations = self.checker.check("def (invalid syntax:")
        assert len(violations) == 1
        assert "Syntax error" in violations[0].description

    def test_blocked_breakpoint(self) -> None:
        violations = self.checker.check("breakpoint()")
        assert len(violations) >= 1

    # --- Key bypass vectors that the old regex approach would miss ---

    def test_blocks_dunder_import_bypass(self) -> None:
        """This bypasses regex `eval\\s*\\(` but AST catches __import__."""
        violations = self.checker.check("__import__('subprocess').run(['ls'])")
        assert len(violations) >= 1

    def test_blocks_importlib_when_not_allowed(self) -> None:
        """importlib is not in the default allowlist."""
        violations = self.checker.check("import importlib")
        assert len(violations) >= 1

    def test_blocks_code_module(self) -> None:
        """code module allows interactive execution."""
        violations = self.checker.check("import code")
        assert len(violations) >= 1


class TestASTSafetyCheckerCustomization:
    """Test customization of the checker."""

    def test_extra_allowed_modules(self) -> None:
        checker = ASTSafetyChecker(extra_allowed_modules=frozenset({"my_custom_lib"}))
        assert checker.check("import my_custom_lib") == []

    def test_custom_allowed_modules_replaces_default(self) -> None:
        checker = ASTSafetyChecker(allowed_modules=frozenset({"only_this"}))
        # Default modules are now blocked
        violations = checker.check("import json")
        assert len(violations) >= 1
        # Custom module is allowed
        assert checker.check("import only_this") == []

    def test_check_and_format_returns_none_for_safe(self) -> None:
        checker = ASTSafetyChecker()
        assert checker.check_and_format("x = 1") is None

    def test_check_and_format_returns_string_for_blocked(self) -> None:
        checker = ASTSafetyChecker()
        result = checker.check_and_format("import subprocess")
        assert result is not None
        assert "Blocked" in result
        assert "subprocess" in result


# ============================================================================
# SandboxConfig tests
# ============================================================================


class TestSandboxConfig:
    def test_default_values(self) -> None:
        config = SandboxConfig()
        assert config.ast_check_enabled is True
        assert config.resource_limits_enabled is True
        assert config.login_shell is False
        assert "PATH" in config.env_allowlist
        assert "HOME" in config.env_allowlist
        assert config.cpu_limit_seconds == 300
        assert config.memory_limit_mb == 512

    def test_env_secret_patterns(self) -> None:
        config = SandboxConfig()
        assert "_KEY" in config.env_secret_patterns
        assert "_SECRET" in config.env_secret_patterns
        assert "_TOKEN" in config.env_secret_patterns

    def test_custom_config(self) -> None:
        config = SandboxConfig(
            login_shell=True,
            cpu_limit_seconds=60,
            memory_limit_mb=256,
        )
        assert config.login_shell is True
        assert config.cpu_limit_seconds == 60
        assert config.memory_limit_mb == 256


# ============================================================================
# SubprocessSandbox tests
# ============================================================================


class TestSubprocessSandboxSafeEnv:
    def test_filters_to_allowlist(self) -> None:
        sandbox = SubprocessSandbox()
        base_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "OPENAI_API_KEY": "sk-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "RANDOM_VAR": "value",
        }
        safe = sandbox.build_safe_env(base_env)
        assert safe["PATH"] == "/usr/bin"
        assert safe["HOME"] == "/home/user"
        assert "OPENAI_API_KEY" not in safe
        assert "AWS_SECRET_ACCESS_KEY" not in safe
        assert "RANDOM_VAR" not in safe

    def test_ensures_path_present(self) -> None:
        sandbox = SubprocessSandbox()
        safe = sandbox.build_safe_env({})
        assert "PATH" in safe

    def test_secret_pattern_double_check(self) -> None:
        """Even if a var name is in the allowlist, secret patterns block it."""
        config = SandboxConfig(
            env_allowlist=frozenset({"MY_API_KEY", "PATH", "HOME"}),
        )
        sandbox = SubprocessSandbox(config)
        base = {"MY_API_KEY": "secret", "PATH": "/usr/bin", "HOME": "/home"}
        safe = sandbox.build_safe_env(base)
        # MY_API_KEY matches _KEY pattern → blocked
        assert "MY_API_KEY" not in safe
        assert safe["PATH"] == "/usr/bin"


class TestSubprocessSandboxShellArgs:
    def test_default_no_login_shell(self) -> None:
        sandbox = SubprocessSandbox()
        exe, args = sandbox.build_shell_args("echo hello")
        assert exe == "/bin/sh"
        assert args == ["-c", "echo hello"]

    def test_login_shell_when_configured(self) -> None:
        sandbox = SubprocessSandbox(SandboxConfig(login_shell=True))
        exe, args = sandbox.build_shell_args("echo hello")
        assert exe == "/bin/sh"
        assert args == ["-lc", "echo hello"]


@pytest.mark.asyncio
class TestSubprocessSandboxRun:
    async def test_simple_echo(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-c", "print('hello sandbox')"],
            timeout=10.0,
        )
        assert result.returncode == 0
        assert "hello sandbox" in result.stdout
        assert result.timed_out is False

    async def test_stdin_data(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-"],
            stdin_data=b"print('from stdin')",
            timeout=10.0,
        )
        assert result.returncode == 0
        assert "from stdin" in result.stdout

    async def test_timeout(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-c", "import time; time.sleep(60)"],
            timeout=1.0,
        )
        assert result.timed_out is True
        assert result.returncode == -1

    async def test_nonzero_exit(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-c", "raise SystemExit(42)"],
            timeout=10.0,
        )
        assert result.returncode == 42
        assert result.timed_out is False

    async def test_stderr_captured(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-c", "import sys; sys.stderr.write('err msg')"],
            timeout=10.0,
        )
        assert "err msg" in result.stderr

    async def test_env_filtering(self) -> None:
        """Verify that subprocess env is filtered through the sandbox."""
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-c", "import os; print(os.environ.get('OPENAI_API_KEY', 'NOT_SET'))"],
            env={"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-secret"},
            timeout=10.0,
        )
        assert "NOT_SET" in result.stdout
        assert "sk-secret" not in result.stdout

    async def test_cwd_respected(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            sys.executable,
            ["-c", "import os; print(os.getcwd())"],
            cwd="/tmp",
            timeout=10.0,
        )
        # On macOS /tmp is a symlink to /private/tmp
        assert "tmp" in result.stdout.lower()

    async def test_bad_executable(self) -> None:
        sandbox = SubprocessSandbox()
        result = await sandbox.run(
            "/nonexistent/binary",
            [],
            timeout=5.0,
        )
        assert result.returncode == -1
        assert "Failed to start" in result.stderr


# ============================================================================
# Integration test: check_python_safety with AST + regex
# ============================================================================


class TestCheckPythonSafetyIntegration:
    """Test the integrated check_python_safety from resource_workspace_tools."""

    def test_safe_code_passes(self) -> None:
        from babybot.resource_workspace_tools import check_python_safety

        assert check_python_safety("print('hello')") is None

    def test_blocked_by_ast(self) -> None:
        from babybot.resource_workspace_tools import check_python_safety

        result = check_python_safety("import subprocess")
        assert result is not None
        assert "Blocked" in result

    def test_blocked_by_regex_fallback(self) -> None:
        from babybot.resource_workspace_tools import check_python_safety

        # shutil.rmtree is in the allowlist (os/shutil are allowed modules)
        # but regex still catches the specific dangerous pattern
        result = check_python_safety("import shutil\nshutil.rmtree('/')")
        assert result is not None
        assert "Blocked" in result

    def test_dunder_import_blocked(self) -> None:
        from babybot.resource_workspace_tools import check_python_safety

        result = check_python_safety("__import__('subprocess').run(['ls'])")
        assert result is not None
        assert "Blocked" in result

    def test_eval_blocked(self) -> None:
        from babybot.resource_workspace_tools import check_python_safety

        result = check_python_safety('eval(\'__import__("os").system("ls")\')')
        assert result is not None

    def test_safe_data_processing(self) -> None:
        from babybot.resource_workspace_tools import check_python_safety

        code = (
            "import json\n"
            "import math\n"
            "data = json.loads('{\"x\": 1}')\n"
            "print(math.sqrt(data['x']))\n"
        )
        assert check_python_safety(code) is None


class TestCheckShellSafetyIntegration:
    def test_safe_command_passes(self) -> None:
        from babybot.resource_workspace_tools import check_shell_safety

        assert check_shell_safety("echo hello") is None
        assert check_shell_safety("ls -la") is None

    def test_dangerous_command_blocked(self) -> None:
        from babybot.resource_workspace_tools import check_shell_safety

        result = check_shell_safety("rm -rf /")
        assert result is not None
        assert "Blocked" in result
