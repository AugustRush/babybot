"""AST-based safety checker — allowlist approach replacing regex denylist.

First-principles design:
  - Regex denylists are fundamentally bypassable (infinite evasion vectors).
  - AST analysis understands Python syntax structure — not affected by
    whitespace, string encoding, comments, or string concatenation tricks.
  - We use an ALLOWLIST of safe modules rather than a denylist of dangerous ones.
  - Dangerous builtins (eval, exec, __import__, etc.) are explicitly blocked.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyViolation:
    """A single safety violation found by the AST checker."""

    kind: Literal["import", "builtin_call", "attribute_access"]
    description: str
    lineno: int = 0
    col_offset: int = 0

    def __str__(self) -> str:
        loc = f"line {self.lineno}" if self.lineno else "unknown location"
        return f"[{self.kind}] {self.description} ({loc})"


# Modules considered safe for general-purpose computation.
# This list is deliberately conservative — it can be extended per-deployment.
_DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        # Python stdlib — pure computation
        "abc",
        "argparse",
        "ast",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "codecs",
        "collections",
        "colorsys",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fnmatch",
        "fractions",
        "functools",
        "glob",
        "graphlib",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "math",
        "mimetypes",
        "numbers",
        "operator",
        "os.path",
        "pathlib",
        "plistlib",
        "posixpath",
        "pprint",
        "pydoc",
        "random",
        "re",
        "secrets",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "time",
        "timeit",
        "token",
        "tokenize",
        "traceback",
        "types",
        "typing",
        "typing_extensions",
        "unicodedata",
        "unittest",
        "unittest.mock",
        "urllib.parse",
        "uuid",
        "warnings",
        "weakref",
        "xml",
        "xml.etree",
        "xml.etree.ElementTree",
        "zipfile",
        "zlib",
        # Commonly needed by data processing skills
        "array",
        "queue",
        "threading",
        # Filesystem (read operations are generally needed)
        "os",
        "shutil",
        "tempfile",
        # Network (often needed by skills)
        "http",
        "http.client",
        "http.server",
        "urllib",
        "urllib.request",
        "urllib.error",
        "socket",
        "ssl",
        # Async
        "asyncio",
        "concurrent",
        "concurrent.futures",
        # Encoding
        "chardet",
        "charset_normalizer",
        # Third-party data/ML libs (extend as needed)
        "numpy",
        "pandas",
        "scipy",
        "PIL",
        "pillow",
        "requests",
        "httpx",
        "aiohttp",
        "yaml",
        "pyyaml",
        "toml",
        "tomli",
        "tomllib",
        "bs4",
        "beautifulsoup4",
        "lxml",
        "markdown",
        "markdownify",
        "pydantic",
        "dotenv",
        "python_dotenv",
        "jinja2",
        "click",
        "rich",
        "tqdm",
        "openai",
        "anthropic",
        # Allow __future__ always
        "__future__",
    }
)

# Builtin names that should never be called dynamically.
_BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "exit",
        "quit",
    }
)

# Attribute names that indicate dangerous dynamic access.
_BLOCKED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__class__",  # when used for escape: obj.__class__.__bases__[0].__subclasses__()
    }
)


class ASTSafetyChecker:
    """Checks Python source code for safety using AST-level analysis.

    This is Layer 1 of the defense-in-depth model.
    It is NOT a sandbox — it is a static filter that catches
    the most common attack vectors before code reaches the subprocess.
    """

    def __init__(
        self,
        *,
        allowed_modules: frozenset[str] | None = None,
        blocked_builtins: frozenset[str] | None = None,
        blocked_attributes: frozenset[str] | None = None,
        extra_allowed_modules: frozenset[str] | None = None,
    ) -> None:
        base_modules = (
            allowed_modules if allowed_modules is not None else _DEFAULT_ALLOWED_MODULES
        )
        if extra_allowed_modules:
            base_modules = base_modules | extra_allowed_modules
        self._allowed_modules = base_modules
        self._blocked_builtins = blocked_builtins or _BLOCKED_BUILTINS
        self._blocked_attributes = blocked_attributes or _BLOCKED_ATTRIBUTES

    def check(self, code: str) -> list[SafetyViolation]:
        """Analyze Python source code and return a list of safety violations.

        Returns an empty list if the code passes all checks.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [
                SafetyViolation(
                    kind="import",
                    description=f"Syntax error: {exc}",
                    lineno=exc.lineno or 0,
                )
            ]

        violations: list[SafetyViolation] = []
        for node in ast.walk(tree):
            self._check_node(node, violations)
        return violations

    def check_and_format(self, code: str) -> str | None:
        """Convenience: returns a formatted error string, or None if safe."""
        violations = self.check(code)
        if not violations:
            return None
        lines = ["Blocked: code failed safety analysis"]
        for v in violations[:5]:  # limit to 5 to keep messages readable
            lines.append(f"  - {v}")
        if len(violations) > 5:
            lines.append(f"  ... and {len(violations) - 5} more violations")
        return "\n".join(lines)

    def _check_node(self, node: ast.AST, violations: list[SafetyViolation]) -> None:
        if isinstance(node, ast.Import):
            self._check_import(node, violations)
        elif isinstance(node, ast.ImportFrom):
            self._check_import_from(node, violations)
        elif isinstance(node, ast.Call):
            self._check_call(node, violations)
        elif isinstance(node, ast.Attribute):
            self._check_attribute(node, violations)

    def _check_import(
        self, node: ast.Import, violations: list[SafetyViolation]
    ) -> None:
        for alias in node.names:
            top_module = alias.name.split(".")[0]
            if not self._is_module_allowed(alias.name):
                violations.append(
                    SafetyViolation(
                        kind="import",
                        description=f"Import of '{alias.name}' is not in the allowlist"
                        f" (top-level: '{top_module}')",
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    )
                )

    def _check_import_from(
        self, node: ast.ImportFrom, violations: list[SafetyViolation]
    ) -> None:
        module = node.module or ""
        if not module:
            return  # relative import without module name — allow
        top_module = module.split(".")[0]
        if not self._is_module_allowed(module):
            violations.append(
                SafetyViolation(
                    kind="import",
                    description=f"Import from '{module}' is not in the allowlist"
                    f" (top-level: '{top_module}')",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            )

    def _check_call(self, node: ast.Call, violations: list[SafetyViolation]) -> None:
        func_name = self._extract_call_name(node.func)
        if func_name and func_name in self._blocked_builtins:
            violations.append(
                SafetyViolation(
                    kind="builtin_call",
                    description=f"Call to blocked builtin '{func_name}'",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            )

    def _check_attribute(
        self, node: ast.Attribute, violations: list[SafetyViolation]
    ) -> None:
        if node.attr in self._blocked_attributes:
            violations.append(
                SafetyViolation(
                    kind="attribute_access",
                    description=f"Access to blocked attribute '{node.attr}'",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            )

    def _is_module_allowed(self, module_path: str) -> bool:
        """Check if a module (possibly dotted) is in the allowlist.

        We check the full path first, then progressively shorter prefixes.
        E.g., 'os.path.join' → check 'os.path.join', 'os.path', 'os'.
        """
        parts = module_path.split(".")
        # Check full path, then progressively shorter prefixes
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self._allowed_modules:
                return True
        return False

    @staticmethod
    def _extract_call_name(func_node: ast.expr) -> str | None:
        """Extract the function name from a Call node's func attribute."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None
