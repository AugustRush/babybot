from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .resource_models import ToolGroup

logger = logging.getLogger(__name__)


class ResourceToolLoader:
    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def register_tool(
        self,
        func: Any,
        group_name: str = "basic",
        preset_kwargs: dict[str, Any] | None = None,
        func_name: str | None = None,
        collect_artifacts: bool = True,
    ) -> None:
        if group_name not in self._owner.groups:
            self._owner.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"{group_name} tools",
                active=False,
            )
        name = func_name or getattr(func, "__name__", "tool")
        self._owner.registry.register(
            self._owner._callable_tool_cls()(
                func=func,
                name=name,
                description=(inspect.getdoc(func) or "").splitlines()[0]
                if inspect.getdoc(func)
                else name,
                schema=self.json_schema_for_callable(func),
                preset_kwargs=preset_kwargs,
                resource_manager=self._owner,
                collect_artifacts=collect_artifacts,
            ),
            group=group_name,
        )

    def discover_workspace_tools(self) -> None:
        tools_root = self._owner.config.workspace_tools_dir
        if not tools_root.exists():
            return
        self.ensure_workspace_on_pythonpath()
        for py_file in sorted(tools_root.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            rel = py_file.relative_to(tools_root)
            group_name = rel.parts[0] if len(rel.parts) > 1 else "basic"
            try:
                module = self.load_tool_module(str(Path("tools") / rel))
                for func_name, func in inspect.getmembers(module, inspect.isfunction):
                    if func.__module__ != module.__name__ or func_name.startswith("_"):
                        continue
                    self.register_tool(func, group_name=group_name, func_name=func_name)
            except Exception as exc:
                logger.warning("Failed to load tools from %s: %s", py_file, exc)

    def ensure_workspace_on_pythonpath(self) -> None:
        workspace = str(self._owner.config.workspace_dir.resolve())
        if workspace not in sys.path:
            sys.path.insert(0, workspace)

    def load_tool_module(self, module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
        except SystemExit as exc:
            raise ModuleNotFoundError(
                f"Module {module_name} called sys.exit() during import"
            ) from exc
        resolved = self._owner.config.resolve_workspace_path(module_name)
        spec = importlib.util.spec_from_file_location(
            f"babybot_custom_{abs(hash(resolved))}",
            resolved,
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except SystemExit as exc:
                raise ModuleNotFoundError(
                    f"Module {module_name} called sys.exit() during import"
                ) from exc
            return module
        raise ModuleNotFoundError(f"Cannot import custom module: {module_name}")

    @classmethod
    def json_schema_for_callable(cls, func: Any) -> dict[str, Any]:
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if name in {"self", "context"}:
                continue
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            anno = hints.get(name, param.annotation)
            properties[name] = cls.schema_for_annotation(anno)
            if param.default is inspect._empty:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @classmethod
    def schema_for_annotation(cls, annotation: Any) -> dict[str, Any]:
        if annotation is inspect._empty:
            return {"type": "string"}

        origin = get_origin(annotation)
        args = [arg for arg in get_args(annotation) if arg is not type(None)]

        if origin is None:
            if annotation in {str}:
                return {"type": "string"}
            if annotation in {bool}:
                return {"type": "boolean"}
            if annotation in {int}:
                return {"type": "integer"}
            if annotation in {float}:
                return {"type": "number"}
            if annotation in {dict}:
                return {"type": "object"}
            if annotation in {list, tuple, set}:
                return {"type": "array"}
            return {"type": "string"}

        if origin in {list, tuple, set}:
            item_schema = cls.schema_for_annotation(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item_schema}

        if origin is dict:
            return {"type": "object"}

        if origin is Literal:
            values = [value for value in args if isinstance(value, (str, int, float, bool))]
            if not values:
                return {"type": "string"}
            first = values[0]
            if isinstance(first, bool):
                field_type = "boolean"
            elif isinstance(first, int):
                field_type = "integer"
            elif isinstance(first, float):
                field_type = "number"
            else:
                field_type = "string"
            return {"type": field_type, "enum": values}

        if origin in {Union, UnionType}:
            if len(args) == 1:
                return cls.schema_for_annotation(args[0])
            return {"anyOf": [cls.schema_for_annotation(arg) for arg in args]}

        return {"type": "string"}
