from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

from .agent_kernel import ToolLease
from .resource_models import (
    CliArgumentSpec,
    CliToolSpec,
    LoadedSkill,
    ScriptFunctionSpec,
    SkillRuntimeConfig,
    ToolGroup,
)
from .resource_protocols import SkillHost

logger = logging.getLogger(__name__)


class SkillLoader:
    def __init__(self, owner: SkillHost) -> None:
        self._owner = owner

    def _record_load_error(
        self,
        *,
        skill_name: str,
        path: Path,
        error: str,
        stage: str,
    ) -> None:
        recorder = getattr(self._owner, "_record_skill_load_error", None)
        if callable(recorder):
            recorder(
                skill=skill_name,
                path=str(path),
                error=error,
                stage=stage,
            )
            return
        errors = getattr(self._owner, "_skill_load_errors", None)
        if errors is None:
            errors = []
            self._owner._skill_load_errors = errors
        errors.append(
            {
                "skill": skill_name,
                "path": str(path),
                "error": error,
                "stage": stage,
            }
        )

    def register_configured_skills(self, configured: dict[str, dict]) -> None:
        for name, conf in configured.items():
            try:
                directory = conf.get("directory", "")
                if not directory:
                    continue
                skill_dir = Path(self._owner.config.resolve_workspace_path(directory))
                if not skill_dir.exists() or not skill_dir.is_dir():
                    logger.warning("Configured skill not found: %s", skill_dir)
                    continue
                meta, prompt_summary, prompt_body = self.read_skill_document(skill_dir)
                resolved_name = meta.get("name", name)
                runtime = self._owner._build_skill_runtime({**meta, **conf})
                tool_group, tool_names = self.register_skill_tools(
                    resolved_name,
                    skill_dir,
                    runtime=runtime,
                    callable_tool_cls=self._owner._callable_tool_cls(),
                )
                meta_include_groups = self._parse_frontmatter_list(
                    meta.get("include_groups")
                )
                meta_include_tools = self._parse_frontmatter_list(
                    meta.get("include_tools")
                )
                meta_exclude_tools = self._parse_frontmatter_list(
                    meta.get("exclude_tools")
                )
                description = conf.get("description") or meta.get("description", "")
                keywords = self.normalize_keywords(
                    conf.get("keywords"),
                    fallback=(description, name),
                )
                phrases = self.normalize_phrases(
                    conf.get("keywords"),
                    fallback=(description, name),
                )
                self._owner._upsert_skill(
                    LoadedSkill(
                        name=meta.get("name", name),
                        description=description or f"Skill: {name}",
                        directory=str(skill_dir.resolve()),
                        prompt=prompt_summary,
                        prompt_body="",
                        prompt_body_path=str((skill_dir / "SKILL.md").resolve()),
                        role=meta.get("role", ""),
                        keywords=keywords,
                        phrases=phrases,
                        lease=ToolLease(
                            include_groups=tuple(
                                set(
                                    self._parse_frontmatter_list(
                                        conf.get("include_groups")
                                    )
                                )
                                | set(meta_include_groups)
                                | ({tool_group} if tool_group else set())
                            ),
                            include_tools=tuple(
                                dict.fromkeys(
                                    [
                                        *meta_include_tools,
                                        *self._parse_frontmatter_list(
                                            conf.get("include_tools")
                                        ),
                                    ]
                                )
                            ),
                            exclude_tools=tuple(
                                dict.fromkeys(
                                    [
                                        *meta_exclude_tools,
                                        *self._parse_frontmatter_list(
                                            conf.get("exclude_tools")
                                        ),
                                    ]
                                )
                            ),
                        ),
                        source="config",
                        active=bool(conf.get("active", True)),
                        tool_group=tool_group,
                        tools=tool_names,
                        runtime=runtime,
                    )
                )
            except Exception as exc:
                self._record_load_error(
                    skill_name=name,
                    path=Path(conf.get("directory", ""))
                    if conf.get("directory")
                    else Path("."),
                    error=str(exc),
                    stage="configured-skill",
                )
                logger.warning("Failed to load configured skill %s: %s", name, exc)

    def discover_skills(self) -> None:
        roots = [
            self._owner.config.workspace_skills_dir,
            self._owner.config.builtin_skills_dir,
        ]
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir() or not (child / "SKILL.md").exists():
                    continue
                try:
                    meta, prompt_summary, prompt_body = self.read_skill_document(child)
                    name = meta.get("name", child.name)
                    runtime = self._owner._build_skill_runtime(meta)
                    tool_group, tool_names = self.register_skill_tools(
                        name,
                        child,
                        runtime=runtime,
                        callable_tool_cls=self._owner._callable_tool_cls(),
                    )
                    meta_include_groups = self._parse_frontmatter_list(
                        meta.get("include_groups")
                    )
                    meta_include_tools = self._parse_frontmatter_list(
                        meta.get("include_tools")
                    )
                    meta_exclude_tools = self._parse_frontmatter_list(
                        meta.get("exclude_tools")
                    )
                    key = name.strip().lower()
                    if key in self._owner.skills:
                        continue
                    description = meta.get("description", f"Skill: {name}")
                    keywords = self.normalize_keywords(
                        None,
                        fallback=(description, name, prompt_summary[:400]),
                    )
                    phrases = self.normalize_phrases(
                        None,
                        fallback=(description, name),
                    )
                    self._owner._upsert_skill(
                        LoadedSkill(
                            name=name,
                            description=description,
                            directory=str(child.resolve()),
                            prompt=prompt_summary,
                            prompt_body="",
                            prompt_body_path=str((child / "SKILL.md").resolve()),
                            role=meta.get("role", ""),
                            keywords=keywords,
                            phrases=phrases,
                            source="auto",
                            active=True,
                            lease=ToolLease(
                                include_groups=tuple(
                                    dict.fromkeys(
                                        [
                                            *meta_include_groups,
                                            *([tool_group] if tool_group else []),
                                        ]
                                    )
                                ),
                                include_tools=tuple(meta_include_tools),
                                exclude_tools=tuple(meta_exclude_tools),
                            ),
                            tool_group=tool_group,
                            tools=tool_names,
                            runtime=runtime,
                        )
                    )
                except Exception as exc:
                    self._record_load_error(
                        skill_name=child.name,
                        path=child,
                        error=str(exc),
                        stage="discover-skill",
                    )
                    logger.warning("Failed to auto-load skill %s: %s", child, exc)

    def register_skill_tools(
        self,
        skill_name: str,
        skill_dir: Path,
        runtime: SkillRuntimeConfig | None = None,
        *,
        callable_tool_cls: Any,
    ) -> tuple[str, tuple[str, ...]]:
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.exists() or not scripts_dir.is_dir():
            return "", ()
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", skill_name.strip().lower()).strip("_")
        if not slug:
            slug = "skill"
        group_name = f"skill_{slug}"
        if group_name not in self._owner.groups:
            self._owner.groups[group_name] = ToolGroup(
                name=group_name,
                description=f"Tools from skill {skill_name}",
                notes=f"Skill tools for {skill_name}",
                active=False,
            )
        tool_names: list[str] = []
        for py_file in sorted(scripts_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            before_count = len(tool_names)
            try:
                ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
            except Exception as exc:
                self._record_load_error(
                    skill_name=skill_name,
                    path=py_file,
                    error=str(exc),
                    stage="parse-script",
                )
                logger.warning("Failed to parse skill script %s: %s", py_file, exc)
                continue
            specs = self.extract_function_specs_from_script(py_file)
            if specs:
                for spec in specs:
                    tool_name = f"{slug}__{spec.name}"
                    proxy = self._owner._build_external_skill_callable(
                        script_path=py_file,
                        function_name=spec.name,
                        runtime=runtime,
                    )
                    self._owner.registry.register(
                        callable_tool_cls(
                            func=proxy,
                            name=tool_name,
                            description=spec.description,
                            schema=spec.schema,
                            resource_manager=self._owner,
                        ),
                        group=group_name,
                    )
                    tool_names.append(tool_name)
            if len(tool_names) == before_count:
                cli_spec = self.extract_cli_tool_spec_from_script(py_file)
                if cli_spec is not None:
                    tool_name = f"{slug}__{cli_spec.name}"
                    self._owner.registry.register(
                        callable_tool_cls(
                            func=self._owner._build_external_cli_script_callable(
                                py_file, cli_spec, runtime=runtime
                            ),
                            name=tool_name,
                            description=cli_spec.description,
                            schema=cli_spec.schema,
                            resource_manager=self._owner,
                        ),
                        group=group_name,
                    )
                    tool_names.append(tool_name)
        return group_name, tuple(sorted(set(tool_names)))

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        if not text.startswith("---\n"):
            return {}, text.strip()
        end = text.find("\n---", 4)
        if end == -1:
            return {}, text.strip()
        header = text[4:end].strip()
        body = text[end + 4 :].strip()
        meta: dict[str, Any] = {}
        current_key: str | None = None
        list_buffer: list[str] = []

        def _flush_list_buffer() -> None:
            nonlocal current_key, list_buffer
            if current_key is not None and list_buffer:
                meta[current_key] = list_buffer[:]
            current_key = None
            list_buffer = []

        for raw_line in header.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if current_key is not None and stripped.startswith("- "):
                list_buffer.append(stripped[2:].strip().strip("'\""))
                continue
            _flush_list_buffer()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                meta[key] = value.strip("'\"")
            else:
                current_key = key
        _flush_list_buffer()
        return meta, body

    @staticmethod
    def _parse_frontmatter_list(raw_value: Any) -> tuple[str, ...]:
        if raw_value is None:
            return ()
        if isinstance(raw_value, (list, tuple, set)):
            items = [str(item).strip() for item in raw_value if str(item).strip()]
        else:
            items = [
                part.strip()
                for part in re.split(r"[,\n]+", str(raw_value))
                if part.strip()
            ]
        return tuple(dict.fromkeys(items))

    @classmethod
    def read_skill_document(cls, skill_dir: Path) -> tuple[dict[str, str], str, str]:
        skill_md = skill_dir / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        meta, body = cls.parse_frontmatter(text)
        prompt_body = body.strip()
        prompt_summary = cls.build_prompt_summary(prompt_body)
        return meta, prompt_summary, prompt_body

    @staticmethod
    def build_prompt_summary(prompt_body: str, max_chars: int = 1200) -> str:
        prompt = (prompt_body or "").strip()
        if len(prompt) <= max_chars:
            return prompt
        return (
            prompt[:max_chars].rstrip()
            + "\n\n[技能正文已省略；仅在强命中时再加载完整 SKILL.md]"
        )

    @staticmethod
    def skip_skill_function_name(name: str) -> bool:
        return name.startswith("_") or name in {
            "main",
            "parse_arguments",
            "create_client",
        }

    @classmethod
    def normalize_keywords(
        cls,
        raw_keywords: Any,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        values: list[str] = []
        if isinstance(raw_keywords, (list, tuple)):
            values.extend(str(x) for x in raw_keywords if str(x).strip())
        elif isinstance(raw_keywords, str) and raw_keywords.strip():
            values.extend([w for w in re.split(r"[,\n]+", raw_keywords) if w.strip()])
        values.extend(str(item) for item in fallback if str(item).strip())
        terms: set[str] = set()
        for value in values:
            terms.update(cls.tokenize(str(value)))
        return tuple(sorted(terms))

    @staticmethod
    def normalize_phrases(
        raw_keywords: Any,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        phrases: list[str] = []
        if isinstance(raw_keywords, (list, tuple)):
            phrases.extend(
                str(x).strip().lower() for x in raw_keywords if str(x).strip()
            )
        elif isinstance(raw_keywords, str) and raw_keywords.strip():
            phrases.extend(
                p.strip().lower()
                for p in re.split(r"[,\n]+", raw_keywords)
                if p.strip()
            )
        phrases.extend(str(x).strip().lower() for x in fallback if str(x).strip())
        normalized = []
        for phrase in phrases:
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase and len(phrase) >= 2:
                normalized.append(phrase)
        return tuple(sorted(set(normalized)))

    @staticmethod
    def tokenize(text: str) -> set[str]:
        text = (text or "").lower()
        tokens: set[str] = set()
        words = re.findall(r"[a-z0-9_]{2,}", text)
        tokens.update(words)
        for word in words:
            if len(word) >= 5:
                for i in range(0, len(word) - 2):
                    tokens.add(word[i : i + 3])
        for seg in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            tokens.add(seg)
            if len(seg) >= 2:
                for i in range(0, len(seg) - 1):
                    tokens.add(seg[i : i + 2])
        return {token for token in tokens if token}

    @classmethod
    def extract_function_specs_from_script(
        cls,
        script_path: Path,
    ) -> list[ScriptFunctionSpec]:
        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except Exception as exc:
            logger.warning("Failed to parse skill script %s: %s", script_path, exc)
            return []

        specs: list[ScriptFunctionSpec] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if cls.skip_skill_function_name(node.name):
                continue
            schema = cls.schema_from_ast_function(node)
            doc = ast.get_docstring(node) or node.name
            description = doc.splitlines()[0].strip() if doc else node.name
            specs.append(
                ScriptFunctionSpec(
                    name=node.name,
                    description=description or node.name,
                    schema=schema,
                )
            )
        return specs

    @classmethod
    def extract_cli_tool_spec_from_script(
        cls,
        script_path: Path,
    ) -> CliToolSpec | None:
        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except Exception as exc:
            logger.warning("Failed to parse CLI skill script %s: %s", script_path, exc)
            return None

        parse_func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        has_main = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "parse_arguments":
                    parse_func = node
                elif node.name == "main":
                    has_main = True
        if parse_func is None or not has_main:
            return None

        parser_names: set[str] = set()
        args: list[CliArgumentSpec] = []
        required: list[str] = []
        properties: dict[str, Any] = {}

        for node in ast.walk(parse_func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                func = call.func
                if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            parser_names.add(target.id)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "add_argument"
                and isinstance(func.value, ast.Name)
                and func.value.id in parser_names
            ):
                continue
            option_strings = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if not option_strings:
                continue
            flag = next(
                (item for item in option_strings if item.startswith("--")),
                option_strings[-1],
            )
            name = flag.lstrip("-").replace("-", "_")
            schema, is_required, action = cls.schema_from_argparse_call(node)
            properties[name] = schema
            args.append(
                CliArgumentSpec(
                    name=name,
                    flag=flag,
                    schema=schema,
                    required=is_required,
                    action=action,
                )
            )
            if is_required:
                required.append(name)

        if not args:
            return None

        return CliToolSpec(
            name=script_path.stem,
            description=f"Run CLI script {script_path.stem}",
            schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            arguments=tuple(args),
        )

    @classmethod
    def schema_from_argparse_call(
        cls,
        call: ast.Call,
    ) -> tuple[dict[str, Any], bool, str | None]:
        action: str | None = None
        arg_type: Any = None
        required = False
        for keyword in call.keywords:
            if keyword.arg == "action" and isinstance(keyword.value, ast.Constant):
                action = str(keyword.value.value)
            elif keyword.arg == "type":
                arg_type = cls.annotation_name_from_ast(keyword.value)
            elif keyword.arg == "required" and isinstance(keyword.value, ast.Constant):
                required = bool(keyword.value.value)
        if action == "store_true":
            return {"type": "boolean"}, False, action
        if arg_type == "int":
            schema = {"type": "integer"}
        elif arg_type == "float":
            schema = {"type": "number"}
        elif arg_type == "bool":
            schema = {"type": "boolean"}
        else:
            schema = {"type": "string"}
        return schema, required, action

    @staticmethod
    def annotation_name_from_ast(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @classmethod
    def schema_from_ast_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        defaults = list(node.args.defaults or [])
        plain_args = list(node.args.args or [])
        default_offset = len(plain_args) - len(defaults)
        for idx, arg in enumerate(plain_args):
            name = arg.arg
            if name in {"self", "context"}:
                continue
            properties[name] = cls.schema_for_ast_annotation(arg.annotation)
            if idx < default_offset:
                required.append(name)
        for kw_arg, kw_default in zip(
            node.args.kwonlyargs or [], node.args.kw_defaults or []
        ):
            name = kw_arg.arg
            if name in {"self", "context"}:
                continue
            properties[name] = cls.schema_for_ast_annotation(kw_arg.annotation)
            if kw_default is None:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def schema_for_ast_annotation(annotation: ast.AST | None) -> dict[str, Any]:
        if annotation is None:
            return {"type": "string"}

        def _base_name(node: ast.AST | None) -> str:
            if node is None:
                return ""
            if isinstance(node, ast.Name):
                return node.id.lower()
            if isinstance(node, ast.Attribute):
                return node.attr.lower()
            if isinstance(node, ast.Subscript):
                return _base_name(node.value)
            return ast.unparse(node).strip().lower()

        def _literal_type(node: ast.AST | None) -> str | None:
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool):
                    return "boolean"
                if isinstance(node.value, int) and not isinstance(node.value, bool):
                    return "integer"
                if isinstance(node.value, float):
                    return "number"
                if isinstance(node.value, str):
                    return "string"
            if isinstance(node, ast.Tuple):
                item_types = {
                    inferred
                    for inferred in (_literal_type(item) for item in node.elts)
                    if inferred is not None
                }
                if len(item_types) == 1:
                    return next(iter(item_types))
            return None

        base_name = _base_name(annotation)
        if base_name == "bool":
            return {"type": "boolean"}
        if base_name in {"int", "long"}:
            return {"type": "integer"}
        if base_name in {"float", "decimal"}:
            return {"type": "number"}
        if base_name in {"dict", "mapping"}:
            return {"type": "object"}
        if base_name in {"list", "tuple", "set", "sequence"}:
            return {"type": "array"}
        if base_name == "literal":
            inferred = _literal_type(getattr(annotation, "slice", None))
            if inferred is not None:
                return {"type": inferred}
        return {"type": "string"}
