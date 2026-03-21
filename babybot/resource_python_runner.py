from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .resource_models import SkillRuntimeConfig


class ExternalPythonRunner:
    def __init__(self, config: Any) -> None:
        self._config = config

    @staticmethod
    def normalize_string_list(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            parts = re.split(r"[\n,]", value)
        elif isinstance(value, (list, tuple, set)):
            parts = [str(item) for item in value]
        else:
            parts = [str(value)]
        items = [item.strip() for item in parts if str(item).strip()]
        return tuple(dict.fromkeys(items))

    @classmethod
    def build_skill_runtime(cls, raw: dict[str, Any] | None = None) -> SkillRuntimeConfig:
        payload = raw or {}
        return SkillRuntimeConfig(
            python_executable=str(payload.get("python_executable", "") or "").strip(),
            python_fallback_executables=cls.normalize_string_list(
                payload.get("python_fallback_executables")
            ),
            python_required_modules=cls.normalize_string_list(
                payload.get("python_required_modules")
            ),
        )

    @staticmethod
    def is_venv_python(path: str) -> bool:
        normalized = str(path).replace("\\", "/").lower()
        return "/.venv/" in normalized or "/venv/" in normalized

    @staticmethod
    def format_cli_argument(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def discover_host_python_candidates(self) -> list[str]:
        candidates: list[str] = []
        for name in ("python3", "python"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
        candidates.extend(
            [
                "/usr/bin/python3",
                "/usr/local/bin/python3",
                "/opt/homebrew/bin/python3",
            ]
        )
        if not self.is_venv_python(sys.executable):
            candidates.append(sys.executable)
        return candidates

    def get_python_candidates(
        self,
        runtime: SkillRuntimeConfig | None = None,
    ) -> list[dict[str, Any]]:
        skill_runtime = runtime or SkillRuntimeConfig()
        system_conf = getattr(self._config, "system", None)
        configured = getattr(system_conf, "python_executable", "") or ""
        system_fallbacks = self.normalize_string_list(
            getattr(system_conf, "python_fallback_executables", ())
        )
        required_modules = tuple(skill_runtime.python_required_modules)
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []

        def _add(executable: str, source: str) -> None:
            value = (executable or "").strip()
            if not value:
                return
            resolved = value
            if not any(sep in value for sep in (os.sep, "/", "\\")):
                resolved = shutil.which(value) or value
            if self.is_venv_python(resolved) or resolved in seen:
                return
            seen.add(resolved)
            candidates.append(
                {
                    "executable": resolved,
                    "required_modules": required_modules,
                    "source": source,
                }
            )

        _add(skill_runtime.python_executable, "skill.python_executable")
        for executable in skill_runtime.python_fallback_executables:
            _add(executable, "skill.python_fallback_executables")
        _add(configured, "system.python_executable")
        for executable in system_fallbacks:
            _add(executable, "system.python_fallback_executables")
        for executable in self.discover_host_python_candidates():
            _add(executable, "auto")
        return candidates

    def get_user_python(self) -> str:
        candidates = self.get_python_candidates()
        if candidates:
            return candidates[0]["executable"]
        return sys.executable

    @staticmethod
    def probe_python_candidate(
        candidate: dict[str, Any],
        *,
        cache: dict[tuple[str, tuple[str, ...], str], str | None],
        get_active_write_root: Any,
        clean_env: Any,
    ) -> str | None:
        executable = str(candidate.get("executable", "") or "").strip()
        required_modules = tuple(candidate.get("required_modules") or ())
        cache_scope = str(candidate.get("cache_scope", "") or "")
        cache_key = (executable, required_modules, cache_scope)
        if cache_key in cache:
            return cache[cache_key]
        if not executable:
            cache[cache_key] = "empty python executable"
            return cache[cache_key]
        if any(sep in executable for sep in (os.sep, "/", "\\")):
            if not (os.path.isfile(executable) and os.access(executable, os.X_OK)):
                cache[cache_key] = f"python executable not found: {executable}"
                return cache[cache_key]
        elif shutil.which(executable) is None:
            cache[cache_key] = f"python executable not found in PATH: {executable}"
            return cache[cache_key]
        if not required_modules:
            cache[cache_key] = None
            return None
        probe = (
            "import importlib.util, sys\n"
            "missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]\n"
            "if missing:\n"
            "    raise SystemExit('missing modules: ' + ', '.join(missing))\n"
        )
        try:
            proc = subprocess.run(
                [executable, "-c", probe, *required_modules],
                cwd=str(get_active_write_root()),
                env=clean_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            cache[cache_key] = str(exc)
            return cache[cache_key]
        if proc.returncode != 0:
            cache[cache_key] = (
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"probe exit code {proc.returncode}"
            )
            return cache[cache_key]
        cache[cache_key] = None
        return None

    @staticmethod
    def mark_python_candidate_unhealthy(
        candidate: dict[str, Any],
        detail: str,
        *,
        cache: dict[tuple[str, tuple[str, ...], str], str | None],
    ) -> None:
        executable = str(candidate.get("executable", "") or "").strip()
        required_modules = tuple(candidate.get("required_modules") or ())
        cache_scope = str(candidate.get("cache_scope", "") or "")
        cache[(executable, required_modules, cache_scope)] = detail.strip()

    @staticmethod
    def is_environment_failure(
        detail: str,
        *,
        returncode: int | None = None,
        payload_missing: bool = False,
    ) -> bool:
        text = (detail or "").strip().lower()
        if any(
            token in text
            for token in (
                "no module named",
                "modulenotfounderror",
                "importerror",
                "python executable not found",
                "library not loaded",
                "dyld",
                "segmentation fault",
                "abort trap",
                "illegal instruction",
                "nsrangeexception",
                "core dumped",
                "killed",
            )
        ):
            return True
        if returncode is not None and returncode < 0:
            return True
        if payload_missing and returncode not in (None, 0):
            return True
        return False

    @staticmethod
    def parse_progress_line(line: str) -> tuple[str, float | None]:
        text = (line or "").strip()
        if not text:
            return "", None
        if text.startswith("__BABYBOT_RESULT__"):
            return "", None
        progress_match = re.search(r"(\d{1,3})%", text)
        progress = None
        if progress_match is not None:
            progress = max(0.0, min(1.0, int(progress_match.group(1)) / 100.0))
            text = re.sub(r"\s*\(?\d{1,3}%\)?", "", text).strip(" -:")
        return text[:80], progress

    @staticmethod
    async def _collect_process_output(
        proc: Any,
        *,
        timeout_s: float,
        on_output: Any = None,
    ) -> tuple[bytes, bytes]:
        stdout_stream = getattr(proc, "stdout", None)
        stderr_stream = getattr(proc, "stderr", None)
        if not callable(getattr(stdout_stream, "readline", None)) or not callable(
            getattr(stderr_stream, "readline", None)
        ):
            return await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        async def _drain(stream: Any, chunks: list[bytes], stream_name: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                chunks.append(line)
                if on_output is not None:
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text:
                        on_output(text, stream=stream_name)

        drain_task = asyncio.gather(
            _drain(stdout_stream, stdout_chunks, "stdout"),
            _drain(stderr_stream, stderr_chunks, "stderr"),
        )
        try:
            await asyncio.wait_for(drain_task, timeout=timeout_s)
            wait = getattr(proc, "wait", None)
            if callable(wait):
                await asyncio.wait_for(wait(), timeout=max(0.1, timeout_s))
        except asyncio.TimeoutError:
            drain_task.cancel()
            with contextlib.suppress(Exception):
                await drain_task
            raise
        return b"".join(stdout_chunks), b"".join(stderr_chunks)

    def build_external_cli_script_callable(
        self,
        owner: Any,
        script_path: Path,
        cli_spec: Any,
        runtime: SkillRuntimeConfig | None = None,
    ) -> Any:
        resolved = str(script_path.resolve())
        arguments = cli_spec.arguments

        async def _runner(**kwargs: Any) -> str:
            argv_tail = [resolved]
            for spec in arguments:
                if spec.name not in kwargs:
                    continue
                value = kwargs.get(spec.name)
                if spec.action == "store_true":
                    if value:
                        argv_tail.append(spec.flag)
                    continue
                if value is None:
                    continue
                argv_tail.extend([spec.flag, self.format_cli_argument(value)])
            timeout_s = owner._coerce_timeout(kwargs.get("timeout"), default=300.0)
            attempts: list[str] = []
            for candidate in owner._get_python_candidates(runtime):
                scoped_candidate = dict(candidate)
                scoped_candidate["cache_scope"] = resolved
                probe_error = await asyncio.to_thread(
                    owner._probe_python_candidate,
                    scoped_candidate,
                )
                if probe_error:
                    attempts.append(f"{candidate['executable']}: {probe_error}")
                    continue
                try:
                    proc = await asyncio.create_subprocess_exec(
                        candidate["executable"],
                        *argv_tail,
                        cwd=str(owner._get_active_write_root()),
                        env=owner._clean_env(),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except OSError as exc:
                    detail = str(exc)
                    owner._mark_python_candidate_unhealthy(scoped_candidate, detail)
                    attempts.append(f"{candidate['executable']}: {detail}")
                    continue
                try:
                    stdout, stderr = await self._collect_process_output(
                        proc,
                        timeout_s=timeout_s,
                        on_output=owner._report_external_process_output,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.communicate()
                    raise RuntimeError(f"CLI tool timeout after {timeout_s}s")
                out_text = (stdout or b"").decode("utf-8", errors="ignore").strip()
                err_text = (stderr or b"").decode("utf-8", errors="ignore").strip()
                if proc.returncode != 0:
                    detail = err_text or out_text or f"exit code {proc.returncode}"
                    if owner._is_environment_failure(
                        detail,
                        returncode=proc.returncode,
                        payload_missing=True,
                    ):
                        owner._mark_python_candidate_unhealthy(
                            scoped_candidate, detail
                        )
                        attempts.append(f"{candidate['executable']}: {detail}")
                        continue
                    raise RuntimeError(detail)
                return out_text or err_text
            if attempts:
                raise RuntimeError(
                    "No healthy Python runtime succeeded. "
                    + " | ".join(attempts[-3:])
                )
            raise RuntimeError("No Python runtime candidate available")

        return _runner

    def build_external_skill_callable(
        self,
        owner: Any,
        script_path: Path,
        function_name: str,
        runtime: SkillRuntimeConfig | None = None,
    ) -> Any:
        resolved = str(script_path.resolve())

        async def _runner(**kwargs: Any) -> str:
            return await owner._invoke_external_skill_function(
                script_path=resolved,
                function_name=function_name,
                arguments=kwargs,
                runtime=runtime,
            )

        return _runner

    async def invoke_external_skill_function(
        self,
        owner: Any,
        *,
        script_path: str,
        function_name: str,
        arguments: dict[str, Any],
        runtime: SkillRuntimeConfig | None = None,
        result_normalizer: Any,
    ) -> str:
        runner = (
            "import asyncio, importlib.util, inspect, json, sys, traceback\n"
            "MARK='__BABYBOT_RESULT__'\n"
            "script, fn, raw = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "try:\n"
            "    spec = importlib.util.spec_from_file_location('babybot_skill_proxy', script)\n"
            "    if spec is None or spec.loader is None:\n"
            "        raise RuntimeError(f'cannot load script: {script}')\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(mod)\n"
            "    if not hasattr(mod, fn):\n"
            "        raise AttributeError(f'function not found: {fn}')\n"
            "    func = getattr(mod, fn)\n"
            "    kwargs = json.loads(raw) if raw else {}\n"
            "    if inspect.iscoroutinefunction(func):\n"
            "        out = asyncio.run(func(**kwargs))\n"
            "    else:\n"
            "        out = func(**kwargs)\n"
            "    print(MARK + json.dumps({'ok': True, 'result': out}, ensure_ascii=False, default=str))\n"
            "except Exception as exc:\n"
            "    traceback.print_exc()\n"
            "    print(MARK + json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False))\n"
        )
        args_json = json.dumps(arguments or {}, ensure_ascii=False)
        timeout_s = owner._coerce_timeout(arguments.get("timeout"), default=300.0)
        attempts: list[str] = []
        last_tool_error = "Tool error: external execution failed."
        for candidate in owner._get_python_candidates(runtime):
            scoped_candidate = dict(candidate)
            scoped_candidate["cache_scope"] = script_path
            probe_error = await asyncio.to_thread(
                owner._probe_python_candidate,
                scoped_candidate,
            )
            if probe_error:
                attempts.append(f"{candidate['executable']}: {probe_error}")
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    candidate["executable"],
                    "-c",
                    runner,
                    script_path,
                    function_name,
                    args_json,
                    cwd=str(owner._get_active_write_root()),
                    env=owner._clean_env(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                detail = str(exc)
                owner._mark_python_candidate_unhealthy(scoped_candidate, detail)
                attempts.append(f"{candidate['executable']}: {detail}")
                last_tool_error = f"Tool error: {detail}"
                continue

            try:
                if timeout_s and timeout_s > 0:
                    stdout, stderr = await self._collect_process_output(
                        proc,
                        timeout_s=timeout_s,
                        on_output=owner._report_external_process_output,
                    )
                else:
                    stdout, stderr = await self._collect_process_output(
                        proc,
                        timeout_s=24 * 3600,
                        on_output=owner._report_external_process_output,
                    )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.communicate()
                except Exception:
                    pass
                return f"Tool error: execution timeout after {timeout_s}s."
            except Exception:
                raise

            out_text = (stdout or b"").decode("utf-8", errors="ignore")
            err_text = (stderr or b"").decode("utf-8", errors="ignore")
            marker = "__BABYBOT_RESULT__"
            payload_line = ""
            for line in reversed(out_text.splitlines()):
                if line.startswith(marker):
                    payload_line = line[len(marker) :].strip()
                    break
            if not payload_line:
                combined = (
                    out_text.strip()
                    + ("\n" + err_text.strip() if err_text.strip() else "")
                ).strip()
                detail = combined or "no result returned"
                last_tool_error = f"Tool error: {detail}"
                if owner._is_environment_failure(
                    detail,
                    returncode=proc.returncode,
                    payload_missing=True,
                ):
                    owner._mark_python_candidate_unhealthy(scoped_candidate, detail)
                    attempts.append(f"{candidate['executable']}: {detail}")
                    continue
                return last_tool_error
            try:
                payload = json.loads(payload_line)
            except json.JSONDecodeError:
                return payload_line
            if not payload.get("ok", False):
                detail = str(payload.get("error", "external execution failed"))
                last_tool_error = f"Tool error: {detail}"
                if owner._is_environment_failure(
                    detail,
                    returncode=proc.returncode,
                    payload_missing=False,
                ):
                    owner._mark_python_candidate_unhealthy(scoped_candidate, detail)
                    attempts.append(f"{candidate['executable']}: {detail}")
                    continue
                return last_tool_error
            return result_normalizer(payload.get("result"))

        if attempts:
            return (
                "Tool error: no healthy Python runtime succeeded. "
                + " | ".join(attempts[-3:])
            )
        return last_tool_error
