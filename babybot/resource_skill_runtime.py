from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_kernel import SkillPack, SystemPromptBuilder, ToolLease


class ResourceSkillRuntime:
    MAX_MATCHED_SKILLS = 6

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def _match_score(self, skill: Any, task_description: str) -> int:
        query = (task_description or "").strip().lower()
        if not query:
            return 0
        skill_name = skill.name.strip().lower()
        resource_id = self._owner._skill_resource_id(skill).strip().lower()
        score = 0
        if f"${skill_name}" in query or resource_id in query:
            score += 100
        elif len(skill_name) >= 3 and skill_name in query:
            score += 50

        phrases = {
            phrase.strip().lower()
            for phrase in getattr(skill, "phrases", ())
            if str(phrase).strip()
        }
        score += 10 * sum(1 for phrase in phrases if phrase in query)

        query_terms = self._owner._tokenize(task_description)
        keywords = {
            str(keyword).strip().lower()
            for keyword in getattr(skill, "keywords", ())
            if str(keyword).strip()
        }
        score += sum(1 for keyword in keywords if keyword in query_terms)
        return score

    def _should_load_full_prompt(
        self,
        skill: Any,
        task_description: str,
        *,
        explicit: bool,
    ) -> bool:
        if explicit:
            return True
        query = (task_description or "").strip().lower()
        if not query:
            return False
        skill_name = skill.name.strip().lower()
        resource_id = self._owner._skill_resource_id(skill).strip().lower()
        return (
            f"${skill_name}" in query
            or resource_id in query
            or (len(skill_name) >= 3 and skill_name in query)
        )

    @staticmethod
    def _load_prompt_body_from_path(path: str) -> str:
        if not path:
            return ""
        skill_md = Path(path)
        try:
            text = skill_md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        if text.startswith("---\n"):
            end = text.find("\n---", 4)
            if end != -1:
                text = text[end + 4 :]
        return text.strip()

    def _resolve_skill_prompt(
        self,
        skill: Any,
        task_description: str,
        *,
        explicit: bool,
    ) -> str:
        if not self._should_load_full_prompt(
            skill,
            task_description,
            explicit=explicit,
        ):
            return skill.prompt
        if getattr(skill, "prompt_body", ""):
            return skill.prompt_body
        body = self._load_prompt_body_from_path(getattr(skill, "prompt_body_path", ""))
        if body:
            skill.prompt_body = body
            return body
        return skill.prompt

    async def select_skill_packs(
        self,
        task_description: str,
        skill_ids: list[str] | None = None,
    ) -> list[SkillPack]:
        active = [skill for skill in self._owner.skills.values() if skill.active]
        if not active:
            return []
        explicit = skill_ids is not None
        if explicit:
            wanted = {
                item.strip().lower()
                for item in skill_ids
                if isinstance(item, str) and item.strip()
            }
            active = [
                skill
                for skill in active
                if skill.name.strip().lower() in wanted
                or self._owner._skill_resource_id(skill) in wanted
            ]
        else:
            scored = [
                (self._match_score(skill, task_description), skill) for skill in active
            ]
            matched = [
                skill
                for score, skill in sorted(
                    scored,
                    key=lambda item: (-item[0], item[1].name.lower()),
                )
                if score > 0
            ]
            if matched:
                active = matched[: self.MAX_MATCHED_SKILLS]
        return [
            SkillPack(
                name=skill.name,
                system_prompt=self._resolve_skill_prompt(
                    skill,
                    task_description,
                    explicit=explicit,
                ),
                tool_lease=skill.lease,
            )
            for skill in active
        ]

    def build_worker_sys_prompt(
        self,
        agent_name: str,
        task_description: str,
        tools_text: str,
        selected_skill_packs: list[SkillPack],
        merged_lease: ToolLease | None = None,
    ) -> str:
        """Build a worker system prompt from composable sections.

        Uses SystemPromptBuilder so every logical block is a named, prioritized
        section — enabling observability, caching, and future per-task
        customization (Claude Code–style prompt-as-runtime).
        """
        builder = self._build_worker_prompt_sections(
            agent_name=agent_name,
            task_description=task_description,
            tools_text=tools_text,
            selected_skill_packs=selected_skill_packs,
            merged_lease=merged_lease,
        )
        return builder.build()

    def _build_worker_prompt_sections(
        self,
        agent_name: str,
        task_description: str,
        tools_text: str,
        selected_skill_packs: list[SkillPack],
        merged_lease: ToolLease | None = None,
    ) -> SystemPromptBuilder:
        """Assemble worker prompt as composable sections.

        Returns the SystemPromptBuilder so callers can inspect individual
        sections for observability, caching, or event emission.

        Task-type differentiation is structural (based on which tools are
        available), never keyword-matching on task content.
        """
        builder = SystemPromptBuilder()

        # ── Detect task class from available tools (structural, not semantic) ──
        task_class = _classify_task_from_tools(tools_text, merged_lease)

        # ── Section: Identity & task (always first) ──────────────────
        builder.add(
            "identity",
            f"你是 {agent_name}，请完成任务并直接输出最终答案。",
            priority=0,
            cacheable=True,
        )
        builder.add(
            "task",
            f"任务：{task_description}",
            priority=10,
        )

        # ── Section: Activated skills ────────────────────────────────
        selected_names = ", ".join(skill.name for skill in selected_skill_packs) or "无"
        builder.add(
            "active_skills",
            f"已激活技能（本次强相关）：{selected_names}",
            priority=20,
        )

        # ── Section: Skill catalog ───────────────────────────────────
        if merged_lease is not None:
            skill_catalog = self.format_skill_catalog_for_lease(
                merged_lease,
                max_items=24,
            )
        else:
            skill_catalog = self.format_skill_catalog(max_items=24)
        builder.add(
            "skill_catalog",
            f"可用技能目录（按需选择）：\n{skill_catalog}",
            priority=30,
        )

        # ── Section: Available tools ─────────────────────────────────
        builder.add(
            "tools",
            f"可用工具：{tools_text}",
            priority=40,
        )

        # ── Section: Execution rules (static, cacheable) ─────────────
        builder.add(
            "execution_rules",
            _EXECUTION_RULES,
            priority=50,
            cacheable=True,
        )

        # ── Section: Task-class-specific constraints ──────────────────
        # Each task class gets targeted guidance.  This section is absent
        # for pure text-gen workers (no tools → no extra constraints needed).
        task_specific = _TASK_CLASS_RULES.get(task_class, "")
        if task_specific:
            builder.add(
                f"task_class_rules:{task_class}",
                task_specific,
                priority=55,
                cacheable=True,
            )

        # ── Section: Skill maintenance rules ─────────────────────────
        builder.add(
            "skill_maintenance_rules",
            _SKILL_MAINTENANCE_RULES,
            priority=60,
            cacheable=True,
        )

        return builder

    def format_skill_catalog_for_lease(
        self,
        lease: ToolLease,
        max_items: int = 20,
    ) -> str:
        lease_groups = set(lease.include_groups)
        lease_tools = set(lease.include_tools)
        if not lease_groups and not lease_tools:
            return self.format_skill_catalog(max_items=max_items)

        accessible = []
        for skill in sorted(
            self._owner.skills.values(), key=lambda item: item.name.lower()
        ):
            if not skill.active:
                continue
            skill_lease = skill.lease or ToolLease()
            skill_groups = set(skill_lease.include_groups)
            skill_tools = set(skill_lease.include_tools)
            if not skill_groups and not skill_tools:
                accessible.append(skill)
            elif skill_groups & lease_groups:
                accessible.append(skill)
            elif skill_tools and (skill_tools & lease_tools):
                accessible.append(skill)

        if not accessible:
            return "- 无"
        lines: list[str] = []
        for idx, skill in enumerate(accessible, start=1):
            if idx > max_items:
                lines.append(f"- ... 还有 {len(accessible) - max_items} 个技能")
                break
            desc = skill.description.strip() or "无描述"
            skill_md = str((Path(skill.directory) / "SKILL.md").resolve())
            lines.append(
                f"- {skill.name}: {desc} [source={skill.source}, skill_md={skill_md}]"
            )
        return "\n".join(lines)

    def format_skill_catalog(self, max_items: int = 20) -> str:
        skills = [skill for skill in self._owner.skills.values() if skill.active]
        if not skills:
            return "- 无"
        lines: list[str] = []
        for idx, skill in enumerate(
            sorted(skills, key=lambda item: item.name.lower()),
            start=1,
        ):
            if idx > max_items:
                lines.append(f"- ... 还有 {len(skills) - max_items} 个技能")
                break
            desc = skill.description.strip() or "无描述"
            skill_md = str((Path(skill.directory) / "SKILL.md").resolve())
            lines.append(
                f"- {skill.name}: {desc} [source={skill.source}, skill_md={skill_md}]"
            )
        return "\n".join(lines)


# ── Static prompt section content ────────────────────────────────────────
# Extracted as module constants so they are defined once, cacheable, and
# easily testable in isolation.

_EXECUTION_RULES = """\
要求：
1. 如果任务是文本生成（写作、翻译、分析、总结、创意等），直接输出文本结果，不要调用任何工具。
2. 只有当任务明确需要外部操作（查询信息、生成图片、读写文件、执行代码等）时才调用工具。
3. 只执行任务说明中明确给出的目标，不要自行扩展目标或大范围探索。
4. 禁止编造工具执行结果、虚构文件路径，或把 output/ 当成技能源码目录。
5. 不要创建或派生新的 worker，不要把任务改写成讨论、评审或 team 流程。
6. 不要直接向用户发送消息；只把执行结果写入最终输出返回给主 agent。
7. 缺少输入、目标路径或权限时，立即返回失败原因，不要猜测。"""

_SKILL_MAINTENANCE_RULES = """\
技能维护规则：
8. 对代码或技能维护任务，优先使用明确目标的文件工具；少量定位后仍无目标时应立即停止并说明缺口。
9. 更新或删除现有 workspace 技能前，必须先检查目标技能是否存在，并查看 SKILL.md、skill.yaml、config.yaml 等配置文件。
10. 修改完成后必须 reload_skill，确认技能可重新加载。
11. 不要将 output/ 目录当成技能源码目录；技能代码在技能目录下，output/ 仅用于运行时产出物。"""


# ── Task-class-specific rules ─────────────────────────────────────────────
# Each class gets focused guidance; absent for text_gen (no tools = no extra
# constraints needed).  Classification is always structural (tool availability),
# never keyword-based on task content.

_TASK_CLASS_RULES: dict[str, str] = {
    "code": """\
代码执行约束：
- 执行代码前确认输入存在且合法；执行后验证输出符合预期。
- 不要在同一步骤中既执行又假设结果；每步等待实际返回值。
- 捕获并上报代码运行错误，不要掩盖异常或假装成功。""",
    "tool_action": """\
工具操作约束：
- 每次工具调用必须有明确目的；禁止探索性循环调用。
- 在写入或删除前先读取/确认目标存在。
- 工具失败时立即停止并将错误原因写入最终输出，不要重试无意义的相同调用。""",
    "skill_maintenance": """\
技能维护约束：
- 操作前必须先读取 SKILL.md 和 skill.yaml，确认技能结构。
- 修改后立即调用 reload_skill 验证技能可加载；验证失败时撤销或报告错误。
- 不要在没有读取现有内容的情况下直接覆写文件。""",
}


# ── Task classification (structural, never keyword-based) ─────────────────


# Tool name prefixes/suffixes that indicate code execution capability
_CODE_TOOL_MARKERS: frozenset[str] = frozenset(
    {
        "execute_code",
        "run_code",
        "execute_shell",
        "run_shell",
        "run_python",
        "execute_python",
        "bash",
        "shell",
    }
)

# Tool name substrings that indicate skill maintenance
_SKILL_MAINT_MARKERS: frozenset[str] = frozenset(
    {
        "reload_skill",
        "delete_skill",
        "workspace_write",
        "_workspace_write",
        "write_skill",
    }
)


def _classify_task_from_tools(
    tools_text: str,
    merged_lease: ToolLease | None,
) -> str:
    """Return a task class string based purely on available tools.

    Classes (in priority order):
      skill_maintenance → code → tool_action → text_gen (no tools)

    This is structural classification: we look at what tools the executor
    has access to, not at the natural-language task description.  This
    avoids fragile keyword matching while still enabling task-specific
    prompt differentiation.
    """
    if not tools_text or tools_text.strip() in ("无", ""):
        return "text_gen"

    # Normalise: split by comma/space, strip, lowercase
    tool_names_lower = {
        t.strip().lower()
        for part in tools_text.replace(",", " ").split()
        if (t := part.strip())
    }

    # Also check tool names from lease if available
    if merged_lease is not None:
        tool_names_lower |= {t.lower() for t in merged_lease.include_tools}

    # Skill maintenance tools take highest priority
    if any(
        marker in name for name in tool_names_lower for marker in _SKILL_MAINT_MARKERS
    ):
        return "skill_maintenance"

    # Code execution tools
    if any(
        any(name.startswith(marker) or name == marker for marker in _CODE_TOOL_MARKERS)
        for name in tool_names_lower
    ):
        return "code"

    # Any non-trivial tool set → generic tool_action
    return "tool_action"
