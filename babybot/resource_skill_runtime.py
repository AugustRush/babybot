from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_kernel import SkillPack, ToolLease


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
        selected_names = ", ".join(skill.name for skill in selected_skill_packs) or "无"
        if merged_lease is not None:
            skill_catalog = self.format_skill_catalog_for_lease(
                merged_lease,
                max_items=24,
            )
        else:
            skill_catalog = self.format_skill_catalog(max_items=24)
        lines = [
            f"你是 {agent_name}，请完成任务并直接输出最终答案。",
            f"任务：{task_description}",
            f"已激活技能（本次强相关）：{selected_names}",
            f"可用技能目录（按需选择）：\n{skill_catalog}",
            f"可用工具：{tools_text}",
            "要求：",
            "1. 如果任务是文本生成（写作、翻译、分析、总结、创意等），直接输出文本结果，不要调用任何工具。",
            "2. 只有当任务明确需要外部操作（查询信息、生成图片、读写文件、执行代码等）时才调用工具。",
            "3. 只执行任务说明中明确给出的目标，不要自行扩展目标或大范围探索。",
            "4. 禁止编造工具执行结果、虚构文件路径，或把 output/ 当成技能源码目录。",
            "5. 不要创建或派生新的 worker，不要把任务改写成讨论、评审或 team 流程。",
            "6. 不要直接向用户发送消息；只把执行结果写入最终输出返回给主 agent。",
            "7. 缺少输入、目标路径或权限时，立即返回失败原因，不要猜测。",
            "8. 对代码或技能维护任务，优先使用明确目标的文件工具；少量定位后仍无目标时应立即停止并说明缺口。",
            "9. 更新或删除现有 workspace 技能前，必须先检查目标技能是否存在，并查看 SKILL.md。",
            "10. 修改完成后必须 reload_skill，确认技能可重新加载。",
        ]
        return "\n".join(lines)

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
