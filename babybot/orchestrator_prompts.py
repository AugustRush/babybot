"""Application-layer orchestration prompts and configuration.

All Chinese-language content, domain-specific rules, NLU patterns, and
sub-task prompt templates live here — outside the kernel.

This module builds and returns an ``OrchestratorConfig`` instance that is
injected into ``DynamicOrchestrator`` at construction time.
"""

from __future__ import annotations

from typing import Any

from babybot.agent_kernel.orchestrator_config import OrchestratorConfig

# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是任务编排Agent，负责在最少步骤内调度子Agent完成任务并回复用户。\n\n"
    "核心规则：\n"
    "1. 简单问题直接 reply_to_user；需要外部能力时才 dispatch_task。\n"
    "2. 并行任务不要设 deps；有依赖的任务必须显式声明 deps。\n"
    "3. reply_to_user 必须单独调用且作为收尾；不能与其他工具混用。\n"
    "4. 禁止虚构结果；用户明确表达的执行限制与偏好必须优先遵守。\n"
    "5. 多资源任务优先在一次 dispatch_task 中用 resource_ids 组合能力；查看网页/仓库后创建或更新技能时，不要靠 create_worker 套娃补能力。\n"
    "6. 需要多Agent协作讨论、辩论、评审或头脑风暴时，使用 dispatch_team（debate模式）。\n"
    "7. 需要多Agent并行分工执行可拆分的复杂任务时，使用 dispatch_team（cooperative模式，需提供tasks列表）。\n"
    "8. 【发消息/通知用户/告知用户/发送报告】等动作无需子任务；reply_to_user 的内容即是发送给用户的消息，系统会自动投递到对应渠道。\n"
    "\n\n收敛规则（极其重要）：\n"
    "- 收到子任务结果后，先评估是否已充分回答用户问题。如果已充分，立即 reply_to_user，不要再 dispatch 新任务。\n"
    "- 同一意图不要用不同资源重复执行。例如专业 Skill 已成功返回结果，不要再用通用搜索/网页工具重复验证或补充。\n"
    "- 只有在结果明确不足、用户请求包含多个独立子目标、或当前结果无法满足用户需要时，才 dispatch 新任务。\n"
    "\n\n资源选择优先级：\n"
    "- 专业 Skill/MCP 优先于通用工具组。当资源列表中有与用户意图精确匹配的专业 Skill 或 MCP 时，优先使用；通用工具组（如 web、code）仅在无专业资源匹配时作为 fallback。\n"
    "- 不要为同一个子目标同时 dispatch 专业资源和通用资源。\n"
    "\n\n执行阶段协议：\n"
    "按以下四阶段有序推进，不允许跨阶段跳跃或在错误阶段执行动作：\n"
    "  [Research]     — 信息收集：仅读取、搜索、获取外部数据，不做修改或写入。\n"
    "  [Synthesis]    — 分析综合：基于 Research 阶段结果制定具体方案，不执行任何写入。\n"
    "  [Implementation] — 执行：按 Synthesis 方案执行写入、调用、修改等动作。\n"
    "  [Verification] — 验证：检查执行结果，与目标对比，发现问题则局部补救，完成后 reply_to_user。\n"
    "- 简单任务（直接问答、无外部调用）可省略 Research/Synthesis/Verification，直接 reply_to_user。\n"
    "- 每阶段的子任务描述中必须标注所在阶段，例如：[Research] 搜索相关文档。\n"
    "\n\n任务结果协议：\n"
    "- wait_for_tasks / get_task_result 返回 JSON，不是自由文本。\n"
    "- 当结果中的 reply_artifacts_ready=true 时，表示子任务已经产出可随最终回复自动附带给用户的附件/媒体。\n"
    "- 出现 reply_artifacts_ready=true 后，不要再创建专门的发送子任务；直接调用 reply_to_user 收尾。"
)

# ── Tool descriptions ────────────────────────────────────────────────────────

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "dispatch_task": (
        "创建一个子Agent任务并立即返回 task_id（非阻塞）。"
        "子Agent将使用指定资源执行任务。"
    ),
    "wait_for_tasks": (
        "等待指定任务完成并返回 JSON 结果映射（阻塞直到全部完成）。"
        "每个任务结果都包含 status/output/error，以及 reply_artifacts_ready 等字段。"
    ),
    "get_task_result": (
        "查询任务当前状态和结果（非阻塞，返回 JSON 对象）。"
        "结果包含 status/output/error，以及 reply_artifacts_ready 等字段。"
    ),
    "reply_to_user": (
        "向用户发送最终回复。调用后编排循环结束。"
        "此工具应作为最后一个工具调用单独使用，不与其他工具混用。"
        "宿主会自动附带当前已收集的产物/附件到最终回复，无需再创建专门的发送子任务。"
    ),
    "dispatch_team": (
        "启动一组Agent进行协作。支持两种模式：\n"
        "- debate（默认）：多轮辩论/评审/头脑风暴，Agent交替发言。\n"
        "- cooperative：任务分工协作，Agent从共享任务列表中领取任务并行执行，"
        "通过Mailbox广播结果给下游依赖。适用于可拆分为多个子任务的复杂工作。"
    ),
}

_TOOL_PARAM_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "dispatch_task": {
        "resource_id": "单个资源ID，必须来自可用资源列表",
        "resource_ids": "当一个子任务需要多种能力时，传入多个资源ID并合并使用",
        "description": "子任务的完整描述",
        "deps": "依赖的 task_id 列表，这些任务必须先完成",
        "timeout_s": "子任务超时时间（秒）。未提供时使用运行时默认超时。",
    },
    "wait_for_tasks": {
        "task_ids": "要等待的 task_id 列表",
    },
    "get_task_result": {
        "task_id": "要查询的 task_id",
    },
    "reply_to_user": {
        "text": "回复给用户的文本内容",
    },
    "dispatch_team": {
        "topic": "协作主题/高层目标描述",
        "agents": "参与协作的Agent列表，至少2个",
        "mode": "协作模式。debate=多轮辩论（默认），cooperative=任务分工（需配合 tasks 参数）",
        "max_rounds": "debate模式下的最大讨论轮数，默认5",
        "tasks": "cooperative模式下的任务列表。每个任务可声明依赖(deps)，Agent会自动领取可执行的任务并广播结果。",
        "resource_id": "可选：指定该Agent使用的资源ID",
        "skill_id": "可选：引用预定义的 skill name，自动继承其 role/description/prompt",
    },
}

# ── Deferred-task patterns ───────────────────────────────────────────────────

_DEFERRED_TASK_PATTERNS: tuple[str, ...] = (
    "两分钟后",
    "一分钟后",
    "稍后",
    "待会",
    "过会",
    "定时",
    "预约",
    "提醒我",
    "之后再",
)

_DEFERRED_TASK_GUIDANCE = (
    "\n\n延时/未来任务规则：\n"
    "7. 如果用户要求稍后、几分钟后、定时或未来某个时间执行动作，当前只应创建/更新定时任务，不要立刻执行未来动作。\n"
    "8. 未来一次性任务的描述必须自包含，写入定时任务时要包含届时需要完成的完整步骤，不能依赖当前这次对话还保存在上下文中。\n"
    "9. 定时任务若目标是【发送消息/推送通知/告知用户】，任务描述应写明需要回复的具体内容，届时直接 reply_to_user 即可，无需为发送行为创建子任务。"
)

# ── Resource catalog labels ───────────────────────────────────────────────────

_RESOURCE_CATALOG_HEADER = "\n可用资源：\n"
_RESOURCE_CATALOG_EMPTY = "\n可用资源：无"
_RESOURCE_CATALOG_LINE = "- {rid}: {name} — {purpose} (工具数: {tc}{preview_text})"
_RESOURCE_CATALOG_PREVIEW_PREFIX = "; 示例工具: "
_RESOURCE_CATALOG_SPECIALIST_HEADER = "  [专业能力] 优先使用：\n"
_RESOURCE_CATALOG_GENERAL_HEADER = "  [通用工具] 无专业资源匹配时使用：\n"

# ── Policy hints header ───────────────────────────────────────────────────────

_POLICY_HINTS_HEADER = "\n策略建议：\n- "

# ── Execution constraints wrapper ─────────────────────────────────────────────

_EXECUTION_CONSTRAINTS_WRAPPER = "[执行约束]\n{constraints}\n\n[用户请求]\n{goal}"
_ORIGINAL_REQUEST_HEADER = "--- 原始用户请求 ---"
_UPSTREAM_RESULTS_HEADER = "--- 上游任务结果 ---"

# ── Sub-task prompt ───────────────────────────────────────────────────────────

_CHILD_TASK_SENTINEL = "[执行型子任务]"

# Markers for detecting skill/repo maintenance tasks (triggers extra guidance).
_MAINTENANCE_MARKERS = (
    "查漏补缺",
    "补齐",
    "对比",
    "比较",
    "同步",
    "skill.md",
    "skill 文档",
    "github",
    "仓库",
    "repo",
    "文档",
    "技能",
)


def _is_maintenance_task(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _MAINTENANCE_MARKERS)


def _build_child_task_prompt(
    raw_description: str,
    original_goal: str,
    resource_ids: tuple[str, ...],
    upstream_results: dict[str, Any],
) -> str:
    resource_summary = ", ".join(resource_ids) if resource_ids else "-"
    maintenance_like = _is_maintenance_task(f"{raw_description}\n{original_goal}")

    upstream_lines: list[str] = []
    for tid, output in upstream_results.items():
        output_snippet = str(output or "").strip()
        if output_snippet:
            truncated = output_snippet[:300] + (
                "…" if len(output_snippet) > 300 else ""
            )
            upstream_lines.append(f"  [{tid}]: {truncated}")

    upstream_section: list[str] = []
    if upstream_lines:
        upstream_section = [
            "- upstream_results (上游依赖任务的输出摘要):"
        ] + upstream_lines

    expected_output_lines = [
        "- 返回当前子任务的执行结果摘要。",
        "- 若已生成文件/图片/音频/PDF 等产物，返回其绝对路径。",
        "- 如无法继续，明确说明缺少的输入、目标路径或阻塞原因。",
    ]
    done_when_lines = [
        "- 已完成当前子任务说明中的单一目标，或已明确指出无法继续的原因。",
        "- 若产物已生成且路径明确，立即返回；不要继续做无边界只读检查。",
        "- 不再需要额外派生任务或向用户追问即可交回主 agent。",
    ]
    if maintenance_like:
        expected_output_lines.insert(1, "- 若存在差异，列出差异项、证据和建议动作。")
        done_when_lines.insert(
            1, "- 已确认明确的目标文件；若无法确认，立即停止并返回缺口。"
        )

    lines = [
        "[执行型子任务]",
        "你是执行型子任务，不是任务编排器，也不是最终回复器。",
        f"原始子任务：{raw_description}",
        "[输入]",
        f"- parent_goal: {original_goal or '-'}",
        f"- resource_ids: {resource_summary}",
        *upstream_section,
        "[预期输出]",
        *expected_output_lines,
        "[完成条件]",
        *done_when_lines,
        "[禁止事项]",
        "- 不要创建或派生新的 worker。",
        "- 不要进入 team、讨论、评审或新的编排流程。",
        "- 不要直接向用户发送消息。",
        "- 不要在无明确目标时持续进行大范围扫描或无边界探索。",
    ]
    return "\n".join(lines)


# ── Honesty reminder ─────────────────────────────────────────────────────────

_ALL_TASKS_FAILED_REMINDER = (
    "[SYSTEM NOTICE] 所有子任务均已失败，没有任何成功的执行结果。\n"
    "失败任务: {dead_ids}\n"
    "错误摘要: {errors}\n"
    "你必须如实向用户报告任务失败及原因，禁止虚构或推断未执行的结果。"
)

# ── Step-budget fallback ──────────────────────────────────────────────────────

_STEP_BUDGET_EXHAUSTED_HEADER = "（编排步数已达上限，以下为已完成的任务结果）"
_STEP_BUDGET_SUCCEEDED_LINE = "- {task_id}: {output}"
_STEP_BUDGET_FAILED_LINE = "- {task_id}: 失败 — {error}"

# ── Team progress messages ────────────────────────────────────────────────────

_TEAM_DEBATE_STARTED = "已启动 {n_agents} 位专家讨论，最多 {max_rounds} 轮。"
_TEAM_DEBATE_ROUND = "第 {round_num}/{total_rounds} 轮讨论进行中。"
_TEAM_DEBATE_ENDED = "讨论结束"
_TEAM_COOP_STARTED = "已启动 {n_agents} 位专家协作执行 {n_tasks} 个任务。"
_TEAM_COOP_TASK_DONE = "任务 {task_id} 已完成 ({done}/{total})"
_TEAM_COOP_ENDED = "协作执行结束"

# ── Team default agent system prompt ─────────────────────────────────────────

_TEAM_DEFAULT_AGENT_SYSTEM_PROMPT = "你是讨论参与者。根据你的角色，针对主题发表观点。"

# ── NLU tokens for task-shape inference ──────────────────────────────────────

_MULTI_STEP_TOKENS: tuple[str, ...] = ("然后", "再", "并且", "同时", "先")
_PARALLEL_TOKENS: tuple[str, ...] = ("同时", "分别", "并行", "并且")


# ── Dynamic resource selection addendum ──────────────────────────────────────


def _build_resource_selection_addendum(briefs: list[dict[str, Any]]) -> str:
    """Generate a context-aware resource selection hint.

    Based on the mix of currently active specialist (skill/mcp) and general
    (tool_group) resources, produce a short addendum that helps the
    orchestrator make better dispatch decisions.

    Returns an empty string when no special guidance is needed.
    """
    specialist_types = {"skill", "mcp"}
    has_specialist = False
    has_general = False
    specialist_names: list[str] = []
    general_names: list[str] = []

    for b in briefs:
        if not b.get("active"):
            continue
        rtype = b.get("type", "")
        name = b.get("name", "?")
        if rtype in specialist_types:
            has_specialist = True
            specialist_names.append(name)
        else:
            has_general = True
            general_names.append(name)
    management_skills = {
        name
        for name in specialist_names
        if str(name).strip().lower() in {"skill-manager", "agent-admin"}
    }

    if not has_specialist or not has_general:
        # Only one tier present — no disambiguation needed.
        if not management_skills:
            return ""

    parts = [
        "\n资源选择提示：",
    ]
    if has_specialist and has_general:
        parts.extend(
            [
                f"当前同时存在专业资源（{', '.join(specialist_names[:5])}）"
                f"和通用工具组（{', '.join(general_names[:5])}）。",
                "请严格遵守以下选择顺序：",
                "1. 优先使用与用户意图匹配的专业 Skill 或 MCP 资源。",
                "2. 仅当没有匹配的专业资源时，才使用通用工具组（web、code 等）。",
                "3. 专业资源已成功返回结果后，禁止再用通用资源对同一子目标重复执行。",
            ]
        )
    if management_skills:
        parts.extend(
            [
                "4. `skill-manager`、`agent-admin` 这类管理型技能，只用于创建/安装/更新/启用/禁用/删除/reload 技能或修改助手配置。",
                "5. 用户说“使用某个技能/能力完成任务”时，是要调用目标领域能力，不是进入技能维护流程。",
            ]
        )
    return "\n".join(parts)


# ── Factory ───────────────────────────────────────────────────────────────────


def build_orchestrator_config() -> OrchestratorConfig:
    """Return an OrchestratorConfig populated with Chinese babybot prompts."""
    return OrchestratorConfig(
        system_prompt=_SYSTEM_PROMPT,
        tool_descriptions=_TOOL_DESCRIPTIONS,
        tool_param_descriptions=_TOOL_PARAM_DESCRIPTIONS,
        deferred_task_patterns=_DEFERRED_TASK_PATTERNS,
        deferred_task_guidance=_DEFERRED_TASK_GUIDANCE,
        resource_catalog_header=_RESOURCE_CATALOG_HEADER,
        resource_catalog_empty=_RESOURCE_CATALOG_EMPTY,
        resource_catalog_line=_RESOURCE_CATALOG_LINE,
        resource_catalog_preview_prefix=_RESOURCE_CATALOG_PREVIEW_PREFIX,
        resource_catalog_specialist_header=_RESOURCE_CATALOG_SPECIALIST_HEADER,
        resource_catalog_general_header=_RESOURCE_CATALOG_GENERAL_HEADER,
        policy_hints_header=_POLICY_HINTS_HEADER,
        execution_constraints_wrapper=_EXECUTION_CONSTRAINTS_WRAPPER,
        original_request_header=_ORIGINAL_REQUEST_HEADER,
        upstream_results_header=_UPSTREAM_RESULTS_HEADER,
        child_task_sentinel=_CHILD_TASK_SENTINEL,
        build_child_task_prompt=_build_child_task_prompt,
        is_maintenance_task=_is_maintenance_task,
        all_tasks_failed_reminder=_ALL_TASKS_FAILED_REMINDER,
        step_budget_exhausted_header=_STEP_BUDGET_EXHAUSTED_HEADER,
        step_budget_succeeded_line=_STEP_BUDGET_SUCCEEDED_LINE,
        step_budget_failed_line=_STEP_BUDGET_FAILED_LINE,
        team_debate_started_message=_TEAM_DEBATE_STARTED,
        team_debate_round_message=_TEAM_DEBATE_ROUND,
        team_debate_ended_message=_TEAM_DEBATE_ENDED,
        team_coop_started_message=_TEAM_COOP_STARTED,
        team_coop_task_done_message=_TEAM_COOP_TASK_DONE,
        team_coop_ended_message=_TEAM_COOP_ENDED,
        team_default_agent_system_prompt=_TEAM_DEFAULT_AGENT_SYSTEM_PROMPT,
        multi_step_tokens=_MULTI_STEP_TOKENS,
        parallel_tokens=_PARALLEL_TOKENS,
        build_resource_selection_addendum=_build_resource_selection_addendum,
    )
