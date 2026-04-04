"""Budgeted context projections derived from a plan notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan_notebook import (
    NotebookEvent,
    NotebookNode,
    PlanNotebook,
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 3)


@dataclass(frozen=True)
class NotebookContextSection:
    name: str
    content: str
    priority: int


@dataclass(frozen=True)
class NotebookContextView:
    purpose: str
    token_budget: int
    sections: tuple[NotebookContextSection, ...]
    text: str
    selected_node_ids: tuple[str, ...] = ()
    raw_event_ids: tuple[str, ...] = ()
    omitted_event_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _render_section(title: str, lines: list[str]) -> str:
    cleaned = [line.strip() for line in lines if str(line).strip()]
    if not cleaned:
        return ""
    return f"[{title}]\n" + "\n".join(cleaned)


def _pick_dependency_lines(notebook: PlanNotebook, node: NotebookNode) -> list[str]:
    lines: list[str] = []
    for dep_id in node.deps:
        dep = notebook.nodes.get(dep_id)
        if dep is None:
            continue
        detail = dep.result_text or dep.latest_summary or dep.objective
        if detail:
            lines.append(f"- {dep_id}: {detail}")
    return lines


def _pick_decision_lines(notebook: PlanNotebook, node: NotebookNode) -> list[str]:
    decisions = list(node.decisions)
    for dep_id in node.deps:
        dep = notebook.nodes.get(dep_id)
        if dep is not None:
            decisions.extend(dep.decisions[-2:])
    root = notebook.nodes.get(notebook.root_node_id)
    if root is not None and root.node_id != node.node_id:
        decisions.extend(root.decisions[-2:])
    deduped: list[str] = []
    seen: set[str] = set()
    for item in reversed(decisions):
        text = item.summary.strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(f"- {text}")
    deduped.reverse()
    return deduped[:5]


def _pick_blocker_lines(node: NotebookNode) -> list[str]:
    lines: list[str] = []
    for issue in node.issues:
        if issue.status == "open":
            lines.append(f"- {issue.title}: {issue.detail}".rstrip(": "))
    for checkpoint in node.checkpoints:
        if checkpoint.status == "open":
            lines.append(f"- [{checkpoint.kind}] {checkpoint.message}")
    return lines[:5]


def _pick_artifact_lines(node: NotebookNode) -> list[str]:
    lines: list[str] = []
    for artifact in node.artifacts[-5:]:
        label = artifact.label or artifact.kind
        lines.append(f"- {label}: {artifact.path}")
    return lines


def _pick_excerpt_lines(node: NotebookNode) -> tuple[list[str], tuple[str, ...], int]:
    preferred = {
        "observation",
        "decision",
        "issue",
        "summary",
        "checkpoint",
        "status",
        "artifact",
    }
    excerpts: list[str] = []
    event_ids: list[str] = []
    omitted = 0
    for event in reversed(node.events):
        if event.kind not in preferred:
            omitted += 1
            continue
        if event.detail:
            excerpts.append(f"- {event.summary}: {event.detail}")
        else:
            excerpts.append(f"- {event.summary}")
        event_ids.append(event.event_id)
        if len(excerpts) >= 3:
            break
    excerpts.reverse()
    event_ids.reverse()
    return excerpts, tuple(event_ids), omitted


def _build_sections(
    *,
    purpose: str,
    notebook: PlanNotebook,
    current_node: NotebookNode,
    token_budget: int,
    include_completion_summary: bool = False,
) -> NotebookContextView:
    candidate_sections: list[NotebookContextSection] = []
    raw_event_ids: tuple[str, ...] = ()
    omitted_events = 0

    goal_section = _render_section(
        "Goal",
        [
            f"- {notebook.goal}",
            *(
                [f"- hard_constraints: {item}" for item in notebook.metadata.get("hard_constraints", [])]
                if isinstance(notebook.metadata.get("hard_constraints"), (list, tuple))
                else []
            ),
        ],
    )
    if goal_section:
        candidate_sections.append(
            NotebookContextSection("goal", goal_section, priority=0)
        )

    current_section = _render_section(
        "Current Step",
        [
            f"- node_id: {current_node.node_id}",
            f"- title: {current_node.title}",
            f"- objective: {current_node.objective}",
            f"- status: {current_node.status}",
        ],
    )
    candidate_sections.append(
        NotebookContextSection("current_step", current_section, priority=10)
    )

    deps_lines = _pick_dependency_lines(notebook, current_node)
    if deps_lines:
        candidate_sections.append(
            NotebookContextSection(
                "dependencies",
                _render_section("Direct Dependencies", deps_lines),
                priority=20,
            )
        )

    decision_lines = _pick_decision_lines(notebook, current_node)
    if decision_lines:
        candidate_sections.append(
            NotebookContextSection(
                "decisions",
                _render_section("Verified Decisions", decision_lines),
                priority=30,
            )
        )

    blocker_lines = _pick_blocker_lines(current_node)
    if blocker_lines:
        candidate_sections.append(
            NotebookContextSection(
                "blockers",
                _render_section("Blockers", blocker_lines),
                priority=40,
            )
        )

    artifact_lines = _pick_artifact_lines(current_node)
    if artifact_lines:
        candidate_sections.append(
            NotebookContextSection(
                "artifacts",
                _render_section("Artifacts", artifact_lines),
                priority=50,
            )
        )

    if include_completion_summary and notebook.completion_summary:
        completion_lines: list[str] = []
        final_summary = str(notebook.completion_summary.get("final_summary", "") or "").strip()
        if final_summary:
            completion_lines.append(f"- summary: {final_summary}")
        for item in notebook.completion_summary.get("decision_register", []) or []:
            completion_lines.append(f"- decision: {item}")
        for item in notebook.completion_summary.get("artifact_manifest", []) or []:
            completion_lines.append(f"- artifact: {item}")
        if completion_lines:
            candidate_sections.append(
                NotebookContextSection(
                    "completion_summary",
                    _render_section("Completion Summary", completion_lines),
                    priority=15,
                )
            )

    excerpt_lines, raw_event_ids, omitted_events = _pick_excerpt_lines(current_node)
    if excerpt_lines:
        candidate_sections.append(
            NotebookContextSection(
                "excerpts",
                _render_section("Relevant Excerpts", excerpt_lines),
                priority=60,
            )
        )

    selected: list[NotebookContextSection] = []
    used_tokens = 0
    for section in sorted(candidate_sections, key=lambda item: item.priority):
        section_tokens = _estimate_tokens(section.content)
        if selected and used_tokens + section_tokens > token_budget:
            continue
        selected.append(section)
        used_tokens += section_tokens

    text = "\n\n".join(section.content for section in selected if section.content)
    return NotebookContextView(
        purpose=purpose,
        token_budget=token_budget,
        sections=tuple(selected),
        text=text,
        selected_node_ids=(current_node.node_id,),
        raw_event_ids=raw_event_ids,
        omitted_event_count=omitted_events,
        metadata={"used_tokens": used_tokens},
    )


def build_orchestrator_context_view(
    notebook: PlanNotebook,
    token_budget: int = 2400,
    current_node_id: str = "",
) -> NotebookContextView:
    frontier_ids = notebook.frontier_node_ids()
    node_id = (
        current_node_id
        if current_node_id and current_node_id in notebook.nodes
        else (frontier_ids[0] if frontier_ids else notebook.root_node_id)
    )
    return _build_sections(
        purpose="orchestrator",
        notebook=notebook,
        current_node=notebook.get_node(node_id),
        token_budget=token_budget,
    )


def build_worker_context_view(
    notebook: PlanNotebook,
    node_id: str,
    token_budget: int = 2200,
) -> NotebookContextView:
    return _build_sections(
        purpose="worker",
        notebook=notebook,
        current_node=notebook.get_node(node_id),
        token_budget=token_budget,
    )


def build_completion_context_view(
    notebook: PlanNotebook,
    token_budget: int = 2200,
) -> NotebookContextView:
    node_id = notebook.root_node_id
    return _build_sections(
        purpose="completion",
        notebook=notebook,
        current_node=notebook.get_node(node_id),
        token_budget=token_budget,
        include_completion_summary=True,
    )


def search_notebook_text(
    notebook: PlanNotebook,
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip().lower()
    if not normalized_query:
        return []
    matches: list[tuple[int, float, dict[str, Any]]] = []
    for node in notebook.nodes.values():
        for event in node.events:
            search_text = " ".join(
                part for part in (event.summary, event.detail) if str(part).strip()
            ).lower()
            if normalized_query not in search_text:
                continue
            score = search_text.count(normalized_query)
            matches.append(
                (
                    score,
                    event.created_at,
                    {
                        "node_id": node.node_id,
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "summary": event.summary,
                        "detail": event.detail,
                    },
                )
            )
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [payload for _, _, payload in matches[:limit]]
