"""Canonical runtime notebook for complex orchestration work."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


NotebookNodeStatus = Literal[
    "pending",
    "running",
    "waiting",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]

NotebookEventKind = Literal[
    "observation",
    "decision",
    "tool_call",
    "tool_result",
    "status",
    "checkpoint",
    "artifact",
    "issue",
    "summary",
]

NotebookCheckpointStatus = Literal["open", "resolved", "dismissed"]

_TERMINAL_NODE_STATUSES = {"completed", "failed", "cancelled"}
_ALLOWED_STATUS_TRANSITIONS: dict[NotebookNodeStatus, set[NotebookNodeStatus]] = {
    "pending": {"running", "waiting", "blocked", "completed", "failed", "cancelled"},
    "running": {"waiting", "blocked", "completed", "failed", "cancelled"},
    "waiting": {"running", "blocked", "completed", "failed", "cancelled"},
    "blocked": {"running", "waiting", "completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


def _now_ts() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class NotebookEvent:
    event_id: str
    node_id: str
    kind: NotebookEventKind
    summary: str
    detail: str = ""
    created_at: float = field(default_factory=_now_ts)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NotebookArtifact:
    artifact_id: str
    node_id: str
    path: str
    kind: str = "file"
    label: str = ""
    created_at: float = field(default_factory=_now_ts)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NotebookIssue:
    issue_id: str
    node_id: str
    title: str
    detail: str = ""
    status: str = "open"
    severity: str = "normal"
    created_at: float = field(default_factory=_now_ts)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NotebookDecision:
    decision_id: str
    node_id: str
    summary: str
    rationale: str = ""
    status: str = "verified"
    created_at: float = field(default_factory=_now_ts)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NotebookCheckpoint:
    checkpoint_id: str
    node_id: str
    kind: str
    message: str
    status: NotebookCheckpointStatus = "open"
    created_at: float = field(default_factory=_now_ts)
    resolved_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> None:
        self.status = "resolved"
        self.resolved_at = _now_ts()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NotebookNode:
    node_id: str
    kind: str
    title: str
    objective: str
    parent_id: str | None = None
    owner: str = ""
    resource_ids: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    status: NotebookNodeStatus = "pending"
    created_at: float = field(default_factory=_now_ts)
    updated_at: float = field(default_factory=_now_ts)
    latest_summary: str = ""
    result_text: str = ""
    error_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[NotebookEvent] = field(default_factory=list)
    artifacts: list[NotebookArtifact] = field(default_factory=list)
    issues: list[NotebookIssue] = field(default_factory=list)
    decisions: list[NotebookDecision] = field(default_factory=list)
    checkpoints: list[NotebookCheckpoint] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now_ts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "title": self.title,
            "objective": self.objective,
            "parent_id": self.parent_id,
            "owner": self.owner,
            "resource_ids": list(self.resource_ids),
            "deps": list(self.deps),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_summary": self.latest_summary,
            "result_text": self.result_text,
            "error_text": self.error_text,
            "metadata": dict(self.metadata),
            "events": [item.to_dict() for item in self.events],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "issues": [item.to_dict() for item in self.issues],
            "decisions": [item.to_dict() for item in self.decisions],
            "checkpoints": [item.to_dict() for item in self.checkpoints],
        }


@dataclass
class PlanNotebook:
    notebook_id: str
    goal: str
    flow_id: str = ""
    plan_id: str = ""
    root_node_id: str = ""
    created_at: float = field(default_factory=_now_ts)
    updated_at: float = field(default_factory=_now_ts)
    metadata: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, NotebookNode] = field(default_factory=dict)
    raw_events: list[NotebookEvent] = field(default_factory=list)
    completion_summary: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _now_ts()

    def get_node(self, node_id: str) -> NotebookNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown notebook node: {node_id}") from exc

    def add_child_node(
        self,
        *,
        parent_id: str,
        kind: str,
        title: str,
        objective: str,
        owner: str = "",
        resource_ids: tuple[str, ...] = (),
        deps: tuple[str, ...] = (),
        status: NotebookNodeStatus = "pending",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookNode:
        if parent_id not in self.nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        node = NotebookNode(
            node_id=_new_id("node"),
            kind=str(kind or "").strip() or "task",
            title=str(title or "").strip() or str(objective or "").strip() or "Task",
            objective=str(objective or "").strip(),
            parent_id=parent_id,
            owner=str(owner or "").strip(),
            resource_ids=tuple(str(item).strip() for item in resource_ids if str(item).strip()),
            deps=tuple(str(item).strip() for item in deps if str(item).strip()),
            status=status,
            metadata=dict(metadata or {}),
        )
        self.nodes[node.node_id] = node
        self.touch()
        return node

    def transition_node(
        self,
        node_id: str,
        status: NotebookNodeStatus,
        *,
        summary: str = "",
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookNode:
        node = self.get_node(node_id)
        normalized_status = status
        if normalized_status != node.status:
            allowed = _ALLOWED_STATUS_TRANSITIONS.get(node.status, set())
            if normalized_status not in allowed:
                raise ValueError(
                    f"invalid notebook status transition: {node.status} -> {normalized_status}"
                )
            node.status = normalized_status
        if summary:
            node.latest_summary = str(summary).strip()
        if normalized_status == "completed" and detail:
            node.result_text = str(detail)
        if normalized_status == "failed" and detail:
            node.error_text = str(detail)
        if metadata:
            node.metadata.update(dict(metadata))
        node.touch()
        self.touch()
        status_summary = str(summary or normalized_status).strip()
        self.record_event(
            node_id=node_id,
            kind="status",
            summary=status_summary,
            detail=detail,
            metadata={"status": normalized_status, **dict(metadata or {})},
        )
        return node

    def record_event(
        self,
        *,
        node_id: str,
        kind: NotebookEventKind,
        summary: str,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookEvent:
        node = self.get_node(node_id)
        event = NotebookEvent(
            event_id=_new_id("evt"),
            node_id=node_id,
            kind=kind,
            summary=str(summary or "").strip(),
            detail=str(detail or ""),
            metadata=dict(metadata or {}),
        )
        node.events.append(event)
        if event.summary:
            node.latest_summary = event.summary
        node.touch()
        self.raw_events.append(event)
        self.touch()
        return event

    def add_artifact(
        self,
        *,
        node_id: str,
        path: str,
        kind: str = "file",
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookArtifact:
        node = self.get_node(node_id)
        artifact = NotebookArtifact(
            artifact_id=_new_id("art"),
            node_id=node_id,
            path=str(path or "").strip(),
            kind=str(kind or "").strip() or "file",
            label=str(label or "").strip(),
            metadata=dict(metadata or {}),
        )
        node.artifacts.append(artifact)
        node.touch()
        self.touch()
        self.record_event(
            node_id=node_id,
            kind="artifact",
            summary=artifact.label or artifact.path,
            detail=artifact.path,
            metadata={
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                **dict(metadata or {}),
            },
        )
        return artifact

    def add_issue(
        self,
        *,
        node_id: str,
        title: str,
        detail: str = "",
        severity: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookIssue:
        node = self.get_node(node_id)
        issue = NotebookIssue(
            issue_id=_new_id("issue"),
            node_id=node_id,
            title=str(title or "").strip(),
            detail=str(detail or ""),
            severity=str(severity or "").strip() or "normal",
            metadata=dict(metadata or {}),
        )
        node.issues.append(issue)
        node.touch()
        self.touch()
        self.record_event(
            node_id=node_id,
            kind="issue",
            summary=issue.title,
            detail=issue.detail,
            metadata={
                "issue_id": issue.issue_id,
                "severity": issue.severity,
                **dict(metadata or {}),
            },
        )
        return issue

    def add_decision(
        self,
        *,
        node_id: str,
        summary: str,
        rationale: str = "",
        status: str = "verified",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookDecision:
        node = self.get_node(node_id)
        decision = NotebookDecision(
            decision_id=_new_id("decision"),
            node_id=node_id,
            summary=str(summary or "").strip(),
            rationale=str(rationale or ""),
            status=str(status or "").strip() or "verified",
            metadata=dict(metadata or {}),
        )
        node.decisions.append(decision)
        node.latest_summary = decision.summary or node.latest_summary
        node.touch()
        self.touch()
        self.record_event(
            node_id=node_id,
            kind="decision",
            summary=decision.summary,
            detail=decision.rationale,
            metadata={
                "decision_id": decision.decision_id,
                "status": decision.status,
                **dict(metadata or {}),
            },
        )
        return decision

    def add_checkpoint(
        self,
        *,
        node_id: str,
        kind: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> NotebookCheckpoint:
        node = self.get_node(node_id)
        checkpoint = NotebookCheckpoint(
            checkpoint_id=_new_id("checkpoint"),
            node_id=node_id,
            kind=str(kind or "").strip(),
            message=str(message or "").strip(),
            metadata=dict(metadata or {}),
        )
        node.checkpoints.append(checkpoint)
        node.touch()
        self.touch()
        self.record_event(
            node_id=node_id,
            kind="checkpoint",
            summary=checkpoint.message,
            metadata={
                "checkpoint_id": checkpoint.checkpoint_id,
                "kind": checkpoint.kind,
                **dict(metadata or {}),
            },
        )
        return checkpoint

    def open_checkpoints(
        self,
        *,
        node_id: str | None = None,
        kind: str | None = None,
    ) -> list[NotebookCheckpoint]:
        nodes = [self.get_node(node_id)] if node_id else list(self.nodes.values())
        matches: list[NotebookCheckpoint] = []
        for node in nodes:
            for checkpoint in node.checkpoints:
                if checkpoint.status != "open":
                    continue
                if kind and checkpoint.kind != kind:
                    continue
                matches.append(checkpoint)
        return matches

    def resolve_checkpoints(
        self,
        node_id: str,
        *,
        kind: str | None = None,
    ) -> list[NotebookCheckpoint]:
        node = self.get_node(node_id)
        resolved: list[NotebookCheckpoint] = []
        for checkpoint in node.checkpoints:
            if checkpoint.status != "open":
                continue
            if kind and checkpoint.kind != kind:
                continue
            checkpoint.resolve()
            resolved.append(checkpoint)
            self.record_event(
                node_id=node_id,
                kind="checkpoint",
                summary=f"Resolved checkpoint: {checkpoint.message}",
                metadata={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "kind": checkpoint.kind,
                    "status": checkpoint.status,
                },
            )
        if resolved:
            node.touch()
            self.touch()
        return resolved

    def mark_needs_human_input(
        self,
        node_id: str,
        *,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> NotebookCheckpoint:
        checkpoint = self.add_checkpoint(
            node_id=node_id,
            kind="needs_human_input",
            message=message,
            metadata=metadata,
        )
        node = self.get_node(node_id)
        if node.status not in _TERMINAL_NODE_STATUSES:
            self.transition_node(
                node_id,
                "waiting",
                summary=message,
                metadata={"checkpoint_kind": checkpoint.kind, **dict(metadata or {})},
            )
        return checkpoint

    def mark_verification_failed(
        self,
        node_id: str,
        *,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> NotebookCheckpoint:
        checkpoint = self.add_checkpoint(
            node_id=node_id,
            kind="verification_failed",
            message=message,
            metadata=metadata,
        )
        node = self.get_node(node_id)
        if node.status not in _TERMINAL_NODE_STATUSES:
            self.transition_node(
                node_id,
                "blocked",
                summary=message,
                metadata={"checkpoint_kind": checkpoint.kind, **dict(metadata or {})},
            )
        return checkpoint

    def mark_needs_repair(
        self,
        node_id: str,
        *,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> NotebookCheckpoint:
        return self.add_checkpoint(
            node_id=node_id,
            kind="needs_repair",
            message=message,
            metadata=metadata,
        )

    def mark_ready_to_finalize(
        self,
        node_id: str,
        *,
        message: str = "Ready to finalize",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookCheckpoint:
        checkpoint = self.add_checkpoint(
            node_id=node_id,
            kind="ready_to_finalize",
            message=message,
            metadata=metadata,
        )
        checkpoint.resolve()
        self.record_event(
            node_id=node_id,
            kind="summary",
            summary=message,
            metadata={"checkpoint_kind": checkpoint.kind, **dict(metadata or {})},
        )
        return checkpoint

    def promote_failure_to_repair(
        self,
        node_id: str,
        *,
        owner: str = "repair",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> NotebookNode:
        failed_node = self.get_node(node_id)
        repair_message = str(message or failed_node.objective or failed_node.title).strip()
        checkpoint = self.mark_needs_repair(
            node_id,
            message=repair_message or "Repair required",
            metadata=metadata,
        )
        repair_node = self.add_child_node(
            parent_id=node_id,
            kind="repair",
            title=f"Repair: {failed_node.title}",
            objective=repair_message or failed_node.objective,
            owner=owner,
            resource_ids=failed_node.resource_ids,
            metadata={
                "repair_for": node_id,
                "repair_checkpoint_id": checkpoint.checkpoint_id,
                **dict(metadata or {}),
            },
        )
        self.transition_node(
            repair_node.node_id,
            "running",
            summary="Repair branch created",
            detail=repair_message,
            metadata={"repair_for": node_id, **dict(metadata or {})},
        )
        return repair_node

    def unresolved_node_ids(self) -> list[str]:
        return [
            node_id
            for node_id, node in self.nodes.items()
            if node.status not in _TERMINAL_NODE_STATUSES
        ]

    def ready_to_finalize(self) -> bool:
        if not self.nodes:
            return False
        unresolved_non_root = [
            node_id
            for node_id, node in self.nodes.items()
            if node_id != self.root_node_id and node.status not in _TERMINAL_NODE_STATUSES
        ]
        if unresolved_non_root:
            return False
        for node in self.nodes.values():
            if any(item.status == "open" for item in node.checkpoints):
                return False
        return True

    def frontier_node_ids(self) -> list[str]:
        frontiers: list[str] = []
        for node_id, node in self.nodes.items():
            if node.status in _TERMINAL_NODE_STATUSES:
                continue
            if not node.deps:
                frontiers.append(node_id)
                continue
            if all(
                self.nodes.get(dep_id) is not None
                and self.nodes[dep_id].status == "completed"
                for dep_id in node.deps
            ):
                frontiers.append(node_id)
        return frontiers

    def progress_marker_count(self, node_id: str) -> int:
        node = self.get_node(node_id)
        return sum(1 for event in node.events if bool(event.metadata.get("progress")))

    def set_completion_summary(self, payload: dict[str, Any]) -> None:
        self.completion_summary = dict(payload or {})
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "notebook_id": self.notebook_id,
            "goal": self.goal,
            "flow_id": self.flow_id,
            "plan_id": self.plan_id,
            "root_node_id": self.root_node_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "raw_events": [item.to_dict() for item in self.raw_events],
            "completion_summary": dict(self.completion_summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanNotebook":
        notebook = cls(
            notebook_id=str(payload.get("notebook_id", "") or _new_id("notebook")),
            goal=str(payload.get("goal", "") or ""),
            flow_id=str(payload.get("flow_id", "") or ""),
            plan_id=str(payload.get("plan_id", "") or ""),
            root_node_id=str(payload.get("root_node_id", "") or ""),
            created_at=float(payload.get("created_at", _now_ts()) or _now_ts()),
            updated_at=float(payload.get("updated_at", _now_ts()) or _now_ts()),
            metadata=dict(payload.get("metadata") or {}),
            completion_summary=dict(payload.get("completion_summary") or {}),
        )
        raw_nodes = dict(payload.get("nodes") or {})
        for node_id, node_payload in raw_nodes.items():
            node = NotebookNode(
                node_id=str(node_payload.get("node_id", "") or node_id),
                kind=str(node_payload.get("kind", "") or "task"),
                title=str(node_payload.get("title", "") or ""),
                objective=str(node_payload.get("objective", "") or ""),
                parent_id=node_payload.get("parent_id"),
                owner=str(node_payload.get("owner", "") or ""),
                resource_ids=tuple(node_payload.get("resource_ids") or ()),
                deps=tuple(node_payload.get("deps") or ()),
                status=str(node_payload.get("status", "pending") or "pending"),  # type: ignore[arg-type]
                created_at=float(node_payload.get("created_at", _now_ts()) or _now_ts()),
                updated_at=float(node_payload.get("updated_at", _now_ts()) or _now_ts()),
                latest_summary=str(node_payload.get("latest_summary", "") or ""),
                result_text=str(node_payload.get("result_text", "") or ""),
                error_text=str(node_payload.get("error_text", "") or ""),
                metadata=dict(node_payload.get("metadata") or {}),
                events=[
                    NotebookEvent(**dict(item))
                    for item in (node_payload.get("events") or [])
                ],
                artifacts=[
                    NotebookArtifact(**dict(item))
                    for item in (node_payload.get("artifacts") or [])
                ],
                issues=[
                    NotebookIssue(**dict(item))
                    for item in (node_payload.get("issues") or [])
                ],
                decisions=[
                    NotebookDecision(**dict(item))
                    for item in (node_payload.get("decisions") or [])
                ],
                checkpoints=[
                    NotebookCheckpoint(**dict(item))
                    for item in (node_payload.get("checkpoints") or [])
                ],
            )
            notebook.nodes[node.node_id] = node
        notebook.raw_events = [
            NotebookEvent(**dict(item)) for item in (payload.get("raw_events") or [])
        ]
        return notebook


def create_root_notebook(
    *,
    goal: str,
    flow_id: str = "",
    plan_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> PlanNotebook:
    notebook_id = _new_id("notebook")
    root_node = NotebookNode(
        node_id=_new_id("node"),
        kind="root",
        title="Root task",
        objective=str(goal or "").strip(),
        status="running",
        metadata={"flow_id": flow_id, **dict(metadata or {})},
    )
    notebook = PlanNotebook(
        notebook_id=notebook_id,
        goal=str(goal or "").strip(),
        flow_id=str(flow_id or "").strip(),
        plan_id=str(plan_id or "").strip(),
        root_node_id=root_node.node_id,
        metadata=dict(metadata or {}),
        nodes={root_node.node_id: root_node},
    )
    notebook.record_event(
        node_id=root_node.node_id,
        kind="summary",
        summary=root_node.objective or "task created",
        metadata={"root": True},
    )
    return notebook
