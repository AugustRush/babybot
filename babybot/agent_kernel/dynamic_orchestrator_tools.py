"""Static tool schemas for the DynamicOrchestrator.

These definitions describe the orchestration tools that the
orchestrator model may call. Extracted from dynamic_orchestrator.py.
"""

from __future__ import annotations

from typing import Any

# ── Orchestration tool schemas (OpenAI function-calling format) ──────────

_ORCHESTRATION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "dispatch_task",
            "description": (
                "Create a sub-agent task and immediately return a task_id (non-blocking). "
                "The sub-agent will execute the task using the specified resource(s)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": "Single resource ID from the available-resources list.",
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple resource IDs when a sub-task needs combined capabilities.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Full description of the sub-task.",
                    },
                    "deps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task_ids that must complete before this task starts.",
                        "default": [],
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Sub-task timeout in seconds. Omit to use the runtime default.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_tasks",
            "description": (
                "Block until all specified tasks complete and return a JSON result map. "
                "Each result contains status/output/error and reply_artifacts_ready."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task_ids to wait for.",
                    },
                },
                "required": ["task_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_result",
            "description": (
                "Query the current status and result of a task (non-blocking, returns JSON). "
                "Result contains status/output/error and reply_artifacts_ready."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task_id to query.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_user",
            "description": (
                "Send the final reply to the user. The orchestration loop ends after this call. "
                "This tool must be called alone as the last action. "
                "The runtime will automatically attach any collected artifacts to the reply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to send to the user.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_team",
            "description": (
                "Launch a group of agents for collaborative work. Supports two modes:\n"
                "- debate (default): multi-round debate/review/brainstorm with agents taking turns.\n"
                "- cooperative: parallel task execution where agents pick tasks from a shared list "
                "and broadcast results to downstream dependents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Collaboration topic / high-level goal.",
                    },
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string"},
                                "description": {"type": "string"},
                                "resource_id": {
                                    "type": "string",
                                    "description": "Optional: resource ID for this agent.",
                                },
                                "skill_id": {
                                    "type": "string",
                                    "description": "Optional: skill name whose role/description/prompt will be inherited.",
                                },
                            },
                            "required": ["id", "role", "description"],
                        },
                        "description": "Agents participating in the collaboration (at least 2).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["debate", "cooperative"],
                        "description": (
                            "Collaboration mode. debate=multi-round discussion (default), "
                            "cooperative=parallel task execution (requires tasks parameter)."
                        ),
                        "default": "debate",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Maximum discussion rounds in debate mode (default 5).",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "description": "Unique task identifier.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Task description.",
                                },
                                "deps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of task_ids this task depends on.",
                                    "default": [],
                                },
                            },
                            "required": ["task_id", "description"],
                        },
                        "description": (
                            "Task list for cooperative mode. Each task may declare deps; "
                            "agents will automatically pick up available tasks and broadcast results."
                        ),
                    },
                },
                "required": ["topic", "agents"],
            },
        },
    },
)

_ORCHESTRATION_TOOL_BY_NAME: dict[str, dict[str, Any]] = {
    tool["function"]["name"]: tool for tool in _ORCHESTRATION_TOOLS
}
