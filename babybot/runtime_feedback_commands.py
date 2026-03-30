"""Parsing helpers for runtime feedback and policy commands."""

from __future__ import annotations


def parse_policy_command(user_input: str) -> dict[str, str] | None:
    text = (user_input or "").strip()
    if not text.lower().startswith("@policy"):
        return None
    parts = text.split()
    action = parts[1].lower() if len(parts) >= 2 else ""
    if action == "inspect":
        target = parts[2].strip() if len(parts) >= 3 else ""
        return {"action": "inspect", "target": target}
    if action != "feedback":
        return {"action": action}
    if len(parts) >= 5 and parts[2].lower() in {"latest"} | {
        item for item in parts[2:3] if item
    }:
        target = parts[2].strip()
        rating = parts[3].lower().strip()
        reason = " ".join(parts[4:]).strip()
        return {
            "action": "feedback",
            "target": target,
            "rating": rating,
            "reason": reason,
        }
    if len(parts) >= 4 and parts[2].lower() in {"good", "bad"}:
        return {
            "action": "feedback",
            "target": "",
            "rating": parts[2].lower().strip(),
            "reason": " ".join(parts[3:]).strip(),
        }
    if len(parts) >= 5:
        return {
            "action": "feedback",
            "target": parts[2].strip(),
            "rating": parts[3].lower().strip(),
            "reason": " ".join(parts[4:]).strip(),
        }
    return {"action": "feedback", "target": "", "rating": "", "reason": ""}
