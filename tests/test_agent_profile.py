"""Tests for declarative agent profiles."""

from __future__ import annotations
import os
import tempfile
import pytest
from babybot.agent_kernel.agent_profile import AgentProfile, AgentProfileLoader


def test_parse_agent_profile_from_markdown() -> None:
    """Parse an AGENT.md file into an AgentProfile."""
    content = """\
---
name: code-reviewer
role: reviewer
description: Reviews code for quality and correctness
resource_id: skill.code
---

# Code Reviewer

You are an expert code reviewer. Focus on:
- Correctness
- Performance
- Readability
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        profile = AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)

    assert profile.name == "code-reviewer"
    assert profile.role == "reviewer"
    assert profile.description == "Reviews code for quality and correctness"
    assert profile.resource_id == "skill.code"
    assert "expert code reviewer" in profile.system_prompt


def test_parse_agent_profile_minimal() -> None:
    """Minimal AGENT.md with only required fields."""
    content = """\
---
name: debater
role: proponent
---

Argue for the given position.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        profile = AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)

    assert profile.name == "debater"
    assert profile.role == "proponent"
    assert profile.description == ""
    assert profile.resource_id == ""
    assert "Argue for" in profile.system_prompt


def test_load_profiles_from_directory() -> None:
    """AgentProfileLoader.load_dir scans a directory for AGENT.md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, role in [("alice", "proponent"), ("bob", "opponent")]:
            d = os.path.join(tmpdir, name)
            os.makedirs(d)
            with open(os.path.join(d, "AGENT.md"), "w") as f:
                f.write(f"---\nname: {name}\nrole: {role}\n---\n\nPrompt for {name}.\n")

        profiles = AgentProfileLoader.load_dir(tmpdir)

    assert len(profiles) == 2
    names = {p.name for p in profiles}
    assert names == {"alice", "bob"}


def test_profile_to_agent_dict() -> None:
    """AgentProfile.to_agent_dict() produces a dict compatible with dispatch_team."""
    profile = AgentProfile(
        name="reviewer",
        role="reviewer",
        description="Reviews code",
        resource_id="skill.code",
        system_prompt="You are a reviewer.",
    )
    d = profile.to_agent_dict()
    assert d["id"] == "reviewer"
    assert d["role"] == "reviewer"
    assert d["description"] == "Reviews code"
    assert d["resource_id"] == "skill.code"
    assert d["system_prompt"] == "You are a reviewer."


def test_load_file_rejects_missing_name() -> None:
    """AGENT.md without 'name' in frontmatter raises ValueError."""
    content = "---\nrole: pro\n---\nHello\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        with pytest.raises(ValueError, match="name"):
            AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)


def test_load_file_rejects_missing_role() -> None:
    """AGENT.md without 'role' in frontmatter raises ValueError."""
    content = "---\nname: foo\n---\nHello\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        with pytest.raises(ValueError, match="role"):
            AgentProfileLoader.load_file(f.name)

    os.unlink(f.name)
