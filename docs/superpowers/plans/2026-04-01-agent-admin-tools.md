# Agent Admin Tools Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded builtin admin tool group for assistant profile and skill management, plus a guiding skill that teaches agents to use those tools safely.

**Architecture:** Add persistence and manager methods for assistant profile writes and skill active-state toggles, expose them as builtin `admin` tools, and ship an `agent-admin` skill that routes agents through inspect-then-change workflows. Keep changes scoped to profile editing and skill enable/disable/reload only.

**Tech Stack:** Python, builtin tool registry, markdown skills, pytest.

---

### Task 1: Red tests for profile admin

**Files:**
- Modify: `tests/test_memory_store.py`
- Modify: `tests/test_builtin_tools.py`

- [x] Write failing tests for saving assistant profile and admin builtin registration.
- [x] Run targeted pytest commands and verify failures.
- [x] Implement minimal code to pass.

### Task 2: Red tests for skill admin

**Files:**
- Modify: `tests/test_resource_skills.py`
- Modify: `tests/test_builtin_tools.py`

- [x] Write failing tests for list/enable/disable skill flows.
- [x] Run targeted pytest commands and verify failures.
- [x] Implement minimal code to pass.

### Task 3: Implement builtin admin group

**Files:**
- Create: `babybot/builtin_tools/admin.py`
- Modify: `babybot/builtin_tools/registry.py`
- Modify: `babybot/resource.py`
- Modify: `babybot/memory_store.py`
- Modify: `babybot/resource_models.py` (only if needed)

- [x] Add manager and storage methods.
- [x] Add builtin admin tool wrappers and register `admin` group.
- [x] Keep group disabled by default.

### Task 4: Add guiding skill

**Files:**
- Create: `skills/agent-admin/SKILL.md`

- [x] Write concise trigger-focused frontmatter.
- [x] Document safe workflow for profile and skill changes.
- [x] Verify wording points agents to builtin admin tools first.

### Task 5: Verify and commit

**Files:**
- Modify: affected files above

- [x] Run focused regression tests.
- [x] Review git diff for scope.
- [x] Commit with a focused message.
