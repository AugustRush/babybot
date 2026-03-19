# Auto Skill Creator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `skills/auto_skill_creator` so it can deterministically scaffold and validate builtin and workspace skills without changing `babybot` skill discovery.

**Architecture:** Keep `auto_skill_creator` as a normal builtin skill package. Add two helper scripts under the skill itself (`init_skill.py` and `quick_validate.py`) and rewrite the skill instructions to use them. Validate behavior through direct script tests instead of runtime integration changes.

**Tech Stack:** Python 3.11, pytest, existing `babybot` skill directory conventions

---

### Task 1: Add Failing Tests For Skill Initialization

**Files:**
- Create: `tests/test_auto_skill_creator_scripts.py`
- Test: `tests/test_auto_skill_creator_scripts.py`

- [ ] **Step 1: Write failing tests for workspace and builtin initialization**

Cover:
- workspace target creates `<workspace>/skills/<skill-name>/SKILL.md`
- builtin target creates `<repo>/skills/<skill-name>/SKILL.md`
- resource directories are created only when requested
- invalid resource types are rejected

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest -q tests/test_auto_skill_creator_scripts.py -k init_skill`
Expected: FAIL because scripts do not exist yet

- [ ] **Step 3: Implement minimal `init_skill.py`**

Create deterministic scaffolding under `skills/auto_skill_creator/scripts/init_skill.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest -q tests/test_auto_skill_creator_scripts.py -k init_skill`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_auto_skill_creator_scripts.py skills/auto_skill_creator/scripts/init_skill.py
git commit -m "add auto skill creator initializer"
```

### Task 2: Add Failing Tests For Validation

**Files:**
- Modify: `tests/test_auto_skill_creator_scripts.py`
- Create: `skills/auto_skill_creator/scripts/quick_validate.py`

- [ ] **Step 1: Write failing tests for skill validation**

Cover:
- valid generated skill passes
- missing description fails
- unexpected root file fails
- TODO placeholder description fails

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest -q tests/test_auto_skill_creator_scripts.py -k quick_validate`
Expected: FAIL because validator does not exist yet

- [ ] **Step 3: Implement minimal `quick_validate.py`**

Match current `babybot` loader assumptions and avoid unsupported metadata rules.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest -q tests/test_auto_skill_creator_scripts.py -k quick_validate`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_auto_skill_creator_scripts.py skills/auto_skill_creator/scripts/quick_validate.py
git commit -m "add auto skill creator validator"
```

### Task 3: Rewrite The Skill Instructions

**Files:**
- Modify: `skills/auto_skill_creator/SKILL.md`
- Test: `tests/test_auto_skill_creator_scripts.py`

- [ ] **Step 1: Rewrite `SKILL.md` around the new workflow**

Add:
- builtin/workspace targeting guidance
- example-driven design guidance
- resource planning guidance
- explicit script invocation flow
- validation requirement before completion

- [ ] **Step 2: Add or update a lightweight test if needed**

If helpful, assert that generated skeleton plus validator align with the documented structure.

- [ ] **Step 3: Run focused tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_auto_skill_creator_scripts.py`
Expected: PASS

- [ ] **Step 4: Run related regression tests**

Run: `PYTHONPATH=. uv run pytest -q tests/test_resource_skills.py tests/test_agent_kernel_skills.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skills/auto_skill_creator/SKILL.md tests/test_auto_skill_creator_scripts.py
git commit -m "upgrade auto skill creator workflow"
```

### Task 4: Final Verification

**Files:**
- Modify: none expected
- Test: `tests/test_auto_skill_creator_scripts.py`, `tests/test_resource_skills.py`, `tests/test_agent_kernel_skills.py`

- [ ] **Step 1: Run final verification suite**

Run: `PYTHONPATH=. uv run pytest -q tests/test_auto_skill_creator_scripts.py tests/test_resource_skills.py tests/test_agent_kernel_skills.py`
Expected: PASS

- [ ] **Step 2: Inspect git diff for scope control**

Run: `git diff --stat`
Expected: only `skills/auto_skill_creator/*`, `tests/test_auto_skill_creator_scripts.py`, and any minimal related test adjustments

- [ ] **Step 3: Prepare summary**

Summarize:
- what changed
- what stayed intentionally unchanged
- test evidence
