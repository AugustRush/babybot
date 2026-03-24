# tests/test_agent_team.py
"""Tests for agent team: Mailbox and SharedTaskList."""

from __future__ import annotations
import asyncio
import pytest
from babybot.agent_kernel.team import Mailbox, SharedTaskList, TeamTask, TeamRunner


@pytest.mark.asyncio
async def test_mailbox_send_receive() -> None:
    mb = Mailbox()
    mb.send("agent_a", "agent_b", "Hello from A")
    messages = mb.receive("agent_b")
    assert len(messages) == 1
    assert messages[0].sender == "agent_a"
    assert messages[0].content == "Hello from A"


@pytest.mark.asyncio
async def test_mailbox_receive_clears() -> None:
    mb = Mailbox()
    mb.send("a", "b", "msg1")
    mb.receive("b")
    assert mb.receive("b") == []


@pytest.mark.asyncio
async def test_mailbox_broadcast() -> None:
    mb = Mailbox()
    mb.broadcast("lead", ["a", "b", "c"], "Announcement")
    assert len(mb.receive("a")) == 1
    assert len(mb.receive("b")) == 1
    assert len(mb.receive("c")) == 1
    assert mb.receive("lead") == []


@pytest.mark.asyncio
async def test_mailbox_wait_for_message() -> None:
    mb = Mailbox()

    async def delayed_send():
        await asyncio.sleep(0.05)
        mb.send("a", "b", "delayed")

    asyncio.create_task(delayed_send())
    msg = await asyncio.wait_for(mb.wait_for_message("b"), timeout=1.0)
    assert msg.content == "delayed"


def test_task_list_add_and_claim() -> None:
    tl = SharedTaskList()
    tl.add(TeamTask(task_id="t1", description="Do X"))
    tl.add(TeamTask(task_id="t2", description="Do Y"))
    claimed = tl.claim("agent_a")
    assert claimed is not None
    assert claimed.task_id == "t1"
    assert claimed.assigned_to == "agent_a"
    claimed2 = tl.claim("agent_b")
    assert claimed2 is not None
    assert claimed2.task_id == "t2"
    assert tl.claim("agent_c") is None


def test_task_list_complete() -> None:
    tl = SharedTaskList()
    tl.add(TeamTask(task_id="t1", description="Do X"))
    tl.claim("agent_a")
    tl.complete("t1", output="Done")
    status = tl.get_status()
    assert status["t1"].status == "completed"
    assert status["t1"].output == "Done"


def test_task_list_dependencies() -> None:
    tl = SharedTaskList()
    tl.add(TeamTask(task_id="t1", description="First"))
    tl.add(TeamTask(task_id="t2", description="Second", deps=["t1"]))
    claimed = tl.claim("agent_a")
    assert claimed.task_id == "t1"
    assert tl.claim("agent_b") is None  # t2 blocked
    tl.complete("t1", output="ok")
    claimed2 = tl.claim("agent_b")
    assert claimed2 is not None
    assert claimed2.task_id == "t2"


# ── TeamRunner tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_team_runner_debate() -> None:
    """TeamRunner executes a multi-round debate between two agents."""
    round_count = 0

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        nonlocal round_count
        round_count += 1
        if "agent_a" in agent_id:
            return f"A's argument round {round_count}: I disagree because X"
        return f"B's counter round {round_count}: Actually Y"

    runner = TeamRunner(executor=mock_executor, max_rounds=3)
    result = await runner.run_debate(
        topic="Should we use microservices?",
        agents=[
            {
                "id": "agent_a",
                "role": "proponent",
                "description": "Argues for microservices",
            },
            {
                "id": "agent_b",
                "role": "opponent",
                "description": "Argues against microservices",
            },
        ],
    )

    assert result.rounds >= 2
    assert result.rounds <= 3
    assert len(result.transcript) > 0
    assert result.summary  # non-empty summary


@pytest.mark.asyncio
async def test_team_runner_convergence() -> None:
    """TeamRunner stops early when judge signals convergence."""

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        return "I agree with the previous point."

    runner = TeamRunner(executor=mock_executor, max_rounds=10)
    result = await runner.run_debate(
        topic="Test topic",
        agents=[
            {"id": "a", "role": "proponent", "description": "Pro"},
            {"id": "b", "role": "opponent", "description": "Con"},
        ],
        judge=lambda transcript: (True, "Agents reached consensus"),
    )

    # Should stop before max_rounds due to convergence
    assert result.rounds < 10
    assert "consensus" in result.summary.lower()


@pytest.mark.asyncio
async def test_team_runner_per_agent_executor() -> None:
    """TeamRunner routes each agent to its own executor when provided."""
    call_log: list[str] = []

    async def global_exec(agent_id: str, prompt: str, context: dict) -> str:
        call_log.append(f"global:{agent_id}")
        return f"global response from {agent_id}"

    async def special_exec(agent_id: str, prompt: str, context: dict) -> str:
        call_log.append(f"special:{agent_id}")
        return f"special response from {agent_id}"

    runner = TeamRunner(executor=global_exec, max_rounds=1)
    result = await runner.run_debate(
        topic="Test routing",
        agents=[
            {"id": "agent_a", "role": "pro", "description": "Uses global"},
            {
                "id": "agent_b",
                "role": "con",
                "description": "Uses special",
                "executor": special_exec,
            },
        ],
    )

    assert "global:agent_a" in call_log
    assert "special:agent_b" in call_log
    assert "global:agent_b" not in call_log
    assert result.rounds == 1


# ---- on_turn callback tests ----


@pytest.mark.asyncio
async def test_team_runner_on_turn_fires_for_each_agent_turn() -> None:
    """on_turn callback is invoked after each agent produces output."""
    turn_log: list[tuple[str, str, int, str]] = []

    async def fake_executor(agent_id: str, prompt: str, ctx: dict) -> str:
        return f"response from {agent_id}"

    async def on_turn(agent_id: str, role: str, round_num: int, text: str) -> None:
        turn_log.append((agent_id, role, round_num, text))

    runner = TeamRunner(executor=fake_executor, max_rounds=2)
    result = await runner.run_debate(
        topic="test",
        agents=[
            {"id": "a", "role": "Pro", "description": "For"},
            {"id": "b", "role": "Con", "description": "Against"},
        ],
        on_turn=on_turn,
    )

    assert result.rounds == 2
    # 2 rounds x 2 agents = 4 calls
    assert len(turn_log) == 4
    assert turn_log[0] == ("a", "Pro", 1, "response from a")
    assert turn_log[1] == ("b", "Con", 1, "response from b")
    assert turn_log[2] == ("a", "Pro", 2, "response from a")
    assert turn_log[3] == ("b", "Con", 2, "response from b")


@pytest.mark.asyncio
async def test_team_runner_sync_on_turn_works() -> None:
    """on_turn can be a sync callable."""
    turn_count = {"n": 0}

    async def fake_executor(agent_id: str, prompt: str, ctx: dict) -> str:
        return "ok"

    def on_turn_sync(agent_id: str, role: str, round_num: int, text: str) -> None:
        turn_count["n"] += 1

    runner = TeamRunner(executor=fake_executor, max_rounds=1)
    await runner.run_debate(
        topic="test",
        agents=[
            {"id": "a", "role": "X", "description": "x"},
            {"id": "b", "role": "Y", "description": "y"},
        ],
        on_turn=on_turn_sync,
    )
    assert turn_count["n"] == 2


@pytest.mark.asyncio
async def test_team_runner_on_turn_not_required() -> None:
    """Omitting on_turn still works (backward compatible)."""
    async def fake_executor(agent_id: str, prompt: str, ctx: dict) -> str:
        return "ok"

    runner = TeamRunner(executor=fake_executor, max_rounds=1)
    result = await runner.run_debate(
        topic="test",
        agents=[
            {"id": "a", "role": "X", "description": "x"},
            {"id": "b", "role": "Y", "description": "y"},
        ],
    )
    assert result.rounds == 1
    assert len(result.transcript) == 2
