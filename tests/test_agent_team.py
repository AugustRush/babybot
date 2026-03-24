# tests/test_agent_team.py
"""Tests for agent team: Mailbox and SharedTaskList."""

from __future__ import annotations
import asyncio
import pytest
from babybot.agent_kernel.team import Mailbox, SharedTaskList, TeamTask


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
