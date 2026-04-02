# tests/test_agent_team.py
"""Tests for agent team: Mailbox, SharedTaskList, and TeamRunner (debate + cooperative)."""

from __future__ import annotations
import asyncio
import pytest
from babybot.agent_kernel.execution_constraints import TeamExecutionPolicy
from babybot.agent_kernel.team import (
    Mailbox,
    SharedTaskList,
    TeamTask,
    TeamRunner,
    CooperativeResult,
)


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


@pytest.mark.asyncio
async def test_team_runner_on_round_start_fires_for_each_round() -> None:
    """on_round_start is invoked before each debate round begins."""
    started_rounds: list[int] = []

    async def fake_executor(agent_id: str, prompt: str, ctx: dict) -> str:
        del agent_id, prompt, ctx
        return "ok"

    async def on_round_start(round_num: int, max_rounds: int) -> None:
        started_rounds.append(round_num)
        assert max_rounds == 2

    runner = TeamRunner(executor=fake_executor, max_rounds=2)
    result = await runner.run_debate(
        topic="test",
        agents=[
            {"id": "a", "role": "X", "description": "x"},
            {"id": "b", "role": "Y", "description": "y"},
        ],
        on_round_start=on_round_start,
    )

    assert result.rounds == 2
    assert started_rounds == [1, 2]


def test_team_runner_summarizes_partial_result_when_turn_budget_exceeded() -> None:
    async def slow_executor(agent_id: str, prompt: str, ctx: dict) -> str:
        del agent_id, prompt, ctx
        await asyncio.sleep(0.05)
        return "slow"

    runner = TeamRunner(
        executor=slow_executor,
        max_rounds=3,
        policy=TeamExecutionPolicy(
            max_turn_seconds=0.01, on_budget_exhausted="summarize_partial"
        ),
    )
    result = asyncio.run(
        runner.run_debate(
            topic="test",
            agents=[
                {"id": "a", "role": "X", "description": "x"},
                {"id": "b", "role": "Y", "description": "y"},
            ],
        )
    )

    assert result.rounds == 0
    assert result.completed is False
    assert result.termination_reason == "turn_timeout"
    assert "partial" in result.summary.lower()


# ── Cooperative mode tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cooperative_basic_two_agents_two_tasks() -> None:
    """Two agents execute two independent tasks in cooperative mode."""
    call_log: list[tuple[str, str]] = []

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        # Extract task_id from prompt
        task_id = ""
        for line in prompt.split("\n"):
            if line.startswith("Task ["):
                task_id = line.split("[")[1].split("]")[0]
                break
        call_log.append((agent_id, task_id))
        return f"Result from {agent_id} for {task_id}"

    runner = TeamRunner(executor=mock_executor, max_rounds=5)
    result = await runner.run_cooperative(
        topic="Build a website",
        agents=[
            {"id": "agent_a", "role": "frontend", "description": "Frontend developer"},
            {"id": "agent_b", "role": "backend", "description": "Backend developer"},
        ],
        tasks=[
            {"task_id": "t1", "description": "Build homepage"},
            {"task_id": "t2", "description": "Build API"},
        ],
    )

    assert isinstance(result, CooperativeResult)
    assert result.tasks_completed == 2
    assert result.tasks_failed == 0
    assert result.tasks_total == 2
    assert result.completed is True
    assert result.termination_reason == "completed"
    assert "t1" in result.task_outputs
    assert "t2" in result.task_outputs
    assert len(result.mailbox_log) >= 2  # At least 2 task_completed broadcasts


@pytest.mark.asyncio
async def test_cooperative_task_dependencies() -> None:
    """Tasks with deps wait for upstream tasks to complete first."""
    execution_order: list[str] = []

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        task_id = ""
        for line in prompt.split("\n"):
            if line.startswith("Task ["):
                task_id = line.split("[")[1].split("]")[0]
                break
        execution_order.append(task_id)
        await asyncio.sleep(0.02)  # Simulate work
        return f"Done {task_id}"

    runner = TeamRunner(executor=mock_executor, max_rounds=5)
    result = await runner.run_cooperative(
        topic="Sequential pipeline",
        agents=[
            {"id": "a", "role": "worker", "description": "Worker"},
            {"id": "b", "role": "worker", "description": "Worker"},
        ],
        tasks=[
            {"task_id": "step1", "description": "First step"},
            {"task_id": "step2", "description": "Second step", "deps": ["step1"]},
        ],
    )

    assert result.tasks_completed == 2
    assert result.completed is True
    # step1 must complete before step2 can be claimed
    assert execution_order.index("step1") < execution_order.index("step2")


@pytest.mark.asyncio
async def test_cooperative_task_failure() -> None:
    """A failing task is recorded and doesn't block other tasks."""

    async def failing_executor(agent_id: str, prompt: str, context: dict) -> str:
        task_id = ""
        for line in prompt.split("\n"):
            if line.startswith("Task ["):
                task_id = line.split("[")[1].split("]")[0]
                break
        if task_id == "bad_task":
            raise ValueError("Intentional failure")
        return f"Success for {task_id}"

    runner = TeamRunner(executor=failing_executor, max_rounds=5)
    result = await runner.run_cooperative(
        topic="Mixed tasks",
        agents=[
            {"id": "a", "role": "worker", "description": "Worker"},
            {"id": "b", "role": "worker", "description": "Worker"},
        ],
        tasks=[
            {"task_id": "good_task", "description": "This should succeed"},
            {"task_id": "bad_task", "description": "This will fail"},
        ],
    )

    assert result.tasks_completed == 1
    assert result.tasks_failed == 1
    assert result.tasks_total == 2
    assert result.completed is False  # Not all tasks succeeded
    assert "good_task" in result.task_outputs
    assert "bad_task" not in result.task_outputs


@pytest.mark.asyncio
async def test_cooperative_on_task_complete_callback() -> None:
    """on_task_complete callback fires after each successful task."""
    callback_log: list[tuple[str, str]] = []

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        return "done"

    async def on_complete(agent_id: str, task_id: str, output: str) -> None:
        callback_log.append((agent_id, task_id))

    runner = TeamRunner(executor=mock_executor, max_rounds=5)
    result = await runner.run_cooperative(
        topic="Callback test",
        agents=[
            {"id": "a", "role": "worker", "description": "Worker"},
            {"id": "b", "role": "worker", "description": "Worker"},
        ],
        tasks=[
            {"task_id": "t1", "description": "Task 1"},
            {"task_id": "t2", "description": "Task 2"},
        ],
        on_task_complete=on_complete,
    )

    assert result.tasks_completed == 2
    assert len(callback_log) == 2
    task_ids_completed = {tid for _, tid in callback_log}
    assert task_ids_completed == {"t1", "t2"}


@pytest.mark.asyncio
async def test_cooperative_upstream_context_in_prompt() -> None:
    """Dependent tasks receive upstream results via mailbox in their prompts."""
    prompts_seen: dict[str, str] = {}

    async def capturing_executor(agent_id: str, prompt: str, context: dict) -> str:
        task_id = ""
        for line in prompt.split("\n"):
            if line.startswith("Task ["):
                task_id = line.split("[")[1].split("]")[0]
                break
        prompts_seen[task_id] = prompt
        return f"output_of_{task_id}"

    runner = TeamRunner(executor=capturing_executor, max_rounds=5)
    result = await runner.run_cooperative(
        topic="Context passing",
        agents=[
            {"id": "a", "role": "producer", "description": "Produces data"},
            {"id": "b", "role": "consumer", "description": "Consumes data"},
        ],
        tasks=[
            {"task_id": "produce", "description": "Produce data"},
            {"task_id": "consume", "description": "Consume data", "deps": ["produce"]},
        ],
    )

    assert result.tasks_completed == 2
    # The consumer task's prompt should contain upstream context from the producer
    consume_prompt = prompts_seen.get("consume", "")
    assert "上游任务结果" in consume_prompt or "DONE" in consume_prompt


@pytest.mark.asyncio
async def test_cooperative_deadline_exceeded() -> None:
    """Cooperative execution respects deadline and reports partial results."""

    async def slow_executor(agent_id: str, prompt: str, context: dict) -> str:
        await asyncio.sleep(0.5)  # Slow task
        return "done"

    runner = TeamRunner(
        executor=slow_executor,
        max_rounds=5,
        policy=TeamExecutionPolicy(max_total_seconds=0.1),
    )
    result = await runner.run_cooperative(
        topic="Deadline test",
        agents=[
            {"id": "a", "role": "worker", "description": "Worker"},
            {"id": "b", "role": "worker", "description": "Worker"},
        ],
        tasks=[
            {"task_id": "t1", "description": "Slow task 1"},
            {"task_id": "t2", "description": "Slow task 2"},
            {"task_id": "t3", "description": "Slow task 3"},
        ],
    )

    # Should not have completed all tasks due to deadline
    assert result.tasks_total == 3
    assert result.termination_reason == "deadline_exceeded"


@pytest.mark.asyncio
async def test_cooperative_mailbox_log_captured() -> None:
    """Mailbox broadcast messages are recorded in the result log."""

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        return "result"

    runner = TeamRunner(executor=mock_executor, max_rounds=5)
    result = await runner.run_cooperative(
        topic="Log test",
        agents=[
            {"id": "a", "role": "worker", "description": "Worker"},
            {"id": "b", "role": "worker", "description": "Worker"},
        ],
        tasks=[
            {"task_id": "t1", "description": "Task 1"},
        ],
    )

    assert result.tasks_completed == 1
    assert len(result.mailbox_log) >= 1
    assert result.mailbox_log[0]["type"] == "task_completed"
    assert result.mailbox_log[0]["task_id"] == "t1"


@pytest.mark.asyncio
async def test_cooperative_max_agents_policy() -> None:
    """TeamExecutionPolicy.max_agents limits the number of active workers."""

    call_agents: set[str] = set()

    async def mock_executor(agent_id: str, prompt: str, context: dict) -> str:
        call_agents.add(agent_id)
        return "done"

    runner = TeamRunner(
        executor=mock_executor,
        max_rounds=5,
        policy=TeamExecutionPolicy(max_agents=1),
    )
    result = await runner.run_cooperative(
        topic="Agent limit test",
        agents=[
            {"id": "a", "role": "worker1", "description": "Worker 1"},
            {"id": "b", "role": "worker2", "description": "Worker 2"},
            {"id": "c", "role": "worker3", "description": "Worker 3"},
        ],
        tasks=[
            {"task_id": "t1", "description": "Task 1"},
            {"task_id": "t2", "description": "Task 2"},
        ],
    )

    assert result.tasks_completed == 2
    # Only 1 agent should have been active (max_agents=1)
    assert len(call_agents) == 1
