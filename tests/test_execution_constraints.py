import asyncio

from babybot.agent_kernel.execution_constraints import infer_execution_constraints


class _StructuredResult:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self) -> dict:
        return dict(self._payload)


class _FakeGateway:
    def __init__(self, payload: dict | None) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: type,
        heartbeat: object = None,
    ):
        del model_cls, heartbeat
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if self._payload is None:
            return None
        return _StructuredResult(self._payload)


def test_infer_execution_constraints_extracts_rule_based_limits() -> None:
    gateway = _FakeGateway(None)

    constraints = asyncio.run(
        infer_execution_constraints(
            gateway,
            "两个专家讨论，一轮定胜负，总时长不超过10分钟，每轮不超过30秒。",
            default_max_total_seconds=900.0,
        )
    )

    assert gateway.calls == []
    assert constraints["hard_limits"]["max_rounds"] == 1
    assert constraints["hard_limits"]["max_agents"] == 2
    assert constraints["hard_limits"]["max_total_seconds"] == 600.0
    assert constraints["hard_limits"]["max_turn_seconds"] == 30.0
    assert constraints["soft_preferences"]["resolution_style"] == "single_pass"


def test_infer_execution_constraints_falls_back_to_default_budget_when_model_fails() -> None:
    gateway = _FakeGateway(None)

    constraints = asyncio.run(
        infer_execution_constraints(
            gateway,
            "帮我组织两个专家讨论这个问题。",
            default_max_total_seconds=600.0,
        )
    )

    assert constraints["mode"] == "interactive"
    assert constraints["hard_limits"]["max_rounds"] is None
    assert constraints["hard_limits"]["max_total_seconds"] == 600.0
    assert constraints["degradation"]["on_budget_exhausted"] == "summarize_partial"
    assert gateway.calls == []


def test_infer_execution_constraints_extracts_soft_preferences_without_gateway() -> None:
    gateway = _FakeGateway(None)

    constraints = asyncio.run(
        infer_execution_constraints(
            gateway,
            "请尽快给我一个简洁结论，不要展开成长篇讨论。",
            default_max_total_seconds=600.0,
        )
    )

    assert gateway.calls == []
    assert constraints["soft_preferences"]["resolution_style"] == "fast_consensus"


def test_infer_execution_constraints_skips_short_greeting() -> None:
    gateway = _FakeGateway(
        {
            "mode": "deferred",
            "hard_limits": {"max_rounds": 99},
        }
    )

    constraints = asyncio.run(
        infer_execution_constraints(
            gateway,
            "hi",
            default_max_total_seconds=600.0,
        )
    )

    assert gateway.calls == []
    assert constraints["mode"] == "interactive"
    assert constraints["hard_limits"]["max_rounds"] is None
