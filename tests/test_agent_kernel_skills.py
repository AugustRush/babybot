from babybot.agent_kernel import ToolLease
from babybot.agent_kernel.lease_utils import filter_tool_lease, merge_tool_leases
from babybot.agent_kernel.skills import merge_leases


def test_merge_leases_uses_union_for_includes() -> None:
    primary = ToolLease(include_groups=("read",), include_tools=("tool_a", "tool_b"))
    secondary = ToolLease(include_groups=("write", "read"), include_tools=("tool_b",))

    merged = merge_leases(primary, secondary)
    assert merged.include_groups == ("read", "write")
    assert merged.include_tools == ("tool_a", "tool_b")


def test_merge_leases_treats_empty_include_as_no_constraint() -> None:
    primary = ToolLease(include_groups=(), include_tools=())
    secondary = ToolLease(include_groups=("search",), include_tools=("web_search",))

    merged = merge_leases(primary, secondary)
    assert merged.include_groups == ("search",)
    assert merged.include_tools == ("web_search",)


def test_merge_tool_leases_merges_multiple_inputs() -> None:
    merged = merge_tool_leases(
        ToolLease(include_groups=("a",), include_tools=("tool_a",)),
        ToolLease(include_groups=("b",), include_tools=("tool_b",)),
        ToolLease(exclude_tools=("tool_c",)),
    )

    assert merged.include_groups == ("a", "b")
    assert merged.include_tools == ("tool_a", "tool_b")
    assert merged.exclude_tools == ("tool_c",)


def test_filter_tool_lease_removes_forbidden_groups_and_tools() -> None:
    filtered = filter_tool_lease(
        ToolLease(
            include_groups=("basic", "channel_feishu", "worker_control"),
            include_tools=("safe_tool", "send_text"),
            exclude_tools=("already_blocked",),
        ),
        drop_groups=("worker_control",),
        drop_group_prefixes=("channel_",),
        drop_tools=("send_text",),
        extra_exclude_tools=("send_text",),
    )

    assert filtered.include_groups == ("basic",)
    assert filtered.include_tools == ("safe_tool",)
    assert filtered.exclude_tools == ("already_blocked", "send_text")
