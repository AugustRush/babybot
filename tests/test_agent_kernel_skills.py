from babybot.agent_kernel import ToolLease
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
