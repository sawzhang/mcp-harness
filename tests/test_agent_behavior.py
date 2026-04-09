"""
Agent 行为分析测试 — Trace / Behavior / Comparator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harness.agent.trace import (
    AgentTrace,
    TurnTrace,
    ToolCallRecord,
    trace_from_dialogue_result,
)
from harness.agent.behavior import (
    analyze_planning,
    detect_loops,
    analyze_recovery,
    analyze_efficiency,
    analyze_state,
    analyze_behavior,
    BehaviorReport,
)
from harness.agent.comparator import (
    dtw_align,
    diff_cart_state,
    TraceComparator,
)
from harness.eval.adapters.base import DialogueResult, ToolCall


# ===== Trace Tests =====

def _make_trace(tool_calls: list[list[str]], errors: set[int] | None = None) -> AgentTrace:
    """Helper: build a trace from a list of per-turn tool name lists."""
    errors = errors or set()
    turns = []
    call_idx = 0
    for i, tools in enumerate(tool_calls):
        records = []
        for tool in tools:
            records.append(ToolCallRecord(
                tool=tool,
                arguments={},
                was_error=call_idx in errors,
            ))
            call_idx += 1
        turns.append(TurnTrace(turn_number=i + 1, tool_calls=records))

    total = sum(len(t) for t in tool_calls)
    return AgentTrace(
        case_id="test",
        agent_name="test-agent",
        model_name="test-model",
        turns=turns,
        total_tool_calls=total,
    )


def test_agent_trace_tool_sequence():
    trace = _make_trace([["search_menu", "add_cart"], ["calculate_price"]])
    assert trace.tool_sequence == ["search_menu", "add_cart", "calculate_price"]


def test_agent_trace_unique_tools():
    trace = _make_trace([["search_menu", "search_menu"], ["add_cart"]])
    assert trace.unique_tools == {"search_menu", "add_cart"}


def test_agent_trace_error_calls():
    trace = _make_trace([["search_menu", "add_cart"]], errors={0})
    assert len(trace.error_tool_calls) == 1
    assert trace.error_tool_calls[0].tool == "search_menu"


def test_empty_trace():
    trace = _make_trace([])
    assert trace.tool_sequence == []
    assert trace.unique_tools == set()
    assert trace.error_tool_calls == []


def test_trace_from_dialogue_result():
    dr = DialogueResult(
        turns=[
            {"role": "assistant", "content": "OK", "tool_calls": [
                {"name": "search_menu", "arguments": {"q": "latte"}},
            ]},
            {"role": "assistant", "content": "Done", "tool_calls": None},
        ],
        tool_calls=[ToolCall(tool="search_menu", arguments={"q": "latte"})],
        final_text="Done",
        total_latency_ms=200,
    )
    trace = trace_from_dialogue_result("case-1", "agent", "model", dr)
    assert trace.case_id == "case-1"
    assert len(trace.turns) == 2
    assert trace.turns[0].tool_calls[0].tool == "search_menu"


def test_trace_from_dialogue_with_timeline():
    dr = DialogueResult(
        turns=[],
        tool_calls=[ToolCall(tool="search_menu", arguments={})],
        tool_call_timeline=[
            {"turn": 0, "tool": "search_menu", "arguments": {}, "result": {}, "was_error": False},
            {"turn": 0, "tool": "add_cart", "arguments": {}, "result": {}, "was_error": True},
            {"turn": 1, "tool": "calculate_price", "arguments": {}, "result": {}, "was_error": False},
        ],
    )
    trace = trace_from_dialogue_result("case-2", "ag", "mod", dr)
    assert len(trace.turns) == 2
    assert len(trace.turns[0].tool_calls) == 2
    assert trace.turns[0].tool_calls[1].was_error is True


# ===== Planning Tests =====

def test_planning_basic():
    trace = _make_trace([["search_menu"], ["add_cart"], ["calculate_price"]])
    metrics = analyze_planning(trace)
    assert metrics.plan_step_count == 3
    assert metrics.unique_tool_count == 3
    assert metrics.parallel_tool_ratio == 0.0  # no turn has >1 tool
    assert metrics.redundant_search_rate == 0.0


def test_planning_parallel_detection():
    trace = _make_trace([["search_menu", "browse_menu"], ["calculate_price"]])
    metrics = analyze_planning(trace)
    assert metrics.parallel_tool_ratio == 0.5  # 1 of 2 turns has >1 tool


def test_planning_redundant_detection():
    # search_menu called twice with same args
    trace = _make_trace([["search_menu"], ["search_menu"], ["add_cart"]])
    metrics = analyze_planning(trace)
    assert metrics.redundant_search_rate > 0


# ===== Loop Detection Tests =====

def test_loop_search_consecutive():
    # search_menu called 4 times consecutively
    trace = _make_trace([["search_menu"], ["search_menu"], ["search_menu"], ["search_menu"]])
    loops = detect_loops(trace, max_repeat=3)
    assert len(loops.search_loops) == 1
    assert loops.search_loops[0]["tool"] == "search_menu"
    assert loops.search_loops[0]["count"] == 4
    assert loops.total_loops_detected >= 1


def test_loop_dependency_aba():
    # A -> B -> A pattern
    trace = _make_trace([["search_menu"], ["add_cart"], ["search_menu"]])
    loops = detect_loops(trace)
    assert len(loops.dependency_loops) == 1
    assert "search_menu -> add_cart -> search_menu" in loops.dependency_loops[0]["pattern"]


def test_no_loops_clean():
    trace = _make_trace([["search_menu"], ["add_cart"], ["calculate_price"]])
    loops = detect_loops(trace)
    assert loops.total_loops_detected == 0


def test_budget_exceeded():
    # 16 tools > budget of 15
    tools = [[f"tool_{i}"] for i in range(16)]
    trace = _make_trace(tools)
    loops = detect_loops(trace)
    assert loops.budget_exceeded is True


# ===== Recovery Tests =====

def test_recovery_after_error():
    trace = _make_trace([["search_menu", "add_cart"]], errors={0})
    # search_menu errors, add_cart succeeds → recovery
    metrics = analyze_recovery(trace)
    assert metrics.total_errors == 1
    assert metrics.recovery_attempts == 1
    assert metrics.recovery_success_rate == 1.0


def test_recovery_no_errors():
    trace = _make_trace([["search_menu"], ["add_cart"]])
    metrics = analyze_recovery(trace)
    assert metrics.total_errors == 0
    assert metrics.recovery_success_rate == 0


# ===== Efficiency Tests =====

def test_efficiency_optimal():
    trace = _make_trace([["search_menu"], ["add_cart"]])
    metrics = analyze_efficiency(trace, optimal_steps=2)
    assert metrics.chaining_efficiency == 1.0


def test_efficiency_with_waste():
    trace = _make_trace([["search_menu"], ["search_menu"], ["search_menu"], ["add_cart"]])
    metrics = analyze_efficiency(trace, optimal_steps=2)
    assert metrics.chaining_efficiency == 0.5  # 2/4


# ===== State Management Tests =====

def test_state_no_ground_truth():
    trace = _make_trace([["search_menu"], ["add_cart"]])
    metrics = analyze_state(trace)
    assert metrics.cart_state_accuracy == 0.0  # No ground truth, no accuracy


def test_state_exact_match():
    trace = _make_trace([["search_menu"], ["add_cart"]])
    trace.final_cart_state = {"items": [{"product_code": "D001", "size": "grande"}]}
    ground_truth = {"items": [{"product_code": "D001", "size": "grande"}]}
    metrics = analyze_state(trace, ground_truth)
    assert metrics.cart_state_accuracy == 1.0


def test_state_partial_match():
    trace = _make_trace([["search_menu"], ["add_cart"]])
    trace.final_cart_state = {"items": [{"product_code": "D001", "size": "tall"}]}
    ground_truth = {"items": [{"product_code": "D001", "size": "grande"}]}
    metrics = analyze_state(trace, ground_truth)
    assert metrics.cart_state_accuracy < 1.0


def test_state_cross_turn_reference():
    # Turn 1: search_menu, Turn 2: add_cart with store_id (references prior)
    turns = [
        TurnTrace(turn_number=1, tool_calls=[
            ToolCallRecord(tool="search_menu", arguments={"keyword": "latte"}),
        ]),
        TurnTrace(turn_number=2, tool_calls=[
            ToolCallRecord(tool="add_cart", arguments={"store_id": "ST_001", "item_id": "D001"}),
        ]),
    ]
    trace = AgentTrace(
        case_id="test", agent_name="a", model_name="m",
        turns=turns, total_tool_calls=2,
    )
    metrics = analyze_state(trace)
    assert metrics.cross_turn_reference_rate > 0  # add_cart has _id args referencing prior


# ===== Full Behavior Report =====

def test_analyze_behavior_full():
    trace = _make_trace([
        ["search_menu"],
        ["search_menu"],  # redundant
        ["add_cart"],
        ["calculate_price"],
    ])
    ground_truth = {"items": [{"product_code": "D001"}]}
    report = analyze_behavior(trace, optimal_steps=3, ground_truth=ground_truth)
    assert isinstance(report, BehaviorReport)
    assert report.planning.plan_step_count == 4
    assert report.planning.redundant_search_rate > 0
    assert report.efficiency.chaining_efficiency == 0.75  # 3/4
    # State is computed (even if accuracy is 0 due to empty final_cart_state)
    assert report.state is not None


# ===== DTW Comparator Tests =====

def test_dtw_identical():
    result = dtw_align(["A", "B", "C"], ["A", "B", "C"])
    assert result.distance == 0
    assert result.normalized_distance == 0


def test_dtw_empty():
    result = dtw_align([], [])
    assert result.distance == 0


def test_dtw_one_insertion():
    result = dtw_align(["A", "B"], ["A", "X", "B"])
    assert result.distance == 1


def test_dtw_completely_different():
    result = dtw_align(["A", "B"], ["X", "Y"])
    assert result.distance == 2


def test_dtw_one_empty():
    result = dtw_align(["A", "B"], [])
    assert result.distance == 2
    assert result.normalized_distance == 1.0


# ===== Cart Diff Tests =====

def test_cart_diff_identical():
    cart = {"items": [{"product_code": "D001", "size": "grande"}]}
    diff = diff_cart_state(cart, cart)
    assert diff.is_equivalent is True
    assert len(diff.missing_items) == 0


def test_cart_diff_missing_item():
    expected = {"items": [{"product_code": "D001"}, {"product_code": "D002"}]}
    actual = {"items": [{"product_code": "D001"}]}
    diff = diff_cart_state(expected, actual)
    assert diff.is_equivalent is False
    assert len(diff.missing_items) == 1


def test_cart_diff_extra_item():
    expected = {"items": [{"product_code": "D001"}]}
    actual = {"items": [{"product_code": "D001"}, {"product_code": "D003"}]}
    diff = diff_cart_state(expected, actual)
    assert diff.is_equivalent is False
    assert len(diff.extra_items) == 1


def test_cart_diff_modified_field():
    expected = {"items": [{"product_code": "D001", "size": "tall"}]}
    actual = {"items": [{"product_code": "D001", "size": "grande"}]}
    diff = diff_cart_state(expected, actual)
    assert diff.is_equivalent is False
    assert len(diff.modified_items) == 1
    assert diff.modified_items[0]["field"] == "size"


# ===== TraceComparator Integration =====

def test_trace_comparator_compare():
    ref = _make_trace([["search_menu"], ["add_cart"], ["calculate_price"]])
    cand = _make_trace([["search_menu"], ["search_menu"], ["add_cart"], ["calculate_price"]])

    comparator = TraceComparator()
    result = comparator.compare_traces(ref, cand)
    assert result["alignment"]["distance"] >= 1  # candidate has extra search_menu


def test_trace_comparator_ground_truth():
    trace = _make_trace([["search_menu"], ["add_cart"]])
    trace.final_cart_state = {"items": [{"product_code": "D001", "size": "grande"}]}

    ground_truth = {"items": [{"product_code": "D001", "size": "tall"}]}

    comparator = TraceComparator()
    result = comparator.compare_to_ground_truth(trace, ground_truth)
    assert result["is_equivalent"] is False
    assert len(result["modified_items"]) == 1
