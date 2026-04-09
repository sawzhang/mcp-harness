"""
Eval Harness 核心逻辑测试（不调用真实模型，用 Mock Adapter）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harness.eval.adapters.base import AgentAdapter, DialogueResult, ToolCall
from harness.eval.harness import MCPEvalHarness
from harness.eval.case_loader import EvalCase
from harness.eval.report import generate_report


class MockAdapter(AgentAdapter):
    """模拟一个总是调用 expected_tool 的 Agent"""

    name = "mock"

    def __init__(self, tool_to_call: str = "calculate_price", params: dict = None):
        self._tool = tool_to_call
        self._params = params or {}

    async def run_dialogue(self, system_prompt, user_message, mcp_tools,
                           context=None, max_turns=5, timeout=30,
                           **kwargs) -> DialogueResult:
        tc = ToolCall(tool=self._tool, arguments=self._params)
        return DialogueResult(
            turns=[{"role": "assistant", "content": "好的", "tool_calls": [{"name": self._tool}]}],
            tool_calls=[tc],
            final_text="好的，已为您处理。",
            total_latency_ms=150,
            token_usage={"input": 100, "output": 50},
        )


class BadAdapter(AgentAdapter):
    """模拟一个选错 Tool 的 Agent"""

    name = "bad_model"

    async def run_dialogue(self, system_prompt, user_message, mcp_tools,
                           context=None, max_turns=5, timeout=30,
                           **kwargs) -> DialogueResult:
        tc = ToolCall(tool="create_order", arguments={"store_id": "ST_001"})
        return DialogueResult(
            turns=[],
            tool_calls=[tc],
            final_text="已下单。",
            total_latency_ms=200,
        )


MCP_TOOLS = [
    {"name": "calculate_price", "description": "计算价格", "inputSchema": {"type": "object"}},
    {"name": "create_order", "description": "创建订单", "inputSchema": {"type": "object"}},
    {"name": "browse_menu", "description": "浏览菜单", "inputSchema": {"type": "object"}},
]


@pytest.fixture
def harness():
    h = MCPEvalHarness(mcp_tools=MCP_TOOLS)
    return h


@pytest.mark.asyncio
async def test_tool_selection_pass(harness):
    harness.register_model("mock", MockAdapter(tool_to_call="calculate_price"), tier=1)

    case = EvalCase(
        id="test-001",
        layer="tool_selection",
        criticality="P0",
        user_instruction="来杯拿铁",
        expected_tool="calculate_price",
    )

    results = await harness.run_suite([case])
    assert len(results) == 1
    assert results[0].passed is True


@pytest.mark.asyncio
async def test_tool_selection_fail(harness):
    harness.register_model("bad", BadAdapter(), tier=1)

    case = EvalCase(
        id="test-002",
        layer="tool_selection",
        criticality="P0",
        user_instruction="看看菜单",
        expected_tool="browse_menu",
    )

    results = await harness.run_suite([case])
    assert len(results) == 1
    assert results[0].passed is False


@pytest.mark.asyncio
async def test_tool_exclusion(harness):
    harness.register_model("bad", BadAdapter(), tier=1)

    case = EvalCase(
        id="test-003",
        layer="tool_selection",
        criticality="P0",
        user_instruction="下单",
        expected_not_tool="create_order",
        expected_behavior="应先调 calculate_price",
    )

    results = await harness.run_suite([case])
    assert results[0].passed is False


@pytest.mark.asyncio
async def test_param_check(harness):
    harness.register_model(
        "mock",
        MockAdapter(tool_to_call="calculate_price", params={"items": [{"size": "tall"}]}),
        tier=1,
    )

    case = EvalCase(
        id="test-004",
        layer="param_mapping",
        criticality="P0",
        user_instruction="中杯冰美式",
        expected_tool="calculate_price",
        expected_params={"items": [{"size": "tall"}]},
    )

    results = await harness.run_suite([case])
    assert results[0].passed is True


@pytest.mark.asyncio
async def test_param_check_fail(harness):
    harness.register_model(
        "mock",
        MockAdapter(tool_to_call="calculate_price", params={"items": [{"size": "medium"}]}),
        tier=1,
    )

    case = EvalCase(
        id="test-005",
        layer="param_mapping",
        criticality="P0",
        user_instruction="中杯冰美式",
        expected_tool="calculate_price",
        expected_params={"items": [{"size": "tall"}]},
    )

    results = await harness.run_suite([case])
    assert results[0].passed is False


@pytest.mark.asyncio
async def test_multi_model_suite(harness):
    harness.register_model("good", MockAdapter(tool_to_call="calculate_price"), tier=1)
    harness.register_model("bad", BadAdapter(), tier=1)

    case = EvalCase(
        id="test-006",
        layer="tool_selection",
        criticality="P0",
        user_instruction="来杯拿铁",
        expected_tool="calculate_price",
    )

    results = await harness.run_suite([case])
    assert len(results) == 2

    good_result = next(r for r in results if r.model == "good")
    bad_result = next(r for r in results if r.model == "bad")
    assert good_result.passed is True
    assert bad_result.passed is False


@pytest.mark.asyncio
async def test_report_generation(harness):
    harness.register_model("good", MockAdapter(tool_to_call="calculate_price"), tier=1)
    harness.register_model("bad", BadAdapter(), tier=2)

    cases = [
        EvalCase(id="r-001", layer="tool_selection", criticality="P0",
                 user_instruction="来杯拿铁", expected_tool="calculate_price"),
        EvalCase(id="r-002", layer="tool_selection", criticality="P0",
                 user_instruction="看菜单", expected_tool="browse_menu"),
    ]

    results = await harness.run_suite(cases, tier_filter=2)
    report = generate_report(results)

    assert "pass_rate" in report
    assert "gate_decision" in report
    assert report["meta"]["total_cases"] == len(results)


@pytest.mark.asyncio
async def test_tier_filter(harness):
    harness.register_model("tier1", MockAdapter(), tier=1)
    harness.register_model("tier2", MockAdapter(), tier=2)
    harness.register_model("tier3", MockAdapter(), tier=3)

    case = EvalCase(id="t-001", layer="tool_selection", criticality="P0",
                    user_instruction="test", expected_tool="calculate_price")

    results_t1 = await harness.run_suite([case], tier_filter=1)
    results_t3 = await harness.run_suite([case], tier_filter=3)

    assert len(results_t1) == 1  # 只有 tier1
    assert len(results_t3) == 3  # 全部
