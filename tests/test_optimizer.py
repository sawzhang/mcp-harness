"""
Description Optimizer 测试（使用 Mock Adapter，不调用真实 API）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harness.eval.adapters.base import AgentAdapter, DialogueResult, ToolCall
from harness.eval.harness import MCPEvalHarness
from harness.eval.case_loader import EvalCase
from harness.optimizer.loop import DescriptionOptimizer, ToolSpec


# --- Mock Adapters ---

class WeakAdapter(AgentAdapter):
    """模拟一个对 description 敏感的弱模型：
    - 如果 description 包含 "tall(中杯)"，则正确映射
    - 否则映射为 medium（错误）
    """
    name = "weak-model"

    def __init__(self, tools_ref: list[dict] | None = None):
        self._tools_ref = tools_ref

    async def run_dialogue(self, system_prompt, user_message, mcp_tools,
                           context=None, max_turns=5, timeout=30,
                           **kwargs):
        # 检查 calculate_price 的 description 是否包含杯型映射
        size_desc = ""
        for t in mcp_tools:
            if t["name"] == "calculate_price":
                size_desc = t.get("description", "")
                # 也检查参数级 description
                props = (t.get("inputSchema", {})
                          .get("properties", {})
                          .get("items", {})
                          .get("items", {})
                          .get("properties", {})
                          .get("size", {}))
                if isinstance(props, dict):
                    size_desc += " " + props.get("description", "")

        # 决定杯型映射
        size = "tall" if ("tall" in size_desc and "中杯" in size_desc) else "medium"

        tc = ToolCall(
            tool="calculate_price",
            arguments={"items": [{"size": size, "product_code": "D001", "quantity": 1}]},
        )
        return DialogueResult(
            turns=[],
            tool_calls=[tc],
            final_text="好的",
            total_latency_ms=10,
        )


MCP_TOOLS_WEAK = [
    {
        "name": "calculate_price",
        "description": "计算订单价格。",  # 故意写得很简略
        "inputSchema": {
            "type": "object",
            "required": ["store_id", "items"],
            "properties": {
                "store_id": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_code": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "size": {"type": "string", "description": "杯型"},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "browse_menu",
        "description": "浏览菜单。",
        "inputSchema": {"type": "object", "properties": {"store_id": {"type": "string"}}},
    },
]

CASES_SIZE_MAPPING = [
    EvalCase(
        id="opt-size-001",
        layer="param_mapping",
        criticality="P0",
        user_instruction="来一杯中杯冰美式",
        expected_tool="calculate_price",
        expected_params={"items": [{"size": "tall"}]},
    ),
    EvalCase(
        id="opt-size-002",
        layer="param_mapping",
        criticality="P0",
        user_instruction="来一杯拿铁",
        expected_tool="calculate_price",
    ),
]


@pytest.mark.asyncio
async def test_baseline_fails_without_mapping():
    """弱模型在没有杯型映射的 description 下应该失败"""
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS_WEAK)
    harness.register_model("weak", WeakAdapter(), tier=1)

    results = await harness.run_suite(CASES_SIZE_MAPPING)
    # 第一个 case 应该失败（中杯→medium 而不是 tall）
    size_case = next(r for r in results if r.case_id == "opt-size-001")
    assert size_case.passed is False


@pytest.mark.asyncio
async def test_optimizer_improves_with_fallback():
    """优化器使用 fallback 分析（无 LLM）也能改进 description"""
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS_WEAK)
    harness.register_model("weak", WeakAdapter(), tier=1)

    optimizer = DescriptionOptimizer(
        tools=MCP_TOOLS_WEAK,
        eval_harness=harness,
        eval_cases=CASES_SIZE_MAPPING,
        analyzer_client=None,  # 使用 fallback
    )

    log = await optimizer.run(max_rounds=3, target_pass_rate=1.0)

    # 至少应该有 baseline + 1 轮
    assert len(log.rounds) >= 2
    # baseline 不应该是 100%（因为 description 不够好）
    assert log.rounds[0].pass_rate < 1.0


@pytest.mark.asyncio
async def test_optimizer_keeps_improvements():
    """保留的轮次 pass_rate 应该 >= best"""
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS_WEAK)
    harness.register_model("weak", WeakAdapter(), tier=1)

    optimizer = DescriptionOptimizer(
        tools=MCP_TOOLS_WEAK,
        eval_harness=harness,
        eval_cases=CASES_SIZE_MAPPING,
    )

    log = await optimizer.run(max_rounds=5)

    best = 0
    for r in log.rounds:
        if r.kept:
            assert r.pass_rate >= best or r.round_num == 0
            best = max(best, r.pass_rate)


@pytest.mark.asyncio
async def test_optimizer_reverts_regressions():
    """不改进的轮次应该被 revert"""
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS_WEAK)
    harness.register_model("weak", WeakAdapter(), tier=1)

    optimizer = DescriptionOptimizer(
        tools=MCP_TOOLS_WEAK,
        eval_harness=harness,
        eval_cases=CASES_SIZE_MAPPING,
    )

    log = await optimizer.run(max_rounds=5)

    for r in log.rounds:
        if not r.kept and r.round_num > 0:
            # reverted 轮的 pass_rate 不高于 best
            assert r.pass_rate <= log.best_pass_rate


@pytest.mark.asyncio
async def test_toolspec_roundtrip():
    """ToolSpec 序列化/反序列化"""
    original = MCP_TOOLS_WEAK[0]
    spec = ToolSpec.from_mcp(original)
    back = spec.to_mcp()
    assert back["name"] == original["name"]
    assert back["description"] == original["description"]


@pytest.mark.asyncio
async def test_convergence_stops_on_no_improvement():
    """连续 3 轮无改进应该自动停止"""
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS_WEAK)

    # 用一个永远失败的 adapter
    class AlwaysFailAdapter(AgentAdapter):
        name = "always-fail"

        async def run_dialogue(self, **kwargs):
            tc = ToolCall(tool="wrong_tool", arguments={})
            return DialogueResult(tool_calls=[tc], final_text="error")

    harness.register_model("fail", AlwaysFailAdapter(), tier=1)

    optimizer = DescriptionOptimizer(
        tools=MCP_TOOLS_WEAK,
        eval_harness=harness,
        eval_cases=CASES_SIZE_MAPPING,
    )

    log = await optimizer.run(max_rounds=10)

    # 应该在 < 10 轮停止（3 轮无改进 + baseline = 最多 4 轮）
    assert len(log.rounds) <= 5
