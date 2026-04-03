"""
Description Optimizer 运行入口

autoresearch 模式：固定评估 + 贪心爬山 + 自动迭代
修改 description → 跑 eval → 看 pass_rate → 保留/回滚 → 重复

用法：
    # 使用 Mock Adapter（无需 API Key，验证流程）
    python scripts/run_optimize.py --mock --rounds 5

    # 使用真实模型
    DASHSCOPE_API_KEY=xxx ANTHROPIC_API_KEY=xxx \\
    python scripts/run_optimize.py --model qianwen --rounds 10

    # 指定目标通过率
    python scripts/run_optimize.py --model qianwen --target 0.98
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.eval.harness import MCPEvalHarness
from harness.eval.case_loader import load_cases_from_dir
from harness.optimizer.loop import (
    DescriptionOptimizer,
    save_optimization_log,
    save_optimized_tools,
)


# 初始 Tool 定义（与 run_eval.py 相同）
MCP_TOOLS = [
    {
        "name": "nearby_stores",
        "description": "查找附近门店。当用户想点咖啡但还没选门店时使用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "keyword": {"type": "string", "description": "搜索关键词"},
            },
        },
    },
    {
        "name": "browse_menu",
        "description": "浏览门店菜单。当用户想看有什么可以点时使用。",
        "inputSchema": {
            "type": "object",
            "required": ["store_id"],
            "properties": {
                "store_id": {"type": "string", "description": "门店ID，从 nearby_stores 返回结果中获取"},
                "compact": {"type": "boolean", "description": "精简模式", "default": False},
            },
        },
    },
    {
        "name": "drink_detail",
        "description": "查看饮品详细信息和定制选项。",
        "inputSchema": {
            "type": "object",
            "required": ["product_code"],
            "properties": {
                "product_code": {"type": "string", "description": "商品编码"},
            },
        },
    },
    {
        "name": "calculate_price",
        "description": "计算订单价格。下单前必须先调用此工具。",
        "inputSchema": {
            "type": "object",
            "required": ["store_id", "items"],
            "properties": {
                "store_id": {"type": "string", "description": "门店ID"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["product_code", "quantity"],
                        "properties": {
                            "product_code": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "size": {"type": "string", "description": "杯型"},
                            "milk": {"type": "string", "description": "奶型"},
                            "temperature": {"type": "string", "description": "温度"},
                            "extras": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "coupon_code": {"type": "string"},
            },
        },
    },
    {
        "name": "create_order",
        "description": "创建订单。必须先调用 calculate_price。",
        "inputSchema": {
            "type": "object",
            "required": ["store_id", "items", "pickup_type", "idempotency_key", "confirmation_token"],
            "properties": {
                "store_id": {"type": "string"},
                "items": {"type": "array", "items": {"type": "object"}},
                "pickup_type": {"type": "string", "description": "取餐方式"},
                "idempotency_key": {"type": "string", "description": "幂等键"},
                "confirmation_token": {"type": "string", "description": "确认令牌"},
                "coupon_code": {"type": "string"},
                "address_id": {"type": "string"},
            },
        },
    },
    {
        "name": "order_status",
        "description": "查询订单状态。",
        "inputSchema": {
            "type": "object",
            "required": ["order_id"],
            "properties": {
                "order_id": {"type": "string", "description": "订单ID"},
            },
        },
    },
    {
        "name": "my_account",
        "description": "查看用户账户信息。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "my_coupons",
        "description": "查看用户的优惠券列表。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "筛选状态", "default": "valid"},
            },
        },
    },
    {
        "name": "nutrition_info",
        "description": "查看饮品营养信息。",
        "inputSchema": {
            "type": "object",
            "required": ["product_code"],
            "properties": {
                "product_code": {"type": "string", "description": "商品编码"},
                "compact": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "claim_all_coupons",
        "description": "一键领取所有可用优惠券。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="MCP Tool Description Optimizer")
    parser.add_argument("--model", type=str, help="Model to evaluate with")
    parser.add_argument("--rounds", type=int, default=10, help="Max optimization rounds")
    parser.add_argument("--target", type=float, default=0.98, help="Target pass rate")
    parser.add_argument("--cases", type=str, default="evals/cases", help="Eval cases directory")
    parser.add_argument("--output", type=str, default="evals/optimized", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock adapter (no API key needed)")
    parser.add_argument("--analyzer", type=str, default="claude-sonnet-4-20250514",
                        help="Model for analyzing failures")
    return parser.parse_args()


class MockAdapterForOptimizer:
    """
    Mock Adapter：根据 Tool Description 的质量模拟不同的通过率。
    description 越详细（三段式、有触发短语、有参数来源），通过率越高。
    用于无 API Key 时验证优化流程。
    """
    name = "mock-evaluator"

    async def run_dialogue(self, system_prompt, user_message, mcp_tools,
                           context=None, max_turns=5, timeout=30):
        from harness.eval.adapters.base import DialogueResult, ToolCall

        # 根据 user_message 中的关键词选择 tool
        tool_map = {
            "菜单": "browse_menu",
            "卡路里": "nutrition_info",
            "热量": "nutrition_info",
            "优惠券": "my_coupons",
            "领券": "claim_all_coupons",
            "订单": "order_status",
            "门店": "nearby_stores",
            "账户": "my_account",
        }

        selected_tool = "calculate_price"  # default
        for keyword, tool_name in tool_map.items():
            if keyword in user_message:
                selected_tool = tool_name
                break

        # 模拟杯型映射——根据 description 质量决定是否映射正确
        size = "grande"  # default
        tool_desc = ""
        for t in mcp_tools:
            if t["name"] == "calculate_price":
                tool_desc = t.get("description", "")
                size_desc = (t.get("inputSchema", {})
                              .get("properties", {})
                              .get("items", {})
                              .get("items", {})
                              .get("properties", {})
                              .get("size", {})
                              .get("description", ""))
                tool_desc += " " + size_desc

        if "中杯" in user_message:
            # 如果 description 包含 tall(中杯) 映射，则正确映射
            if "tall" in tool_desc.lower() and "中杯" in tool_desc:
                size = "tall"  # 正确
            else:
                size = "medium"  # 错误——description 不够明确

        if "大杯" in user_message:
            if "grande" in tool_desc.lower() and "大杯" in tool_desc:
                size = "grande"
            else:
                size = "large"

        params = {"items": [{"size": size, "product_code": "D001", "quantity": 1}]}

        tc = ToolCall(tool=selected_tool, arguments=params)
        return DialogueResult(
            turns=[{"role": "assistant", "content": "好的"}],
            tool_calls=[tc],
            final_text="好的，已为您处理。",
            total_latency_ms=50,
            token_usage={"input": 50, "output": 20},
        )


async def run():
    args = parse_args()
    cases_dir = Path(args.cases)

    cases = load_cases_from_dir(cases_dir)
    if not cases:
        print("No eval cases found.")
        sys.exit(1)

    print(f"Loaded {len(cases)} eval cases")
    print(f"Max rounds: {args.rounds}, Target: {args.target:.0%}")

    # 创建 Harness
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS)

    if args.mock:
        harness.register_model("mock", MockAdapterForOptimizer(), tier=1)
        analyzer_client = None  # 用 fallback 分析
        print("Using mock adapter (no API calls)")
    else:
        # 注册真实模型
        if args.model == "qianwen" or not args.model:
            from harness.eval.adapters.openai_adapter import OpenAIAdapter
            qw_key = os.environ.get("QIANWEN_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
            if qw_key:
                harness.register_model("qianwen-max", OpenAIAdapter(
                    model="qwen-max",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    api_key=qw_key, name="qianwen-max",
                ), tier=1)

        if args.model == "claude" or not args.model:
            from harness.eval.adapters.claude_adapter import ClaudeAdapter
            ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if ant_key:
                harness.register_model("claude-sonnet", ClaudeAdapter(
                    api_key=ant_key, name="claude-sonnet",
                ), tier=1)

        analyzer_client = None  # 使用默认客户端

    if not harness.models:
        print("No models configured. Use --mock or set API keys.")
        sys.exit(1)

    # 运行优化循环
    print(f"\n{'='*60}")
    print(f"  MCP Tool Description Optimizer")
    print(f"  Pattern: autoresearch (greedy hill-climbing)")
    print(f"{'='*60}")

    optimizer = DescriptionOptimizer(
        tools=MCP_TOOLS,
        eval_harness=harness,
        eval_cases=cases,
        analyzer_client=analyzer_client,
        analyzer_model=args.analyzer,
    )

    log = await optimizer.run(
        max_rounds=args.rounds,
        target_pass_rate=args.target,
    )

    # 保存结果
    output_dir = Path(args.output)
    save_optimization_log(log, output_dir / "optimization_log.yaml")
    save_optimized_tools(log.best_tools, output_dir / "optimized_tools.yaml")

    print(f"\n{'='*60}")
    print(f"  Optimization Complete")
    print(f"  Best pass rate: {log.best_pass_rate:.1%} (round {log.best_round})")
    print(f"  Total rounds: {len(log.rounds)}")
    print(f"  Kept: {sum(1 for r in log.rounds if r.kept)}")
    print(f"  Reverted: {sum(1 for r in log.rounds if not r.kept)}")
    print(f"  Results: {output_dir}/")
    print(f"{'='*60}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
