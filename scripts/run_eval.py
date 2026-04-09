"""
Eval 运行入口

用法：
    # 跑 Qianwen（Tier 1，每次 PR）
    python scripts/run_eval.py --model qianwen --cases evals/cases/

    # 跑全部模型（Tier 1-3）
    python scripts/run_eval.py --tier 3 --cases evals/cases/

    # 只跑 P0 用例
    python scripts/run_eval.py --model qianwen --criticality P0

    # 生成报告
    python scripts/run_eval.py --model qianwen --report --output evals/reports/latest.yaml
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.eval.harness import MCPEvalHarness
from harness.eval.case_loader import load_cases_from_dir
from harness.eval.report import generate_report, save_report, print_report


# MCP Tool specs — 从 coffee-mcp 的 toc_server 提取的工具定义
# 实际使用时应从 MCP Server 动态加载
MCP_TOOLS = [
    {
        "name": "nearby_stores",
        "description": "查找附近门店。当用户想点咖啡但还没选门店时使用。当用户说'附近有什么店'、'找个门店'时触发。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如'上海'、'北京'"},
                "keyword": {"type": "string", "description": "搜索关键词"},
            },
        },
    },
    {
        "name": "browse_menu",
        "description": "浏览门店菜单。当用户想看有什么可以点时使用。当用户说'看看菜单'、'有什么喝的'时触发。store_id 从 nearby_stores 返回结果中获取。",
        "inputSchema": {
            "type": "object",
            "required": ["store_id"],
            "properties": {
                "store_id": {"type": "string", "description": "门店ID，从 nearby_stores 返回结果中获取"},
                "compact": {"type": "boolean", "description": "是否精简模式（减少 token 消耗）", "default": False},
            },
        },
    },
    {
        "name": "drink_detail",
        "description": "查看饮品详细信息和可定制选项。当用户想了解某个饮品的杯型、奶型、温度等选项时使用。product_code 从 browse_menu 返回结果中获取。",
        "inputSchema": {
            "type": "object",
            "required": ["product_code"],
            "properties": {
                "product_code": {"type": "string", "description": "商品编码，从 browse_menu 返回结果中获取"},
            },
        },
    },
    {
        "name": "calculate_price",
        "description": "计算订单价格。下单前必须先调用此工具获取价格和 confirmation_token。当用户选好商品准备下单时使用。返回的 confirmation_token 需要传给 create_order。",
        "inputSchema": {
            "type": "object",
            "required": ["store_id", "items"],
            "properties": {
                "store_id": {"type": "string", "description": "门店ID，从 nearby_stores 获取"},
                "items": {
                    "type": "array",
                    "description": "商品列表",
                    "items": {
                        "type": "object",
                        "required": ["product_code", "quantity"],
                        "properties": {
                            "product_code": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 1},
                            "size": {"type": "string", "description": "杯型：tall(中杯12oz) | grande(大杯16oz) | venti(超大杯20oz)"},
                            "milk": {"type": "string", "description": "奶型：whole | skim | oat | almond | soy | coconut"},
                            "temperature": {"type": "string", "description": "温度：hot | iced | blended"},
                            "extras": {"type": "array", "items": {"type": "string"}, "description": "额外选项：extra_shot, vanilla_syrup 等"},
                        },
                    },
                },
                "coupon_code": {"type": "string", "description": "优惠券码（可选）"},
            },
        },
    },
    {
        "name": "create_order",
        "description": "创建订单（L3高风险操作）。必须先调用 calculate_price 获取 confirmation_token。当用户确认价格后说'下单'、'确认'时使用。需要 idempotency_key 防止重复下单。",
        "inputSchema": {
            "type": "object",
            "required": ["store_id", "items", "pickup_type", "idempotency_key", "confirmation_token"],
            "properties": {
                "store_id": {"type": "string"},
                "items": {"type": "array", "items": {"type": "object"}},
                "pickup_type": {"type": "string", "description": "取餐方式：自提 | 外送 | 堂食"},
                "idempotency_key": {"type": "string", "description": "幂等键，UUID 格式，防止重复下单"},
                "confirmation_token": {"type": "string", "description": "确认令牌，从 calculate_price 返回值中获取"},
                "coupon_code": {"type": "string"},
                "address_id": {"type": "string", "description": "外送地址ID（pickup_type=外送 时必填）"},
            },
        },
    },
    {
        "name": "order_status",
        "description": "查询订单状态。当用户问'我的订单怎么样了'、'做好了吗'时使用。order_id 从 create_order 返回结果中获取。",
        "inputSchema": {
            "type": "object",
            "required": ["order_id"],
            "properties": {
                "order_id": {"type": "string", "description": "订单ID，从 create_order 返回结果中获取"},
            },
        },
    },
    {
        "name": "my_account",
        "description": "查看用户账户信息，包括会员等级、星星余额、权益。当用户问'我的账户'、'我有多少星星'时使用。无需传参，从 token 推断用户身份。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "my_coupons",
        "description": "查看用户的优惠券列表。当用户问'我有什么券'、'有没有优惠'时使用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "筛选状态：valid(有效) | used(已用) | all(全部)", "default": "valid"},
            },
        },
    },
    {
        "name": "nutrition_info",
        "description": "查看饮品营养信息（卡路里、蛋白质等）。当用户问'多少卡'、'热量高吗'时使用。",
        "inputSchema": {
            "type": "object",
            "required": ["product_code"],
            "properties": {
                "product_code": {"type": "string", "description": "商品编码，从 browse_menu 获取"},
                "compact": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "claim_all_coupons",
        "description": "一键领取所有可用优惠券（L2写操作）。当用户说'帮我领券'、'一键领取'时使用。无需传参。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="MCP Harness Eval Runner")
    parser.add_argument("--model", type=str, help="Specific model to test (qianwen/claude/gpt4o)")
    parser.add_argument("--tier", type=int, default=1, help="Max tier to include (1-3)")
    parser.add_argument("--cases", type=str, default="evals/cases", help="Eval cases directory")
    parser.add_argument("--criticality", type=str, help="Filter by criticality (P0/P1/P2)")
    parser.add_argument("--layer", type=str, help="Filter by layer")
    parser.add_argument("--report", action="store_true", help="Generate YAML report")
    parser.add_argument("--output", type=str, default="evals/reports/latest.yaml", help="Report output path")
    parser.add_argument("--dry-run", action="store_true", help="List cases without running")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-Judge for behavior assertions")
    parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514", help="Model for judge")
    parser.add_argument("--fingerprint", action="store_true", help="Include fingerprint matrix in report")
    parser.add_argument("--behavior", action="store_true", help="Run agent behavior analysis")
    return parser.parse_args()


def create_harness(args) -> MCPEvalHarness:
    harness = MCPEvalHarness(mcp_tools=MCP_TOOLS)

    # 注册模型 — 根据环境变量配置
    if args.model == "qianwen" or not args.model:
        from harness.eval.adapters.openai_adapter import OpenAIAdapter
        qw_key = os.environ.get("QIANWEN_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
        if qw_key:
            harness.register_model("qianwen-max", OpenAIAdapter(
                model="qwen-max",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=qw_key,
                name="qianwen-max",
            ), tier=1)

    if args.model == "gpt4o" or (not args.model and args.tier >= 2):
        from harness.eval.adapters.openai_adapter import OpenAIAdapter
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        if oai_key:
            harness.register_model("gpt-4o", OpenAIAdapter(
                model="gpt-4o",
                api_key=oai_key,
                name="gpt-4o",
            ), tier=2)

    if args.model == "claude" or (not args.model and args.tier >= 2):
        from harness.eval.adapters.claude_adapter import ClaudeAdapter
        ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if ant_key:
            harness.register_model("claude-sonnet", ClaudeAdapter(
                model="claude-sonnet-4-20250514",
                api_key=ant_key,
                name="claude-sonnet",
            ), tier=2)

    return harness


async def run():
    args = parse_args()
    cases_dir = Path(args.cases)

    if not cases_dir.exists():
        print(f"Cases directory not found: {cases_dir}")
        sys.exit(1)

    cases = load_cases_from_dir(
        cases_dir,
        layer=args.layer,
        criticality=args.criticality,
    )

    if not cases:
        print("No eval cases found.")
        sys.exit(1)

    print(f"Loaded {len(cases)} eval cases from {cases_dir}")

    if args.dry_run:
        for c in cases:
            print(f"  [{c.criticality}] [{c.layer}] {c.id}: {c.user_instruction[:60]}")
        return

    harness = create_harness(args)

    # Set up LLM-as-Judge if requested
    if args.judge:
        from harness.eval.judge import LLMJudge
        judge_client = None
        ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if ant_key:
            try:
                from anthropic import AsyncAnthropic
                judge_client = AsyncAnthropic(api_key=ant_key)
            except ImportError:
                pass
        harness.judge = LLMJudge(client=judge_client, model=args.judge_model)
        print(f"LLM-as-Judge enabled (model: {args.judge_model})")

    if not harness.models:
        print("No models configured. Set API keys via environment variables:")
        print("  QIANWEN_API_KEY / DASHSCOPE_API_KEY")
        print("  OPENAI_API_KEY")
        print("  ANTHROPIC_API_KEY")
        sys.exit(1)

    print(f"Running with models: {list(harness.models.keys())}")
    results = await harness.run_suite(cases, tier_filter=args.tier)

    report = generate_report(results, include_fingerprint=args.fingerprint)
    print_report(report)

    # Agent behavior analysis
    if args.behavior:
        from harness.agent.trace import AgentTrace, TurnTrace, ToolCallRecord
        from harness.agent.behavior import analyze_behavior

        print(f"\n--- Agent Behavior Analysis ---")
        for r in results:
            # Build trace from eval result
            tc_records = [
                ToolCallRecord(tool=tc["tool"], arguments=tc.get("args", {}))
                for tc in r.tool_calls
            ]
            trace = AgentTrace(
                case_id=r.case_id,
                agent_name=r.model,
                model_name=r.model,
                turns=[TurnTrace(turn_number=1, tool_calls=tc_records)],
                total_tool_calls=len(r.tool_calls),
                total_latency_ms=r.latency_ms,
            )
            behavior = analyze_behavior(trace)
            if behavior.loops.total_loops_detected > 0 or behavior.planning.redundant_search_rate > 0:
                print(f"  {r.case_id} ({r.model}): "
                      f"steps={behavior.planning.plan_step_count} "
                      f"loops={behavior.loops.total_loops_detected} "
                      f"redundant={behavior.planning.redundant_search_rate:.0%}")

    if args.report:
        save_report(report, args.output)
        print(f"\nReport saved to {args.output}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
