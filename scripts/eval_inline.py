"""
Inline Eval — 用 Claude Code 自身作为被测 Agent

不走 API adapter，而是把每个 Eval Case 构造成一个 prompt，
让 Claude Code（通过 subprocess 调用 claude CLI）直接回答，
然后解析输出中的 tool_call 决策。

用法：
    uv run python scripts/eval_inline.py --cases evals/cases/ --limit 5
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.eval.case_loader import load_cases_from_dir, EvalCase
from harness.eval.harness import _param_match
from harness.eval.report import generate_report, print_report, save_report


TOOL_SPECS_PATH = Path(__file__).parent.parent / "evals" / "tool_specs.yaml"

EVAL_PROMPT_TEMPLATE = """你是一个咖啡点单 AI 助手的决策模拟器。给定用户指令和可用的 MCP 工具列表，你需要判断应该调用哪个工具、传什么参数。

## 可用工具（仅列出 name 和 description 摘要）

{tools_summary}

## 用户指令

{user_instruction}

{context_section}

## 要求

请严格按以下 JSON 格式输出你的决策（只输出 JSON，不要其他文字）：

```json
{{
  "tool": "工具名称",
  "arguments": {{参数键值对}}
}}
```

规则：
- 星巴克杯型：中杯=tall, 大杯=grande, 超大杯=venti（不是 medium/large）
- 下单前必须先调 calculate_price，不能直接调 create_order
- 如果用户要看菜单/查信息，选对应的查询工具
- 如果不确定用户意图，选最合理的工具"""


def load_tool_specs() -> list[dict]:
    if TOOL_SPECS_PATH.exists():
        with open(TOOL_SPECS_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Fallback to hardcoded
    from scripts.run_eval import MCP_TOOLS
    return MCP_TOOLS


def build_tools_summary(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        desc = t.get("description", "").split("\n")[0][:80]
        params = list(t.get("inputSchema", {}).get("properties", {}).keys())
        params_str = f"({', '.join(params)})" if params else "(无参数)"
        lines.append(f"- {t['name']}{params_str}: {desc}")
    return "\n".join(lines)


def build_prompt(case: EvalCase, tools: list[dict]) -> str:
    tools_summary = build_tools_summary(tools)
    context_section = ""
    if case.order_context:
        context_section = f"## 当前上下文\n{json.dumps(case.order_context, ensure_ascii=False, indent=2)}"

    return EVAL_PROMPT_TEMPLATE.format(
        tools_summary=tools_summary,
        user_instruction=case.user_instruction,
        context_section=context_section,
    )


def call_claude(prompt: str, timeout: int = 30) -> str:
    """通过 claude CLI 调用当前 Claude Code 实例的模型"""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return '{"error": "timeout"}'
    except FileNotFoundError:
        return '{"error": "claude CLI not found"}'


def parse_tool_decision(response: str) -> dict:
    """从 Claude 响应中提取 tool call JSON"""
    # 尝试提取 ```json ... ``` 块
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        text = response

    # 找 { ... }
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return {"tool": "", "arguments": {}, "parse_error": True}

    try:
        parsed = json.loads(text[start:end + 1])
        return {
            "tool": parsed.get("tool", ""),
            "arguments": parsed.get("arguments", {}),
        }
    except json.JSONDecodeError:
        return {"tool": "", "arguments": {}, "parse_error": True}


def check_case(case: EvalCase, decision: dict) -> dict:
    """检查单个 case 的通过/失败"""
    checks = []

    if case.expected_tool:
        passed = decision["tool"] == case.expected_tool
        checks.append({
            "name": "tool_selection",
            "passed": passed,
            "detail": f"Expected '{case.expected_tool}', got '{decision['tool']}'",
        })

    if case.expected_not_tool:
        passed = decision["tool"] != case.expected_not_tool
        checks.append({
            "name": "tool_exclusion",
            "passed": passed,
            "detail": f"'{case.expected_not_tool}' should NOT be called",
        })

    if case.expected_params and case.expected_tool and decision["tool"] == case.expected_tool:
        for key, expected_val in case.expected_params.items():
            actual = decision["arguments"].get(key)
            passed = _param_match(expected_val, actual)
            checks.append({
                "name": f"param_{key}",
                "passed": passed,
                "detail": f"Expected {key}={expected_val}, got {actual}",
            })

    all_passed = all(c["passed"] for c in checks) if checks else True

    return {
        "case_id": case.id,
        "model": "claude-code",
        "layer": case.layer,
        "passed": all_passed,
        "checks": checks,
        "tool_calls": [{"tool": decision["tool"], "args": decision["arguments"]}],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inline Eval using Claude Code")
    parser.add_argument("--cases", type=str, default="evals/cases")
    parser.add_argument("--limit", type=int, default=0, help="Max cases to run (0=all)")
    parser.add_argument("--layer", type=str, help="Filter by layer")
    parser.add_argument("--criticality", type=str, default="P0")
    parser.add_argument("--report", type=str, help="Save report to YAML")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tools = load_tool_specs()
    cases = load_cases_from_dir(args.cases, layer=args.layer, criticality=args.criticality)

    # 跳过 multi_turn（inline 模式不支持）
    cases = [c for c in cases if c.layer != "multi_turn" and c.user_instruction]

    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"Tools: {len(tools)}, Cases: {len(cases)} (P0, excluding multi-turn)")
    print(f"Model: claude-code (current instance)")
    print(f"{'='*60}")

    results = []
    passed_count = 0

    for i, case in enumerate(cases):
        prompt = build_prompt(case, tools)
        response = call_claude(prompt)
        decision = parse_tool_decision(response)
        result = check_case(case, decision)
        results.append(result)

        if result["passed"]:
            passed_count += 1
            status = "✅"
        else:
            status = "❌"

        print(f"  [{i+1}/{len(cases)}] {status} {case.id}: "
              f"{case.user_instruction[:40]}... → {decision['tool']}")

        if args.verbose and not result["passed"]:
            for c in result["checks"]:
                if not c["passed"]:
                    print(f"         FAIL: {c['detail']}")

    pass_rate = passed_count / len(cases) if cases else 0
    print(f"\n{'='*60}")
    print(f"  Pass rate: {pass_rate:.1%} ({passed_count}/{len(cases)})")
    print(f"  Gate: {'PASS' if pass_rate >= 0.95 else 'BLOCK'}")
    print(f"{'='*60}")

    if args.report:
        # 转换为 EvalResult 格式用 report 模块
        from harness.eval.harness import EvalResult
        eval_results = [
            EvalResult(
                case_id=r["case_id"], model=r["model"], layer=r["layer"],
                passed=r["passed"], checks=r["checks"], tool_calls=r["tool_calls"],
            )
            for r in results
        ]
        report = generate_report(eval_results)
        save_report(report, args.report)
        print(f"  Report saved to {args.report}")


if __name__ == "__main__":
    main()
