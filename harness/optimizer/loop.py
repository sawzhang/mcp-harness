"""
Description Optimizer — autoresearch 模式的 MCP Tool Description 自动优化

核心思路（借鉴 autoresearch）：
  autoresearch:  修改 train.py → 跑训练 → 看 val_bpb → 保留/回滚
  这里：         修改 description → 跑 eval → 看 pass_rate → 保留/回滚

三个角色（类比 autoresearch 的三文件架构）：
  prepare.py  →  Eval Harness（固定，不可修改的评估基础设施）
  train.py    →  Tool Descriptions（Agent 可修改的优化对象）
  program.md  →  优化策略指令（约束和目标）

用法：
    python -m harness.optimizer.loop \\
        --tools tools.yaml \\
        --cases evals/cases/ \\
        --model qianwen \\
        --rounds 10
"""

from __future__ import annotations

import asyncio
import copy
import json
import yaml
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    def to_mcp(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    @staticmethod
    def from_mcp(d: dict) -> "ToolSpec":
        return ToolSpec(
            name=d["name"],
            description=d.get("description", ""),
            input_schema=d.get("inputSchema", {}),
        )


@dataclass
class RoundResult:
    round_num: int
    pass_rate: float
    total_cases: int
    passed_cases: int
    failed_cases: list[dict] = field(default_factory=list)
    changes_made: list[dict] = field(default_factory=list)  # [{tool, field, old, new, reason}]
    kept: bool = False
    duration_ms: float = 0


@dataclass
class OptimizationLog:
    started_at: str = ""
    rounds: list[RoundResult] = field(default_factory=list)
    best_pass_rate: float = 0
    best_round: int = 0
    best_tools: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "total_rounds": len(self.rounds),
            "best_pass_rate": self.best_pass_rate,
            "best_round": self.best_round,
            "trajectory": [
                {
                    "round": r.round_num,
                    "pass_rate": r.pass_rate,
                    "kept": r.kept,
                    "changes": len(r.changes_made),
                    "duration_ms": r.duration_ms,
                }
                for r in self.rounds
            ],
            "best_tools": self.best_tools,
        }


ANALYZER_PROMPT = """你是 MCP Tool Description 优化专家。

当前的 Tool Description 在多模型评估中出现了以下失败。请分析失败原因，并给出具体的 description 修改建议。

## 约束
1. 只修改 description 和 inputSchema 中的 description 字段
2. 不修改 tool name、参数名、参数类型
3. 修改必须让更多模型能正确理解和调用工具
4. 星巴克杯型映射：中杯=tall, 大杯=grande, 超大杯=venti（必须在 description 中显式说明）
5. 参数来源标注：引用型参数（_id, _code）必须说明从哪个 tool 获取

## 当前 Tool 定义

{tools_json}

## 失败的 Eval Case

{failures_json}

## 历史优化记录（已尝试过的修改）

{history_json}

## 要求

针对失败最多的 Tool，给出修改建议。输出严格 JSON 格式：

```json
[
  {{
    "tool": "tool_name",
    "field": "description 或 inputSchema.properties.xxx.description",
    "old": "原文本",
    "new": "修改后文本",
    "reason": "修改原因（对应哪个失败 Case）"
  }}
]
```

每轮最多修改 3 个字段。优先修改失败最多的 Tool。不要重复之前已尝试过但无效的修改。"""


class DescriptionOptimizer:
    """
    autoresearch 模式的 Description 优化循环：

    1. Baseline: 跑一轮 Eval，记录 pass_rate
    2. Analyze: LLM 分析失败原因，提出 description 修改
    3. Apply: 应用修改到 Tool specs
    4. Evaluate: 重新跑 Eval
    5. Decision: pass_rate 提升 → 保留；否则回滚
    6. Repeat until 收敛或达到 max_rounds
    """

    def __init__(
        self,
        tools: list[dict],
        eval_harness,
        eval_cases: list,
        analyzer_client=None,
        analyzer_model: str = "claude-sonnet-4-20250514",
    ):
        self.original_tools = [ToolSpec.from_mcp(t) for t in tools]
        self.current_tools = [ToolSpec.from_mcp(t) for t in tools]
        self.harness = eval_harness
        self.cases = eval_cases
        self.analyzer_client = analyzer_client
        self.analyzer_model = analyzer_model
        self.log = OptimizationLog(started_at=datetime.now().isoformat())

    async def run(
        self,
        max_rounds: int = 10,
        convergence_threshold: float = 0.001,
        target_pass_rate: float = 0.98,
        tier_filter: int = 1,
    ) -> OptimizationLog:
        # Round 0: Baseline
        baseline = await self._evaluate(tier_filter)
        self.log.best_pass_rate = baseline.pass_rate
        self.log.best_round = 0
        self.log.best_tools = [t.to_mcp() for t in self.current_tools]
        baseline.round_num = 0
        baseline.kept = True
        self.log.rounds.append(baseline)

        _print_round(baseline, is_baseline=True)

        rounds_since_improvement = 0

        for round_num in range(1, max_rounds + 1):
            # 已达目标
            if self.log.best_pass_rate >= target_pass_rate:
                print(f"\n  Target pass rate {target_pass_rate:.1%} reached. Stopping.")
                break

            # 收敛检查：自上次改进以来连续 3 轮无进展
            if rounds_since_improvement >= 3:
                print(f"\n  {rounds_since_improvement} rounds without improvement. Stopping.")
                break

            # 分析失败 + 生成修改建议
            changes = await self._analyze_and_propose(round_num)
            if not changes:
                print(f"\n  Round {round_num}: No changes proposed. Stopping.")
                break

            # 保存当前状态（用于回滚）
            snapshot = [copy.deepcopy(t) for t in self.current_tools]

            # 应用修改
            self._apply_changes(changes)

            # 重新评估
            start = time.monotonic()
            result = await self._evaluate(tier_filter)
            result.round_num = round_num
            result.changes_made = changes
            result.duration_ms = (time.monotonic() - start) * 1000

            # Decision: 保留 or 回滚
            improvement = result.pass_rate - self.log.best_pass_rate

            if improvement > 0:
                # KEEP
                result.kept = True
                self.log.best_pass_rate = result.pass_rate
                self.log.best_round = round_num
                self.log.best_tools = [t.to_mcp() for t in self.current_tools]
                rounds_since_improvement = 0
                _print_round(result, improvement=improvement)
            else:
                # REVERT
                result.kept = False
                self.current_tools = snapshot
                self.harness.mcp_tools = [t.to_mcp() for t in self.current_tools]
                rounds_since_improvement += 1
                _print_round(result, improvement=improvement)

            self.log.rounds.append(result)

        return self.log

    async def _evaluate(self, tier_filter: int) -> RoundResult:
        self.harness.mcp_tools = [t.to_mcp() for t in self.current_tools]
        results = await self.harness.run_suite(self.cases, tier_filter=tier_filter)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / total if total else 0

        failures = []
        for r in results:
            if not r.passed:
                failures.append({
                    "case_id": r.case_id,
                    "model": r.model,
                    "layer": r.layer,
                    "checks": r.checks,
                    "tool_calls": r.tool_calls,
                    "error": r.error,
                })

        return RoundResult(
            round_num=0,
            pass_rate=pass_rate,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failures,
        )

    async def _analyze_and_propose(self, round_num: int) -> list[dict]:
        last_round = self.log.rounds[-1]
        if not last_round.failed_cases:
            return []

        # 构建历史（避免重复无效修改）
        history = []
        for r in self.log.rounds:
            if r.changes_made:
                history.append({
                    "round": r.round_num,
                    "kept": r.kept,
                    "changes": r.changes_made,
                })

        tools_json = json.dumps(
            [t.to_mcp() for t in self.current_tools],
            ensure_ascii=False, indent=2,
        )
        failures_json = json.dumps(
            last_round.failed_cases[:15],  # 最多 15 个失败
            ensure_ascii=False, indent=2,
        )
        history_json = json.dumps(history, ensure_ascii=False, indent=2)

        prompt = ANALYZER_PROMPT.format(
            tools_json=tools_json,
            failures_json=failures_json,
            history_json=history_json,
        )

        if self.analyzer_client is None:
            return self._fallback_analysis(last_round)

        try:
            response = await self._call_analyzer(prompt)
            changes = self._parse_changes(response)
            return changes[:3]  # 每轮最多 3 个修改
        except Exception as e:
            print(f"  Analyzer error: {e}")
            return self._fallback_analysis(last_round)

    async def _call_analyzer(self, prompt: str) -> str:
        try:
            from anthropic import AsyncAnthropic
            client = self.analyzer_client or AsyncAnthropic()
            resp = await client.messages.create(
                model=self.analyzer_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except ImportError:
            pass

        try:
            from openai import AsyncOpenAI
            client = self.analyzer_client or AsyncOpenAI()
            resp = await client.chat.completions.create(
                model=self.analyzer_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except ImportError:
            pass

        raise RuntimeError("No LLM client available (install anthropic or openai)")

    def _parse_changes(self, response: str) -> list[dict]:
        # 提取 JSON 块
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # 尝试直接解析
            text = response

        # 找到第一个 [ 和最后一个 ]
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            return []

        try:
            changes = json.loads(text[start:end + 1])
            # 验证格式
            valid = []
            for c in changes:
                if all(k in c for k in ("tool", "field", "new")):
                    valid.append(c)
            return valid
        except json.JSONDecodeError:
            return []

    def _fallback_analysis(self, last_round: RoundResult) -> list[dict]:
        """无 LLM 时的规则化回退分析"""
        changes = []

        # 统计失败最多的 tool
        tool_failures = {}
        for f in last_round.failed_cases:
            for tc in f.get("tool_calls", []):
                tool_name = tc.get("tool", "")
                tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1

        if not tool_failures:
            return []

        worst_tool = max(tool_failures, key=tool_failures.get)

        # 找到对应的 tool spec
        tool_spec = next((t for t in self.current_tools if t.name == worst_tool), None)
        if not tool_spec:
            return []

        # 规则化建议：补充杯型映射
        if "size" in json.dumps(tool_spec.input_schema) and "tall" not in tool_spec.description:
            changes.append({
                "tool": worst_tool,
                "field": "inputSchema.properties.size.description" if "size" in tool_spec.input_schema.get("properties", {}) else "description",
                "old": "",
                "new": "杯型：tall(中杯12oz) | grande(大杯16oz) | venti(超大杯20oz)。注意星巴克中杯=tall",
                "reason": "Fallback: 补充星巴克杯型映射",
            })

        return changes

    def _apply_changes(self, changes: list[dict]):
        for change in changes:
            tool = next((t for t in self.current_tools if t.name == change["tool"]), None)
            if not tool:
                continue

            field_path = change["field"]
            new_value = change["new"]

            if field_path == "description":
                tool.description = new_value
            elif field_path.startswith("inputSchema.properties."):
                # e.g. inputSchema.properties.size.description
                parts = field_path.split(".")
                if len(parts) >= 4:
                    param_name = parts[2]
                    sub_field = parts[3]
                    props = tool.input_schema.get("properties", {})
                    if param_name in props:
                        props[param_name][sub_field] = new_value


def _print_round(result: RoundResult, is_baseline=False, improvement=None):
    if is_baseline:
        print(f"\n  Round 0 (baseline): {result.pass_rate:.1%} "
              f"({result.passed_cases}/{result.total_cases})")
        return

    status = "KEEP" if result.kept else "REVERT"
    imp_str = f" ({improvement:+.1%})" if improvement is not None else ""
    change_str = ", ".join(c.get("tool", "?") + "." + c.get("field", "?")[:20]
                          for c in result.changes_made[:3])

    print(f"  Round {result.round_num}: {result.pass_rate:.1%}{imp_str} "
          f"→ [{status}] changes: {change_str or 'none'}")


def save_optimization_log(log: OptimizationLog, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(log.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def save_optimized_tools(tools: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(tools, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
