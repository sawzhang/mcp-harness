"""
LLM-as-Judge — 语义评估模块

用 LLM 评估 Agent 的工具调用行为，弥补纯规则匹配无法覆盖的语义判断：
1. 4 维度 Rubric 评估（商品识别 / 工具效率 / 修改理解 / 订单等价）
2. expected_behavior 断言判定（当前仅靠人工描述、无规则可匹配的 case）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeScore:
    dimension: str  # "product_recognition" | "tool_efficiency" | "modification_understanding" | "order_equivalence"
    score: float  # 0.0 - 1.0
    explanation: str = ""


@dataclass
class JudgeResult:
    scores: list[JudgeScore] = field(default_factory=list)
    behavior_passed: bool = True
    behavior_explanation: str = ""
    raw_response: str = ""

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def score_dict(self) -> dict[str, float]:
        return {s.dimension: s.score for s in self.scores}


JUDGE_RUBRIC_PROMPT = """你是一个咖啡订单 AI 评估裁判。请根据以下标准评估 Agent 的表现。

## 评估维度（每项 0-1 分）

1. **商品识别** (product_recognition): Agent 是否正确识别了用户想要的饮品名称、温度、杯型？
2. **工具效率** (tool_efficiency): Agent 是否用最少的工具调用完成了任务？有无冗余调用？
3. **修改理解** (modification_understanding): 用户修改意图时，Agent 是否正确理解并保留了未变更的属性？
4. **订单等价** (order_equivalence): Agent 最终生成的订单参数是否等价于用户的真实意图？

## 用户指令
{user_instruction}

## Agent 工具调用序列
{tool_calls_json}

## Agent 最终回复
{final_response}

## 期望行为
{expected_behavior}

## 期望参数（如有）
{expected_params_json}

请严格按以下 JSON 格式输出评分：

```json
{{
  "scores": [
    {{"dimension": "product_recognition", "score": 0.0, "explanation": "..."}},
    {{"dimension": "tool_efficiency", "score": 0.0, "explanation": "..."}},
    {{"dimension": "modification_understanding", "score": 0.0, "explanation": "..."}},
    {{"dimension": "order_equivalence", "score": 0.0, "explanation": "..."}}
  ],
  "behavior_passed": true,
  "behavior_explanation": "..."
}}
```"""


BEHAVIOR_JUDGE_PROMPT = """你是一个行为断言裁判。判断以下 Agent 的实际行为是否满足期望行为描述。

## 期望行为
{expected_behavior}

## Agent 实际工具调用
{tool_calls_json}

## Agent 最终回复
{final_response}

请回答：Agent 的行为是否满足期望？输出严格 JSON：

```json
{{"passed": true, "explanation": "..."}}
```"""


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling ```json blocks."""
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


class LLMJudge:
    """
    Uses an LLM to evaluate Agent responses on semantic dimensions
    that cannot be checked structurally.
    """

    def __init__(self, client=None, model: str = "claude-sonnet-4-20250514"):
        self.client = client
        self.model = model

    async def evaluate_rubric(
        self,
        user_instruction: str,
        tool_calls: list[dict],
        final_response: str,
        expected_behavior: str = "",
        expected_params: dict | None = None,
    ) -> JudgeResult:
        """Full 4-dimension rubric evaluation."""
        prompt = JUDGE_RUBRIC_PROMPT.format(
            user_instruction=user_instruction,
            tool_calls_json=json.dumps(tool_calls, ensure_ascii=False, indent=2),
            final_response=final_response or "(无回复)",
            expected_behavior=expected_behavior or "(未指定)",
            expected_params_json=json.dumps(
                expected_params, ensure_ascii=False, indent=2
            )
            if expected_params
            else "(未指定)",
        )

        raw = await self._call_llm(prompt)
        parsed = _parse_json_from_response(raw)
        if not parsed:
            return JudgeResult(raw_response=raw)

        scores = []
        for s in parsed.get("scores", []):
            scores.append(
                JudgeScore(
                    dimension=s.get("dimension", ""),
                    score=float(s.get("score", 0)),
                    explanation=s.get("explanation", ""),
                )
            )

        return JudgeResult(
            scores=scores,
            behavior_passed=parsed.get("behavior_passed", True),
            behavior_explanation=parsed.get("behavior_explanation", ""),
            raw_response=raw,
        )

    async def evaluate_behavior(
        self,
        expected_behavior: str,
        tool_calls: list[dict],
        final_response: str,
    ) -> tuple[bool, str]:
        """Evaluate a single expected_behavior assertion. Returns (passed, explanation)."""
        prompt = BEHAVIOR_JUDGE_PROMPT.format(
            expected_behavior=expected_behavior,
            tool_calls_json=json.dumps(tool_calls, ensure_ascii=False, indent=2),
            final_response=final_response or "(无回复)",
        )

        raw = await self._call_llm(prompt)
        parsed = _parse_json_from_response(raw)
        if not parsed:
            return False, f"Judge response parse failed: {raw[:200]}"

        return parsed.get("passed", False), parsed.get("explanation", "")

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for judging. Supports Anthropic and OpenAI clients."""
        if self.client is None:
            return self._fallback(prompt)

        # Try Anthropic
        if hasattr(self.client, "messages"):
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        # Try OpenAI-compatible
        if hasattr(self.client, "chat"):
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            return resp.choices[0].message.content

        return self._fallback(prompt)

    def _fallback(self, prompt: str) -> str:
        """No LLM available — return a conservative default."""
        return json.dumps(
            {
                "scores": [
                    {"dimension": "product_recognition", "score": 0.5, "explanation": "No judge LLM available"},
                    {"dimension": "tool_efficiency", "score": 0.5, "explanation": "No judge LLM available"},
                    {"dimension": "modification_understanding", "score": 0.5, "explanation": "No judge LLM available"},
                    {"dimension": "order_equivalence", "score": 0.5, "explanation": "No judge LLM available"},
                ],
                "behavior_passed": False,
                "behavior_explanation": "No judge LLM configured, cannot evaluate behavior assertion",
                "passed": False,
                "explanation": "No judge LLM configured",
            }
        )
