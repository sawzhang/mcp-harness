"""
MCP Eval Harness — 多模型并行评估引擎

核心设计：不直接调用 MCP HTTP，而是让 Agent 自主决策 Tool 调用。
测的是 Tool Description × Model 的组合效果。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from .adapters.base import AgentAdapter, DialogueResult, ORDERING_AGENT_SYSTEM_PROMPT
from .case_loader import EvalCase


@dataclass
class EvalResult:
    case_id: str
    model: str
    layer: str
    passed: bool = False
    tool_calls: list[dict] = field(default_factory=list)
    final_response: str = ""
    latency_ms: float = 0
    token_usage: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)  # [{name, passed, detail}]
    error: str = ""


@dataclass
class ModelEntry:
    adapter: AgentAdapter
    tier: int  # 1=every PR, 2=every sprint, 3=monthly


class MCPEvalHarness:

    def __init__(self, mcp_tools: list[dict] | None = None):
        self.mcp_tools = mcp_tools or []
        self.models: dict[str, ModelEntry] = {}

    def register_model(self, name: str, adapter: AgentAdapter, tier: int = 1):
        self.models[name] = ModelEntry(adapter=adapter, tier=tier)

    def load_tools_from_spec(self, spec_path: str):
        import yaml
        from pathlib import Path
        with open(spec_path, encoding="utf-8") as f:
            self.mcp_tools = yaml.safe_load(f)

    async def run_suite(
        self,
        cases: list[EvalCase],
        tier_filter: int = 1,
        system_prompt: str = ORDERING_AGENT_SYSTEM_PROMPT,
        concurrency: int = 5,
    ) -> list[EvalResult]:
        models = {k: v for k, v in self.models.items() if v.tier <= tier_filter}
        if not models:
            raise ValueError("No models registered for the specified tier")

        semaphore = asyncio.Semaphore(concurrency)
        tasks = []

        for model_name, entry in models.items():
            for case in cases:
                tasks.append(
                    self._eval_with_semaphore(
                        semaphore, model_name, entry.adapter, case, system_prompt
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        eval_results = []
        for r in results:
            if isinstance(r, Exception):
                eval_results.append(EvalResult(
                    case_id="unknown", model="unknown", layer="unknown",
                    passed=False, error=str(r),
                ))
            else:
                eval_results.append(r)

        return eval_results

    async def _eval_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        model_name: str,
        adapter: AgentAdapter,
        case: EvalCase,
        system_prompt: str,
    ) -> EvalResult:
        async with semaphore:
            return await self._eval_one(model_name, adapter, case, system_prompt)

    async def _eval_one(
        self,
        model_name: str,
        adapter: AgentAdapter,
        case: EvalCase,
        system_prompt: str,
    ) -> EvalResult:

        if case.layer == "multi_turn" and case.dialogue_turns:
            return await self._eval_multi_turn(model_name, adapter, case, system_prompt)

        try:
            dialogue = await adapter.run_dialogue(
                system_prompt=system_prompt,
                user_message=case.user_instruction,
                mcp_tools=self.mcp_tools,
                context=case.order_context or None,
                max_turns=5,
                timeout=30,
            )
        except Exception as e:
            return EvalResult(
                case_id=case.id, model=model_name, layer=case.layer,
                passed=False, error=str(e),
            )

        checks = self._run_checks(case, dialogue)
        passed = all(c["passed"] for c in checks)

        return EvalResult(
            case_id=case.id,
            model=model_name,
            layer=case.layer,
            passed=passed,
            tool_calls=[{"tool": tc.tool, "args": tc.arguments} for tc in dialogue.tool_calls],
            final_response=dialogue.final_text,
            latency_ms=dialogue.total_latency_ms,
            token_usage=dialogue.token_usage,
            checks=checks,
        )

    async def _eval_multi_turn(
        self,
        model_name: str,
        adapter: AgentAdapter,
        case: EvalCase,
        system_prompt: str,
    ) -> EvalResult:
        turns_text = [t.get("user", t.get("input", "")) for t in case.dialogue_turns]
        try:
            results = await adapter.run_multi_turn(
                system_prompt=system_prompt,
                turns=turns_text,
                mcp_tools=self.mcp_tools,
                max_turns_per_message=3,
                timeout=30,
            )
        except Exception as e:
            return EvalResult(
                case_id=case.id, model=model_name, layer=case.layer,
                passed=False, error=str(e),
            )

        checks = []
        all_tool_calls = []

        for i, (turn_spec, result) in enumerate(zip(case.dialogue_turns, results)):
            all_tool_calls.extend(
                [{"tool": tc.tool, "args": tc.arguments} for tc in result.tool_calls]
            )

            expected_tool = turn_spec.get("expected_tool", "")
            if expected_tool:
                has_it = result.has_tool(expected_tool)
                checks.append({
                    "name": f"turn_{i+1}_tool_{expected_tool}",
                    "passed": has_it,
                    "detail": f"Expected {expected_tool}, got {[tc.tool for tc in result.tool_calls]}",
                })

            expected_params = turn_spec.get("expected_params", {})
            if expected_params and expected_tool:
                tc = result.get_tool_call(expected_tool)
                if tc:
                    for key, val in expected_params.items():
                        actual = tc.arguments.get(key)
                        match = _param_match(val, actual)
                        checks.append({
                            "name": f"turn_{i+1}_param_{key}",
                            "passed": match,
                            "detail": f"Expected {key}={val}, got {actual}",
                        })

        total_latency = sum(r.total_latency_ms for r in results)
        passed = all(c["passed"] for c in checks) if checks else True

        return EvalResult(
            case_id=case.id,
            model=model_name,
            layer=case.layer,
            passed=passed,
            tool_calls=all_tool_calls,
            latency_ms=total_latency,
            checks=checks,
        )

    def _run_checks(self, case: EvalCase, dialogue: DialogueResult) -> list[dict]:
        checks = []

        # Check: expected tool called
        if case.expected_tool:
            has_tool = dialogue.has_tool(case.expected_tool)
            checks.append({
                "name": "tool_selection",
                "passed": has_tool,
                "detail": f"Expected '{case.expected_tool}', got {[tc.tool for tc in dialogue.tool_calls]}",
            })

        # Check: unwanted tool NOT called
        if case.expected_not_tool:
            has_bad = dialogue.has_tool(case.expected_not_tool)
            checks.append({
                "name": "tool_exclusion",
                "passed": not has_bad,
                "detail": f"'{case.expected_not_tool}' should NOT be called",
            })

        # Check: parameter match (supports partial matching for nested structures)
        if case.expected_params and case.expected_tool:
            tc = dialogue.get_tool_call(case.expected_tool)
            if tc:
                for key, expected_val in case.expected_params.items():
                    actual = tc.arguments.get(key)
                    match = _param_match(expected_val, actual)
                    checks.append({
                        "name": f"param_{key}",
                        "passed": match,
                        "detail": f"Expected {key}={expected_val}, got {actual}",
                    })
            else:
                checks.append({
                    "name": "param_check_skipped",
                    "passed": False,
                    "detail": f"Tool '{case.expected_tool}' not called, cannot check params",
                })

        return checks


def _param_match(expected, actual) -> bool:
    """部分匹配：expected 中指定的字段必须出现在 actual 中，actual 可以有额外字段。"""
    if expected == actual:
        return True
    if expected is None:
        return actual is None
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            k in actual and _param_match(v, actual[k])
            for k, v in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_param_match(e, a) for e, a in zip(expected, actual))
    return expected == actual
