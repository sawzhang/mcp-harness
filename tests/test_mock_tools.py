"""
Programmable Mock System + Error Recovery + Judge + Fingerprint 测试
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harness.eval.mock_tools import (
    MockToolResult,
    ToolResultProvider,
    ERROR_404_NOT_FOUND,
    ERROR_409_CONFLICT,
    ERROR_401_UNAUTHORIZED,
    ERROR_408_TIMEOUT,
    ERROR_207_PARTIAL,
)
from harness.eval.adapters.base import AgentAdapter, DialogueResult, ToolCall
from harness.eval.harness import MCPEvalHarness
from harness.eval.case_loader import EvalCase, load_cases_from_dir
from harness.eval.judge import LLMJudge, JudgeResult, JudgeScore, _parse_json_from_response
from harness.eval.report import generate_report
from harness.fingerprint import FingerprintMatrix, ModelFingerprint, DimensionScore


# ===== ToolResultProvider Tests =====

def test_default_provider_returns_success():
    p = ToolResultProvider()
    result = p.get_result("any_tool", {})
    assert result["status_code"] == 200
    assert result["status"] == "success"


def test_static_result_for_tool():
    p = ToolResultProvider()
    p.set_result("search_menu", ERROR_404_NOT_FOUND)
    result = p.get_result("search_menu", {})
    assert result["error"] is True
    assert result["status_code"] == 404


def test_static_does_not_affect_other_tools():
    p = ToolResultProvider()
    p.set_result("search_menu", ERROR_404_NOT_FOUND)
    result = p.get_result("calculate_price", {})
    assert result["status_code"] == 200


def test_sequence_cycles():
    p = ToolResultProvider()
    p.set_sequence("nearby_stores", [
        ERROR_408_TIMEOUT,
        MockToolResult(),  # 200 success
    ])
    r1 = p.get_result("nearby_stores", {})
    r2 = p.get_result("nearby_stores", {})
    r3 = p.get_result("nearby_stores", {})  # cycles back

    assert r1["status_code"] == 408
    assert r2["status_code"] == 200
    assert r3["status_code"] == 408  # back to first


def test_dynamic_provider():
    def dynamic_fn(tool_name, args):
        if args.get("store_id") == "CLOSED":
            return MockToolResult(status_code=404, error_message="Store closed")
        return MockToolResult()

    p = ToolResultProvider()
    p.set_dynamic("browse_menu", dynamic_fn)

    r1 = p.get_result("browse_menu", {"store_id": "CLOSED"})
    assert r1["status_code"] == 404

    r2 = p.get_result("browse_menu", {"store_id": "OPEN"})
    assert r2["status_code"] == 200


def test_error_templates():
    assert ERROR_404_NOT_FOUND.to_response()["status_code"] == 404
    assert ERROR_409_CONFLICT.to_response()["error"] is True
    assert ERROR_401_UNAUTHORIZED.to_response()["status_code"] == 401
    assert ERROR_408_TIMEOUT.to_response()["status_code"] == 408
    assert ERROR_207_PARTIAL.to_response()["status_code"] == 207
    assert ERROR_207_PARTIAL.to_response()["status"] == "partial_success"


def test_call_count():
    p = ToolResultProvider()
    assert p.get_call_count("x") == 0
    p.get_result("x", {})
    p.get_result("x", {})
    assert p.get_call_count("x") == 2


def test_reset_clears_counts():
    p = ToolResultProvider()
    p.get_result("x", {})
    p.reset()
    assert p.get_call_count("x") == 0


# ===== MockAdapter with Provider Integration =====

class MockAdapterWithProvider(AgentAdapter):
    name = "mock-provider"

    def __init__(self, tool_to_call="calculate_price", params=None):
        self._tool = tool_to_call
        self._params = params or {}

    async def run_dialogue(self, system_prompt, user_message, mcp_tools,
                           context=None, max_turns=5, timeout=30,
                           tool_result_provider=None):
        tc = ToolCall(tool=self._tool, arguments=self._params)

        if tool_result_provider:
            mock_result = tool_result_provider.get_result(self._tool, self._params)
        else:
            mock_result = {"status": "success", "mock": True}
        tc.result = mock_result

        was_error = mock_result.get("error", False)

        return DialogueResult(
            turns=[{"role": "assistant", "content": "OK", "tool_calls": [{"name": self._tool}]}],
            tool_calls=[tc],
            final_text="处理完成。" if not was_error else "发生错误。",
            total_latency_ms=100,
            tool_call_timeline=[{
                "turn": 0, "tool": self._tool,
                "arguments": self._params, "result": mock_result,
                "was_error": was_error,
            }],
        )


@pytest.mark.asyncio
async def test_harness_with_error_injection():
    """Error injection case should pass provider to adapter."""
    harness = MCPEvalHarness(mcp_tools=[
        {"name": "calculate_price", "description": "calc", "inputSchema": {"type": "object"}},
    ])
    harness.register_model("mock", MockAdapterWithProvider(), tier=1)

    case = EvalCase(
        id="er-test-001",
        layer="error_recovery",
        criticality="P0",
        user_instruction="来杯拿铁",
        expected_not_tool="create_order",
        error_injection={
            "calculate_price": {"status_code": 409, "message": "Sold out"},
        },
    )

    results = await harness.run_suite([case])
    assert len(results) == 1
    # The mock adapter doesn't change tool based on error, so expected_not_tool check passes
    # (create_order was never called)


@pytest.mark.asyncio
async def test_harness_without_error_injection_backward_compatible():
    """Cases without error_injection should work as before."""
    harness = MCPEvalHarness(mcp_tools=[
        {"name": "calculate_price", "description": "calc", "inputSchema": {"type": "object"}},
    ])
    harness.register_model("mock", MockAdapterWithProvider(), tier=1)

    case = EvalCase(
        id="compat-001",
        layer="tool_selection",
        criticality="P0",
        user_instruction="来杯拿铁",
        expected_tool="calculate_price",
    )

    results = await harness.run_suite([case])
    assert results[0].passed is True


# ===== Case Loader Tests =====

def test_load_error_recovery_cases():
    cases_dir = Path(__file__).parent.parent / "evals" / "cases" / "error-recovery"
    if not cases_dir.exists():
        return
    cases = load_cases_from_dir(cases_dir)
    assert len(cases) >= 1
    er_case = next((c for c in cases if c.id == "ER-SOLD-001"), None)
    assert er_case is not None
    assert er_case.error_injection != {}
    assert "calculate_price" in er_case.error_injection
    assert er_case.error_injection["calculate_price"]["status_code"] == 409


def test_error_injection_default_empty():
    case = EvalCase(id="x", layer="test", criticality="P0")
    assert case.error_injection == {}
    assert case.expected_recovery == ""
    assert case.ground_truth_order == {}
    assert case.eval_mode == ""
    assert case.optimal_steps == 0


# ===== Judge Tests =====

def test_judge_result_average_score():
    result = JudgeResult(scores=[
        JudgeScore(dimension="a", score=0.8),
        JudgeScore(dimension="b", score=0.6),
    ])
    assert abs(result.average_score - 0.7) < 0.01


def test_judge_result_score_dict():
    result = JudgeResult(scores=[
        JudgeScore(dimension="product_recognition", score=0.9),
        JudgeScore(dimension="tool_efficiency", score=0.7),
    ])
    d = result.score_dict
    assert d["product_recognition"] == 0.9
    assert d["tool_efficiency"] == 0.7


def test_parse_json_from_response_with_code_block():
    text = '```json\n{"passed": true, "explanation": "ok"}\n```'
    parsed = _parse_json_from_response(text)
    assert parsed["passed"] is True


def test_parse_json_from_response_plain():
    text = '{"passed": false, "explanation": "not ok"}'
    parsed = _parse_json_from_response(text)
    assert parsed["passed"] is False


def test_parse_json_from_response_invalid():
    parsed = _parse_json_from_response("no json here")
    assert parsed == {}


@pytest.mark.asyncio
async def test_judge_fallback_no_client():
    judge = LLMJudge(client=None)
    passed, explanation = await judge.evaluate_behavior(
        expected_behavior="should inform user",
        tool_calls=[{"tool": "browse_menu", "args": {}}],
        final_response="好的",
    )
    assert passed is False  # Fallback always returns False
    assert "No judge LLM configured" in explanation


# ===== Fingerprint Matrix Tests =====

def test_fingerprint_empty():
    m = FingerprintMatrix()
    assert m.to_dict() == {}


def test_fingerprint_add_single_result():
    m = FingerprintMatrix()
    m.add_result("claude", "tool_selection", True)
    m.add_result("claude", "tool_selection", False)

    fp = m.get_model_fingerprint("claude")
    assert fp is not None
    assert fp.dimensions["tool_selection"].sample_count == 2
    assert fp.dimensions["tool_selection"].pass_rate == 0.5


def test_fingerprint_multiple_models():
    m = FingerprintMatrix()
    m.add_result("claude", "tool_selection", True)
    m.add_result("gpt4o", "tool_selection", False)

    d = m.to_dict()
    assert "claude" in d
    assert "gpt4o" in d


def test_fingerprint_with_judge_scores():
    m = FingerprintMatrix()
    m.add_result("claude", "tool_selection", True,
                 judge_scores={"product_recognition": 0.9, "tool_efficiency": 0.8})

    fp = m.get_model_fingerprint("claude")
    assert "product_recognition" in fp.dimensions
    assert fp.dimensions["product_recognition"].avg_score == 0.9


def test_fingerprint_compute_overall():
    m = FingerprintMatrix()
    m.add_result("claude", "tool_selection", True)
    m.add_result("claude", "param_mapping", True)
    m.add_result("claude", "safety", False)

    m.compute_overall()
    fp = m.get_model_fingerprint("claude")
    # 3 layers: 1.0, 1.0, 0.0 → overall = 2/3
    assert abs(fp.overall_score - 2 / 3) < 0.01


def test_report_with_fingerprint():
    from harness.eval.harness import EvalResult
    results = [
        EvalResult(case_id="t1", model="claude", layer="tool_selection", passed=True),
        EvalResult(case_id="t2", model="claude", layer="param_mapping", passed=False),
        EvalResult(case_id="t3", model="gpt4o", layer="tool_selection", passed=True),
    ]
    report = generate_report(results, include_fingerprint=True)
    assert "fingerprint_matrix" in report
    assert "claude" in report["fingerprint_matrix"]
    assert "gpt4o" in report["fingerprint_matrix"]
