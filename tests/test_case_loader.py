"""
Eval Case 加载器测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.eval.case_loader import load_cases_from_dir, load_case_file


CASES_DIR = Path(__file__).parent.parent / "evals" / "cases"


def test_load_tool_selection_cases():
    cases = load_cases_from_dir(CASES_DIR / "tool-selection")
    assert len(cases) > 0
    for c in cases:
        assert c.layer == "tool_selection"
        assert c.user_instruction


def test_load_param_mapping_cases():
    cases = load_cases_from_dir(CASES_DIR / "param-mapping")
    assert len(cases) > 0
    for c in cases:
        assert c.layer == "param_mapping"


def test_load_multi_turn_cases():
    cases = load_cases_from_dir(CASES_DIR / "multi-turn")
    assert len(cases) > 0
    for c in cases:
        assert c.layer == "multi_turn"
        assert len(c.dialogue_turns) > 0


def test_load_safety_cases():
    cases = load_cases_from_dir(CASES_DIR / "safety")
    assert len(cases) > 0


def test_load_all_cases():
    cases = load_cases_from_dir(CASES_DIR)
    assert len(cases) >= 20  # 至少 20 个 case


def test_filter_by_criticality():
    all_cases = load_cases_from_dir(CASES_DIR)
    p0_cases = load_cases_from_dir(CASES_DIR, criticality="P0")
    assert len(p0_cases) <= len(all_cases)
    for c in p0_cases:
        assert c.criticality == "P0"


def test_filter_by_layer():
    cases = load_cases_from_dir(CASES_DIR, layer="safety")
    for c in cases:
        assert c.layer == "safety"
