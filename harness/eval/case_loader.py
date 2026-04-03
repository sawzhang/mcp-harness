"""
Eval Case 加载器

从 YAML 文件加载评估用例，支持按 layer / criticality / tag 过滤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalCase:
    id: str
    layer: str  # tool_selection, param_mapping, multi_turn, safety
    criticality: str  # P0, P1, P2
    tags: list[str] = field(default_factory=list)
    source: str = "manual"  # manual, fid_generated, prod_distilled, adversarial

    # 单轮
    user_instruction: str = ""
    order_context: dict = field(default_factory=dict)

    # 预期
    expected_tool: str = ""
    expected_not_tool: str = ""
    expected_params: dict = field(default_factory=dict)
    expected_params_semantic: str = ""
    expected_behavior: str = ""
    violation: str = ""
    trap: str = ""

    # 多轮
    dialogue_turns: list[dict] = field(default_factory=list)

    # 参数映射专用
    mapping_tests: list[dict] = field(default_factory=list)


def load_case_file(path: Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        return []

    cases_raw = raw.get("cases", [])
    if not cases_raw:
        # 单个 case 文件
        cases_raw = [raw]

    # 全局元数据
    defaults = {
        "layer": raw.get("layer", ""),
        "criticality": raw.get("criticality", "P1"),
        "tags": raw.get("tags", []),
        "source": raw.get("source", "manual"),
    }

    cases = []
    for i, c in enumerate(cases_raw):
        case = EvalCase(
            id=c.get("id", f"{path.stem}-{i:03d}"),
            layer=c.get("layer", defaults["layer"]),
            criticality=c.get("criticality", defaults["criticality"]),
            tags=c.get("tags", defaults["tags"]),
            source=c.get("source", defaults["source"]),
            user_instruction=c.get("user_instruction", c.get("input", "")),
            order_context=c.get("order_context", {}),
            expected_tool=c.get("expected_tool", ""),
            expected_not_tool=c.get("expected_not_tool", ""),
            expected_params=c.get("expected_params", {}),
            expected_params_semantic=c.get("expected_params_semantic", ""),
            expected_behavior=c.get("expected_behavior", ""),
            violation=c.get("violation", ""),
            trap=c.get("trap", ""),
            dialogue_turns=c.get("dialogue", c.get("dialogue_turns", [])),
            mapping_tests=c.get("mapping_tests", c.get("variants", [])),
        )
        cases.append(case)

    return cases


def load_cases_from_dir(
    directory: str | Path,
    layer: str | None = None,
    criticality: str | None = None,
    tags: list[str] | None = None,
) -> list[EvalCase]:
    directory = Path(directory)
    all_cases = []

    for yaml_file in sorted(directory.rglob("*.yaml")):
        cases = load_case_file(yaml_file)
        all_cases.extend(cases)

    for yml_file in sorted(directory.rglob("*.yml")):
        cases = load_case_file(yml_file)
        all_cases.extend(cases)

    if layer:
        all_cases = [c for c in all_cases if c.layer == layer]
    if criticality:
        all_cases = [c for c in all_cases if c.criticality == criticality]
    if tags:
        tag_set = set(tags)
        all_cases = [c for c in all_cases if tag_set & set(c.tags)]

    return all_cases
