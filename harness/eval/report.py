"""
Eval Report 生成器

从 EvalResult 列表生成结构化报告（YAML + 终端输出）。
核心产出不是 Pass/Fail，是 Tool Description 改进建议。
"""

from __future__ import annotations

import yaml
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .harness import EvalResult


def generate_report(
    results: list[EvalResult],
    mcp_version: str = "0.1.0",
    include_fingerprint: bool = False,
) -> dict:
    models = sorted(set(r.model for r in results))
    layers = sorted(set(r.layer for r in results))

    # Pass rate by model × criticality
    pass_rates = {}
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        passed = sum(1 for r in model_results if r.passed)
        pass_rates[model] = round(passed / total * 100, 1) if total else 0

    # Pass rate by model × layer
    by_layer = {}
    for layer in layers:
        by_layer[layer] = {}
        for model in models:
            lr = [r for r in results if r.model == model and r.layer == layer]
            total = len(lr)
            passed = sum(1 for r in lr if r.passed)
            by_layer[layer][model] = round(passed / total * 100, 1) if total else 0

    # Failures
    failures = []
    for r in results:
        if not r.passed:
            failed_checks = [c for c in r.checks if not c["passed"]]
            failures.append({
                "case_id": r.case_id,
                "model": r.model,
                "layer": r.layer,
                "failed_checks": failed_checks,
                "error": r.error,
                "tool_calls": r.tool_calls,
            })

    # Top failure patterns
    failure_patterns = defaultdict(int)
    for f in failures:
        for check in f.get("failed_checks", []):
            pattern = check["name"]
            failure_patterns[pattern] += 1
    top_patterns = sorted(failure_patterns.items(), key=lambda x: -x[1])[:10]

    # Gate decision
    gate_decision = "PASS"
    gate_reason = ""
    for model in models:
        rate = pass_rates.get(model, 0)
        if rate < 95:
            gate_decision = "BLOCK"
            gate_reason = f"{model} 通过率 {rate}% < 95%"
            break

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "mcp_version": mcp_version,
            "total_cases": len(results),
            "models_tested": models,
        },
        "pass_rate": pass_rates,
        "by_layer": by_layer,
        "failures": failures[:20],
        "top_failure_patterns": [{"pattern": p, "count": c} for p, c in top_patterns],
        "gate_decision": {
            "result": gate_decision,
            "reason": gate_reason,
        },
    }

    # Fingerprint matrix — multi-dimensional capability profile
    if include_fingerprint:
        from ..fingerprint import FingerprintMatrix

        matrix = FingerprintMatrix()
        for r in results:
            judge_scores = r.fingerprint.get("judge_scores") if r.fingerprint else None
            matrix.add_result(r.model, r.layer, r.passed, judge_scores)
        report["fingerprint_matrix"] = matrix.to_dict()

    return report


def save_report(report: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def print_report(report: dict):
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        _print_plain(report)
        return

    console.print(f"\n[bold]MCP Eval Report[/bold]")
    console.print(f"Generated: {report['meta']['generated_at']}")
    console.print(f"Cases: {report['meta']['total_cases']}  Models: {', '.join(report['meta']['models_tested'])}")

    # Pass rate table
    table = Table(title="Pass Rate")
    table.add_column("Model")
    table.add_column("Overall", justify="right")
    for layer in report.get("by_layer", {}):
        table.add_column(layer, justify="right")

    for model in report["meta"]["models_tested"]:
        overall = f"{report['pass_rate'].get(model, 0)}%"
        layer_rates = [
            f"{report['by_layer'].get(layer, {}).get(model, 0)}%"
            for layer in report.get("by_layer", {})
        ]
        color = "green" if report["pass_rate"].get(model, 0) >= 95 else "red"
        table.add_row(f"[{color}]{model}[/{color}]", f"[{color}]{overall}[/{color}]", *layer_rates)

    console.print(table)

    # Gate decision
    gate = report["gate_decision"]
    if gate["result"] == "PASS":
        console.print(f"\n[green bold]GATE: PASS[/green bold]")
    else:
        console.print(f"\n[red bold]GATE: BLOCK — {gate['reason']}[/red bold]")

    # Top failures
    if report.get("top_failure_patterns"):
        console.print("\n[bold]Top Failure Patterns:[/bold]")
        for item in report["top_failure_patterns"][:5]:
            console.print(f"  {item['count']}x  {item['pattern']}")

    # Failed cases
    failures = report.get("failures", [])
    if failures:
        console.print(f"\n[bold]Failed Cases ({len(failures)}):[/bold]")
        for f in failures[:10]:
            checks_str = ", ".join(c["name"] for c in f.get("failed_checks", []))
            console.print(f"  [red]FAIL[/red] {f['case_id']} ({f['model']}) — {checks_str or f.get('error', '')}")

    # Fingerprint matrix
    fp_matrix = report.get("fingerprint_matrix")
    if fp_matrix:
        console.print(f"\n[bold]Capability Fingerprint Matrix:[/bold]")
        # Collect all dimensions
        all_dims = set()
        for fp in fp_matrix.values():
            all_dims.update(fp.get("dimensions", {}).keys())
        all_dims = sorted(all_dims)

        fp_table = Table(title="Fingerprint")
        fp_table.add_column("Model")
        fp_table.add_column("Overall", justify="right")
        for dim in all_dims:
            fp_table.add_column(dim[:16], justify="right")

        for model_name, fp in sorted(fp_matrix.items()):
            overall = f"{fp.get('overall', 0):.1%}"
            dim_values = []
            for dim in all_dims:
                d = fp.get("dimensions", {}).get(dim, {})
                rate = d.get("pass_rate", 0)
                dim_values.append(f"{rate:.0%}")
            fp_table.add_row(model_name, overall, *dim_values)

        console.print(fp_table)


def _print_plain(report: dict):
    print(f"\nMCP Eval Report")
    print(f"Cases: {report['meta']['total_cases']}")
    print(f"Models: {', '.join(report['meta']['models_tested'])}")
    print(f"\nPass Rates:")
    for model, rate in report["pass_rate"].items():
        print(f"  {model}: {rate}%")
    gate = report["gate_decision"]
    print(f"\nGate: {gate['result']} {gate.get('reason', '')}")
