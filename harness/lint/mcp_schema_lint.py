"""
MCP Schema Linter — C1-C10 自动化检查

扫描 MCP Server 源文件中的 @mcp.tool() 定义，逐条审查。
可作为 CI Gate（G1）阻断不合规的 PR。

用法：
    python -m harness.lint.mcp_schema_lint <server.py>
    python -m harness.lint.mcp_schema_lint ../coffee-mcp/src/coffee_mcp/toc_server.py
"""

from __future__ import annotations

import ast
import re
import sys
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LintIssue:
    rule: str
    tool_name: str
    severity: Severity
    message: str
    fix_hint: str = ""


@dataclass
class ToolDef:
    name: str
    docstring: str
    params: list[dict]
    return_type: str
    lineno: int
    is_write_op: bool = False


@dataclass
class LintReport:
    file_path: str
    tools: list[ToolDef] = field(default_factory=list)
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


WRITE_OP_KEYWORDS = {"create", "update", "delete", "remove", "claim", "redeem", "add", "modify"}
INFERRABLE_PARAMS = {"user_id", "member_id", "account_id", "phone", "mobile", "token", "session_id"}


def extract_tools(source: str) -> list[ToolDef]:
    tree = ast.parse(source)
    tools = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        is_tool = False
        for dec in node.decorator_list:
            dec_src = ast.dump(dec)
            if "mcp" in dec_src.lower() and "tool" in dec_src.lower():
                is_tool = True
                break

        if not is_tool:
            continue

        docstring = ast.get_docstring(node) or ""
        params = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            annotation = ""
            if arg.annotation:
                annotation = ast.unparse(arg.annotation)
            params.append({"name": arg.arg, "type": annotation})

        name = node.name
        is_write = any(kw in name.lower() for kw in WRITE_OP_KEYWORDS)

        tools.append(ToolDef(
            name=name,
            docstring=docstring,
            params=params,
            return_type="str",
            lineno=node.lineno,
            is_write_op=is_write,
        ))

    return tools


def check_c1_description(tool: ToolDef) -> list[LintIssue]:
    issues = []
    desc = tool.docstring.strip()

    if not desc:
        issues.append(LintIssue(
            rule="C1", tool_name=tool.name, severity=Severity.ERROR,
            message="缺少 description（docstring 为空）",
            fix_hint="添加三段式描述：做什么 + 什么时候用 + 触发短语",
        ))
        return issues

    sentences = [s.strip() for s in re.split(r'[。\.\n]', desc) if s.strip()]
    if len(sentences) < 2:
        issues.append(LintIssue(
            rule="C1", tool_name=tool.name, severity=Severity.WARNING,
            message=f"Description 只有 {len(sentences)} 段，建议至少2段（做什么+什么时候用）",
            fix_hint="添加使用场景，如'当用户说XXX时使用此工具'",
        ))

    trigger_patterns = ["当用户", "当顾客", "用户说", "用户问", "when user", "use this tool"]
    has_trigger = any(p in desc.lower() for p in trigger_patterns)
    if not has_trigger:
        issues.append(LintIssue(
            rule="C1", tool_name=tool.name, severity=Severity.INFO,
            message="Description 缺少触发短语（可选但推荐）",
            fix_hint="添加自然语言触发提示，如'当用户说\"帮我点一杯\"时触发'",
        ))

    return issues


def check_c2_minimal_params(tool: ToolDef) -> list[LintIssue]:
    issues = []
    for param in tool.params:
        if param["name"] in INFERRABLE_PARAMS:
            issues.append(LintIssue(
                rule="C2", tool_name=tool.name, severity=Severity.WARNING,
                message=f"参数 '{param['name']}' 通常可从 token/上下文推断，确认是否必要",
                fix_hint="ToC 场景下用户身份应从 token 推断，不需要显式传参",
            ))
    return issues


def check_c3_flat_params(tool: ToolDef) -> list[LintIssue]:
    issues = []
    for param in tool.params:
        ptype = param.get("type", "")
        if "dict" in ptype.lower() or "Dict" in ptype:
            issues.append(LintIssue(
                rule="C3", tool_name=tool.name, severity=Severity.WARNING,
                message=f"参数 '{param['name']}' 类型为 dict/object，LLM 构造嵌套 JSON 容易出错",
                fix_hint="考虑扁平化为 string 或拆分为多个参数",
            ))
        if ptype.startswith("list[dict") or ptype.startswith("List[dict"):
            issues.append(LintIssue(
                rule="C3", tool_name=tool.name, severity=Severity.INFO,
                message=f"参数 '{param['name']}' 是嵌套数组，确保内部结构尽量简单（≤5个字段）",
            ))
    return issues


def check_c4_param_source(tool: ToolDef) -> list[LintIssue]:
    issues = []
    id_like_params = [p for p in tool.params if p["name"].endswith("_id") or p["name"].endswith("_code")]

    for param in id_like_params:
        desc = tool.docstring.lower()
        param_mentioned = param["name"] in desc
        has_source_hint = any(kw in desc for kw in ["从", "from", "返回", "获取"])
        if not (param_mentioned and has_source_hint):
            issues.append(LintIssue(
                rule="C4", tool_name=tool.name, severity=Severity.WARNING,
                message=f"参数 '{param['name']}' 像引用型字段，description 未标注值从哪个 tool 获取",
                fix_hint=f"在 docstring 中添加来源说明，如'{param['name']}：从 xxx 返回结果中获取'",
            ))
    return issues


def check_c5_naming(tool: ToolDef, all_tools: list[ToolDef]) -> list[LintIssue]:
    issues = []
    name = tool.name

    if "_" in name and "-" in name:
        issues.append(LintIssue(
            rule="C5", tool_name=name, severity=Severity.ERROR,
            message="命名混用了 snake_case 和 kebab-case",
            fix_hint="统一命名风格",
        ))

    if not re.match(r'^[a-z]', name):
        issues.append(LintIssue(
            rule="C5", tool_name=name, severity=Severity.WARNING,
            message="Tool 名称不以小写字母开头",
        ))

    return issues


def check_c7_error_codes(tool: ToolDef) -> list[LintIssue]:
    issues = []
    if not tool.is_write_op:
        return issues

    desc = tool.docstring.lower()
    has_error_mention = any(kw in desc for kw in ["error", "错误", "失败", "400", "404", "409"])
    if not has_error_mention:
        issues.append(LintIssue(
            rule="C7", tool_name=tool.name, severity=Severity.INFO,
            message="写操作的 description 未提及错误场景",
            fix_hint="在 description 中说明可能的错误情况，帮助 Agent 理解异常处理",
        ))
    return issues


def check_c9_write_safety(tool: ToolDef) -> list[LintIssue]:
    issues = []
    if not tool.is_write_op:
        return issues

    param_names = {p["name"] for p in tool.params}

    if "idempotency_key" not in param_names:
        issues.append(LintIssue(
            rule="C9", tool_name=tool.name, severity=Severity.ERROR,
            message="写操作缺少 idempotency_key 参数",
            fix_hint="所有写操作必须支持幂等，添加 idempotency_key: str 参数",
        ))

    high_risk_ops = {"create_order", "stars_redeem", "submit", "purchase", "pay"}
    if any(kw in tool.name.lower() for kw in high_risk_ops):
        if "confirmation_token" not in param_names:
            issues.append(LintIssue(
                rule="C9", tool_name=tool.name, severity=Severity.ERROR,
                message="高风险写操作缺少 confirmation_token 参数（L3 安全要求）",
                fix_hint="L3 操作必须有前置确认步骤，添加 confirmation_token 参数",
            ))

    return issues


def check_c6_examples(tool: ToolDef) -> list[LintIssue]:
    """C6: 参数是否有 example 值，帮助 LLM 理解期望格式"""
    issues = []
    desc = tool.docstring.lower()
    for param in tool.params:
        pname = param["name"]
        # 检查 docstring 中是否提及该参数的示例
        has_example = any(kw in desc for kw in [
            f"{pname}:", f"如'{pname}", f"例如", "example", "如 ", "格式",
        ])
        if not has_example and pname not in ("self",):
            # 对于 _id / _code 类参数，示例特别重要
            if pname.endswith(("_id", "_code", "_key", "_token")):
                issues.append(LintIssue(
                    rule="C6", tool_name=tool.name, severity=Severity.INFO,
                    message=f"参数 '{pname}' 缺少示例值，LLM 可能不清楚格式",
                    fix_hint=f"在 description 中添加示例，如 '{pname}: \"ST_SH_001\"'",
                ))
    return issues


def check_c8_progressive_disclosure(tool: ToolDef, all_tools: list[ToolDef]) -> list[LintIssue]:
    """C8: List 工具是否有对应的 Detail 工具（渐进式披露）"""
    issues = []
    list_patterns = ["list", "browse", "search", "query", "nearby"]
    detail_patterns = ["detail", "info", "get"]

    is_list_tool = any(p in tool.name.lower() for p in list_patterns)
    if not is_list_tool:
        return issues

    # 检查是否有对应的 detail 工具
    base_name = tool.name.lower()
    for p in list_patterns:
        base_name = base_name.replace(p, "").strip("_")

    has_detail = any(
        any(dp in t.name.lower() for dp in detail_patterns) and
        base_name in t.name.lower().replace("_", "")
        for t in all_tools if t.name != tool.name
    )

    # 如果是列表工具，description 应提到返回摘要/ID
    desc = tool.docstring.lower()
    mentions_summary = any(kw in desc for kw in ["摘要", "id", "summary", "列表", "list"])

    if not mentions_summary:
        issues.append(LintIssue(
            rule="C8", tool_name=tool.name, severity=Severity.INFO,
            message="列表工具的 description 未提及返回摘要 + ID 模式",
            fix_hint="说明返回的是摘要信息，详情需调用对应 detail 工具",
        ))

    return issues


def check_c10_pii(tool: ToolDef) -> list[LintIssue]:
    issues = []
    desc = tool.docstring.lower()
    pii_keywords = ["手机号", "phone", "身份证", "id_card", "地址", "address"]
    if any(kw in desc for kw in pii_keywords):
        mask_keywords = ["脱敏", "mask", "****", "隐藏"]
        if not any(kw in desc for kw in mask_keywords):
            issues.append(LintIssue(
                rule="C10", tool_name=tool.name, severity=Severity.WARNING,
                message="Tool 涉及 PII 字段，description 未提及脱敏处理",
                fix_hint="确认返回值中 PII 字段已脱敏（如 152****6666）",
            ))
    return issues


def lint_file(file_path: str) -> LintReport:
    source = Path(file_path).read_text(encoding="utf-8")
    tools = extract_tools(source)
    report = LintReport(file_path=file_path, tools=tools)

    for tool in tools:
        report.issues.extend(check_c1_description(tool))
        report.issues.extend(check_c2_minimal_params(tool))
        report.issues.extend(check_c3_flat_params(tool))
        report.issues.extend(check_c6_examples(tool))
        report.issues.extend(check_c8_progressive_disclosure(tool, tools))
        report.issues.extend(check_c4_param_source(tool))
        report.issues.extend(check_c5_naming(tool, tools))
        report.issues.extend(check_c7_error_codes(tool))
        report.issues.extend(check_c9_write_safety(tool))
        report.issues.extend(check_c10_pii(tool))

    return report


def format_report(report: LintReport) -> str:
    lines = []
    lines.append(f"# MCP Schema Lint Report")
    lines.append(f"File: {report.file_path}")
    lines.append(f"Tools: {len(report.tools)}")
    lines.append("")

    for tool in report.tools:
        tool_issues = [i for i in report.issues if i.tool_name == tool.name]
        errors = [i for i in tool_issues if i.severity == Severity.ERROR]
        warnings = [i for i in tool_issues if i.severity == Severity.WARNING]
        infos = [i for i in tool_issues if i.severity == Severity.INFO]

        status = "✅" if not errors else "❌"
        lines.append(f"## {status} `{tool.name}` (line {tool.lineno})")
        lines.append(f"  params: {len(tool.params)} | write: {tool.is_write_op}")

        if not tool_issues:
            lines.append("  All checks passed.")
        else:
            for issue in tool_issues:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity.value]
                lines.append(f"  {icon} [{issue.rule}] {issue.message}")
                if issue.fix_hint:
                    lines.append(f"     → {issue.fix_hint}")
        lines.append("")

    # Summary
    total_errors = len(report.errors)
    total_warnings = len(report.warnings)
    lines.append("---")
    lines.append("## Summary")
    lines.append(f"  Tools: {len(report.tools)}")
    lines.append(f"  ❌ Errors: {total_errors}")
    lines.append(f"  ⚠️  Warnings: {total_warnings}")
    lines.append(f"  Gate: {'PASS' if report.passed else 'BLOCK'}")

    if not report.passed:
        lines.append("")
        lines.append("## Top Errors (must fix)")
        for issue in report.errors:
            lines.append(f"  - [{issue.rule}] {issue.tool_name}: {issue.message}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m harness.lint.mcp_schema_lint <server.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    report = lint_file(file_path)
    print(format_report(report))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
