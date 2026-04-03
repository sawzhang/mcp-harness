"""
从 coffee-mcp MCP Server 源码提取完整 Tool 定义。

两种模式：
  1. AST 模式：解析 Python 源码提取 @mcp.tool() 函数签名和 docstring
  2. MCP 模式：启动 MCP Server，通过 tools/list 协议获取（需要 mcp 依赖）

用法：
    # AST 提取（推荐，无需启动 server）
    python scripts/extract_tools.py --source ../coffee-mcp/src/coffee_mcp/toc_server.py

    # 保存为 YAML
    python scripts/extract_tools.py --source ../coffee-mcp/src/coffee_mcp/toc_server.py \
        --output evals/tool_specs.yaml
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml


def _parse_type_to_json_schema(annotation: str) -> dict:
    """Python 类型注解 → JSON Schema（简化版）"""
    ann = annotation.strip()
    if not ann:
        return {"type": "string"}

    # str | None
    ann_clean = ann.replace(" | None", "").replace("Optional[", "").rstrip("]")

    type_map = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
    }

    if ann_clean in type_map:
        return type_map[ann_clean]

    if ann_clean.startswith("list["):
        inner = ann_clean[5:-1]
        return {"type": "array", "items": _parse_type_to_json_schema(inner)}

    if ann_clean.startswith("dict"):
        return {"type": "object"}

    return {"type": "string"}


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """从 docstring 的 Args: 段提取参数描述"""
    param_descs = {}
    in_args = False
    current_param = None

    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            # 新参数行：  param_name: description
            match = re.match(r'^(\w+):\s*(.+)', stripped)
            if match:
                current_param = match.group(1)
                param_descs[current_param] = match.group(2).strip()
            elif stripped and current_param:
                # 续行
                param_descs[current_param] += " " + stripped
            elif not stripped:
                # 空行结束 Args 段
                if current_param:
                    in_args = False

    return param_descs


def extract_tools_from_source(source_path: str) -> list[dict]:
    """从 Python 源码 AST 提取 @mcp.tool() 定义"""
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    tools = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        is_tool = any(
            "tool" in ast.dump(dec).lower() and "mcp" in ast.dump(dec).lower()
            for dec in node.decorator_list
        )
        if not is_tool:
            continue

        name = node.name
        docstring = ast.get_docstring(node) or ""
        first_line = docstring.split("\n")[0].strip() if docstring else ""

        # 提取参数
        param_descs = _parse_docstring_params(docstring)
        properties = {}
        required = []

        for arg in node.args.args:
            if arg.arg == "self":
                continue

            pname = arg.arg
            annotation = ast.unparse(arg.annotation) if arg.annotation else "str"
            schema = _parse_type_to_json_schema(annotation)

            # 添加描述
            if pname in param_descs:
                schema["description"] = param_descs[pname]

            properties[pname] = schema

        # 判断 required：有默认值的不 required
        defaults_count = len(node.args.defaults)
        total_args = [a for a in node.args.args if a.arg != "self"]
        non_default_count = len(total_args) - defaults_count

        for i, arg in enumerate(total_args):
            if i < non_default_count:
                # 检查是否是 Optional (str | None)
                if arg.annotation:
                    ann = ast.unparse(arg.annotation)
                    if "None" not in ann:
                        required.append(arg.arg)

        input_schema = {"type": "object", "properties": properties}
        if required:
            input_schema["required"] = required

        tool = {
            "name": name,
            "description": docstring.strip(),
            "inputSchema": input_schema,
        }
        tools.append(tool)

    return tools


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MCP Tool specs from source")
    parser.add_argument("--source", required=True, help="Path to MCP server Python file")
    parser.add_argument("--output", help="Output YAML path (default: stdout)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of YAML")
    args = parser.parse_args()

    tools = extract_tools_from_source(args.source)
    print(f"Extracted {len(tools)} tools from {args.source}", file=sys.stderr)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            if args.json:
                json.dump(tools, f, ensure_ascii=False, indent=2)
            else:
                yaml.dump(tools, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        if args.json:
            print(json.dumps(tools, ensure_ascii=False, indent=2))
        else:
            print(yaml.dump(tools, allow_unicode=True, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
