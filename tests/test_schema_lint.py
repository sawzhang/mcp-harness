"""
Schema Linter 测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.lint.mcp_schema_lint import (
    extract_tools,
    check_c1_description,
    check_c2_minimal_params,
    check_c3_flat_params,
    check_c4_param_source,
    check_c5_naming,
    check_c9_write_safety,
    lint_file,
    ToolDef,
    Severity,
)


# --- Fixtures ---

def _make_tool(name="test_tool", docstring="", params=None, is_write=False):
    return ToolDef(
        name=name,
        docstring=docstring,
        params=params or [],
        return_type="str",
        lineno=1,
        is_write_op=is_write,
    )


# --- C1: Description ---

def test_c1_empty_description():
    tool = _make_tool(docstring="")
    issues = check_c1_description(tool)
    assert any(i.severity == Severity.ERROR for i in issues)


def test_c1_single_sentence():
    tool = _make_tool(docstring="查询门店信息")
    issues = check_c1_description(tool)
    assert any(i.rule == "C1" and i.severity == Severity.WARNING for i in issues)


def test_c1_good_description():
    tool = _make_tool(docstring="查询门店信息。当用户说'附近有什么店'时使用此工具。")
    issues = check_c1_description(tool)
    errors = [i for i in issues if i.severity == Severity.ERROR]
    assert len(errors) == 0


def test_c1_trigger_phrase():
    tool = _make_tool(docstring="查询门店。返回门店列表。")
    issues = check_c1_description(tool)
    infos = [i for i in issues if i.severity == Severity.INFO]
    assert any("触发短语" in i.message for i in infos)


# --- C2: Minimal Params ---

def test_c2_inferrable_param():
    tool = _make_tool(params=[{"name": "user_id", "type": "str"}])
    issues = check_c2_minimal_params(tool)
    assert len(issues) == 1
    assert issues[0].rule == "C2"


def test_c2_normal_param():
    tool = _make_tool(params=[{"name": "store_id", "type": "str"}])
    issues = check_c2_minimal_params(tool)
    assert len(issues) == 0


# --- C3: Flat Params ---

def test_c3_dict_param():
    tool = _make_tool(params=[{"name": "location", "type": "dict"}])
    issues = check_c3_flat_params(tool)
    assert any(i.rule == "C3" for i in issues)


def test_c3_nested_list():
    tool = _make_tool(params=[{"name": "items", "type": "list[dict]"}])
    issues = check_c3_flat_params(tool)
    assert any(i.rule == "C3" for i in issues)


def test_c3_string_param():
    tool = _make_tool(params=[{"name": "city", "type": "str"}])
    issues = check_c3_flat_params(tool)
    assert len(issues) == 0


# --- C4: Param Source ---

def test_c4_id_without_source():
    tool = _make_tool(
        docstring="查询门店菜单",
        params=[{"name": "store_id", "type": "str"}],
    )
    issues = check_c4_param_source(tool)
    assert any(i.rule == "C4" for i in issues)


def test_c4_id_with_source():
    tool = _make_tool(
        docstring="查询门店菜单。store_id 从 nearby_stores 返回结果中获取。",
        params=[{"name": "store_id", "type": "str"}],
    )
    issues = check_c4_param_source(tool)
    assert len(issues) == 0


# --- C5: Naming ---

def test_c5_mixed_naming():
    tool = _make_tool(name="get-store_info")
    issues = check_c5_naming(tool, [])
    assert any(i.rule == "C5" and i.severity == Severity.ERROR for i in issues)


def test_c5_consistent_naming():
    tool = _make_tool(name="get_store_info")
    issues = check_c5_naming(tool, [])
    errors = [i for i in issues if i.severity == Severity.ERROR]
    assert len(errors) == 0


# --- C9: Write Safety ---

def test_c9_write_missing_idempotency():
    tool = _make_tool(
        name="create_order",
        params=[{"name": "store_id", "type": "str"}],
        is_write=True,
    )
    issues = check_c9_write_safety(tool)
    assert any(i.rule == "C9" and "idempotency_key" in i.message for i in issues)


def test_c9_write_missing_confirmation():
    tool = _make_tool(
        name="create_order",
        params=[
            {"name": "store_id", "type": "str"},
            {"name": "idempotency_key", "type": "str"},
        ],
        is_write=True,
    )
    issues = check_c9_write_safety(tool)
    assert any(i.rule == "C9" and "confirmation_token" in i.message for i in issues)


def test_c9_write_complete():
    tool = _make_tool(
        name="create_order",
        params=[
            {"name": "store_id", "type": "str"},
            {"name": "idempotency_key", "type": "str"},
            {"name": "confirmation_token", "type": "str"},
        ],
        is_write=True,
    )
    issues = check_c9_write_safety(tool)
    assert len(issues) == 0


def test_c9_read_op_skipped():
    tool = _make_tool(name="get_menu", is_write=False)
    issues = check_c9_write_safety(tool)
    assert len(issues) == 0


# --- Integration: lint real file ---

def test_lint_coffee_mcp():
    coffee_server = Path(__file__).parent.parent.parent / "coffee-mcp" / "src" / "coffee_mcp" / "toc_server.py"
    if not coffee_server.exists():
        return  # Skip if coffee-mcp not available

    report = lint_file(str(coffee_server))
    assert len(report.tools) > 0
    # 打印报告供人工审查
    from harness.lint.mcp_schema_lint import format_report
    print(format_report(report))
