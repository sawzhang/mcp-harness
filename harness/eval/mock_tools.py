"""
Programmable Mock Tool Result System

为 Eval 模式下的 Adapter 提供可配置的 Tool 响应。支持：
1. 默认（全部成功）— 向后兼容现有行为
2. 静态错误注入（某个 tool 总返回 404）
3. 序列注入（第1次 408 超时，第2次 200 成功 — 测试重试）
4. 动态注入（根据参数决定响应）

用法：
    provider = ToolResultProvider()
    provider.set_result("calculate_price", ERROR_409_CONFLICT)
    result = provider.get_result("calculate_price", {"store_id": "ST_001"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MockToolResult:
    """A single mock response for a tool call."""

    status_code: int = 200
    body: dict = field(default_factory=lambda: {"status": "success", "mock": True})
    error_message: str = ""

    def to_response(self) -> dict:
        if self.status_code >= 400:
            return {
                "error": True,
                "status_code": self.status_code,
                "message": self.error_message,
            }
        return {**self.body, "status_code": self.status_code}


# Pre-built error templates matching the design document's 5 error types
ERROR_404_NOT_FOUND = MockToolResult(
    status_code=404, error_message="Resource not found"
)
ERROR_409_CONFLICT = MockToolResult(
    status_code=409, error_message="Conflict: resource state changed"
)
ERROR_401_UNAUTHORIZED = MockToolResult(
    status_code=401, error_message="Authentication required or token expired"
)
ERROR_408_TIMEOUT = MockToolResult(
    status_code=408, error_message="Service timeout, please retry"
)
ERROR_207_PARTIAL = MockToolResult(
    status_code=207,
    body={"status": "partial_success", "succeeded": [], "failed": [], "mock": True},
)


class ToolResultProvider:
    """
    Provides mock tool results. Supports:
    1. Default (all success) — backward compatible
    2. Per-tool static results
    3. Per-tool sequential results (different result per invocation)
    4. Callable for dynamic behavior

    Priority: dynamic > sequence > static > default
    """

    def __init__(self):
        self._static: dict[str, MockToolResult] = {}
        self._sequences: dict[str, list[MockToolResult]] = {}
        self._dynamic: dict[str, Callable] = {}
        self._call_counts: dict[str, int] = {}
        self._default = MockToolResult()

    def set_result(self, tool_name: str, result: MockToolResult):
        """Set a static result for a tool (always returns same response)."""
        self._static[tool_name] = result

    def set_sequence(self, tool_name: str, results: list[MockToolResult]):
        """Set a sequence of results; cycles after exhaustion."""
        self._sequences[tool_name] = results

    def set_dynamic(
        self, tool_name: str, fn: Callable[[str, dict], MockToolResult]
    ):
        """Set a callable that receives (tool_name, arguments) and returns a result."""
        self._dynamic[tool_name] = fn

    def get_result(self, tool_name: str, arguments: dict | None = None) -> dict:
        """Get the next result for a tool call. Called by adapters."""
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        count = self._call_counts[tool_name]

        if tool_name in self._dynamic:
            return self._dynamic[tool_name](tool_name, arguments or {}).to_response()

        if tool_name in self._sequences:
            seq = self._sequences[tool_name]
            idx = (count - 1) % len(seq)
            return seq[idx].to_response()

        if tool_name in self._static:
            return self._static[tool_name].to_response()

        return {**self._default.to_response(), "tool": tool_name}

    def get_call_count(self, tool_name: str) -> int:
        """Get how many times a tool has been called."""
        return self._call_counts.get(tool_name, 0)

    def reset(self):
        """Reset call counts (does not clear configured results)."""
        self._call_counts.clear()
