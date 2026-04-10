"""
Claude Adapter — 基准模型

Claude 使用 tool_use 格式，与 OpenAI 的 function calling 不同。
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import AgentAdapter, DialogueResult, ToolCall

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None


def _convert_mcp_to_claude_tools(mcp_tools: list[dict]) -> list[dict]:
    tools = []
    for t in mcp_tools:
        tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {"type": "object", "properties": {}}),
        })
    return tools


class ClaudeAdapter(AgentAdapter):

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        name: str | None = None,
    ):
        if AsyncAnthropic is None:
            raise ImportError("pip install anthropic")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.name = name or model

    async def run_dialogue(
        self,
        system_prompt: str,
        user_message: str,
        mcp_tools: list[dict],
        context: dict | None = None,
        max_turns: int = 5,
        timeout: int = 30,
        tool_result_provider: Any = None,
    ) -> DialogueResult:
        tools = _convert_mcp_to_claude_tools(mcp_tools)

        context_text = ""
        if context:
            context_text = f"\n\n当前上下文：{json.dumps(context, ensure_ascii=False)}"

        messages = [
            {"role": "user", "content": user_message + context_text},
        ]

        all_tool_calls = []
        turns = []
        tool_call_timeline = []
        start = time.monotonic()
        total_tokens = {"input": 0, "output": 0}

        for _ in range(max_turns):
            resp = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                tools=tools if tools else [],
                max_tokens=4096,
                timeout=timeout,
            )

            total_tokens["input"] += resp.usage.input_tokens
            total_tokens["output"] += resp.usage.output_tokens

            text_parts = []
            tool_use_blocks = []

            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            turn_text = "\n".join(text_parts)
            turns.append({
                "role": "assistant",
                "content": turn_text,
                "tool_calls": [{"name": tb.name, "arguments": tb.input} for tb in tool_use_blocks],
            })

            if not tool_use_blocks:
                break

            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            for tb in tool_use_blocks:
                tool_call = ToolCall(tool=tb.name, arguments=tb.input)
                all_tool_calls.append(tool_call)

                if tool_result_provider:
                    mock_result = tool_result_provider.get_result(tb.name, tb.input)
                else:
                    mock_result = {"status": "success", "mock": True, "tool": tb.name}
                tool_call.result = mock_result

                was_error = mock_result.get("error", False)
                tool_call_timeline.append({
                    "turn": len(turns),
                    "tool": tb.name,
                    "arguments": tb.input,
                    "result": mock_result,
                    "was_error": was_error,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": json.dumps(mock_result, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})

        elapsed = (time.monotonic() - start) * 1000

        final_text = ""
        for t in reversed(turns):
            if t.get("content"):
                final_text = t["content"]
                break

        return DialogueResult(
            turns=turns,
            tool_calls=all_tool_calls,
            final_text=final_text,
            total_latency_ms=elapsed,
            token_usage=total_tokens,
            tool_call_timeline=tool_call_timeline,
        )
