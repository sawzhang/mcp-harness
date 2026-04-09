"""
OpenAI-compatible Adapter (GPT-4o, Qianwen-Max, Doubao, etc.)

Qianwen 和 Doubao 都兼容 OpenAI API 格式，通过 base_url 切换。
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import AgentAdapter, DialogueResult, ToolCall

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


def _convert_mcp_to_openai_tools(mcp_tools: list[dict]) -> list[dict]:
    tools = []
    for t in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return tools


class OpenAIAdapter(AgentAdapter):

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        name: str | None = None,
    ):
        if AsyncOpenAI is None:
            raise ImportError("pip install openai")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
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
        tools = _convert_mcp_to_openai_tools(mcp_tools)

        context_text = ""
        if context:
            context_text = f"\n\n当前上下文：{json.dumps(context, ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message + context_text},
        ]

        all_tool_calls = []
        turns = []
        tool_call_timeline = []
        start = time.monotonic()
        total_tokens = {"input": 0, "output": 0}

        for _ in range(max_turns):
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                timeout=timeout,
            )

            choice = resp.choices[0]
            msg = choice.message
            turns.append({"role": "assistant", "content": msg.content, "tool_calls": None})

            if resp.usage:
                total_tokens["input"] += resp.usage.prompt_tokens
                total_tokens["output"] += resp.usage.completion_tokens

            if not msg.tool_calls:
                break

            turns[-1]["tool_calls"] = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}

                tool_call = ToolCall(tool=tc.function.name, arguments=args)
                all_tool_calls.append(tool_call)

                # Mock tool result — Eval 模式下不真实调用 MCP
                if tool_result_provider:
                    mock_result = tool_result_provider.get_result(tc.function.name, args)
                else:
                    mock_result = {"status": "success", "mock": True, "tool": tc.function.name}
                tool_call.result = mock_result

                was_error = mock_result.get("error", False)
                tool_call_timeline.append({
                    "turn": len(turns),
                    "tool": tc.function.name,
                    "arguments": args,
                    "result": mock_result,
                    "was_error": was_error,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(mock_result, ensure_ascii=False),
                })

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
