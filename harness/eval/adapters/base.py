"""
AgentAdapter 统一接口

每个模型实现一个 Adapter。Eval Harness 通过 Adapter 模拟真实用户场景：
给 Agent 自然语言指令 + MCP Tool specs → Agent 自主决策调用哪个 Tool。

测的不是 MCP Server，是 Tool Description × Model 的组合效果。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    result: Any = None


@dataclass
class DialogueResult:
    turns: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    total_latency_ms: float = 0
    token_usage: dict[str, int] = field(default_factory=dict)

    def extract_tool_calls(self) -> list[ToolCall]:
        return self.tool_calls

    def has_tool(self, tool_name: str) -> bool:
        return any(tc.tool == tool_name for tc in self.tool_calls)

    def get_tool_call(self, tool_name: str) -> ToolCall | None:
        for tc in self.tool_calls:
            if tc.tool == tool_name:
                return tc
        return None


class AgentAdapter(ABC):
    """
    模型适配器接口。每个模型的 function calling 格式不同：
    - Qianwen: functions 格式
    - Claude: tool_use 格式
    - GPT: tools 格式
    Tool Description 相同，但传入格式不同——这本身就是测试点。
    """

    name: str = "base"

    @abstractmethod
    async def run_dialogue(
        self,
        system_prompt: str,
        user_message: str,
        mcp_tools: list[dict],
        context: dict | None = None,
        max_turns: int = 5,
        timeout: int = 30,
    ) -> DialogueResult:
        ...

    async def run_multi_turn(
        self,
        system_prompt: str,
        turns: list[str],
        mcp_tools: list[dict],
        context: dict | None = None,
        max_turns_per_message: int = 3,
        timeout: int = 30,
    ) -> list[DialogueResult]:
        results = []
        conversation_history: list[dict] = []

        for user_msg in turns:
            # 将历史对话拼入 user_message，让单轮 adapter 也能感知上下文
            augmented_msg = user_msg
            if conversation_history:
                history_text = "\n".join(
                    f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                    for m in conversation_history
                )
                augmented_msg = (
                    f"[对话历史]\n{history_text}\n\n"
                    f"[当前用户消息]\n{user_msg}"
                )

            result = await self.run_dialogue(
                system_prompt=system_prompt,
                user_message=augmented_msg,
                mcp_tools=mcp_tools,
                context=context,
                max_turns=max_turns_per_message,
                timeout=timeout,
            )
            results.append(result)

            # 记录完整交互（包括 tool calls）用于下轮上下文
            conversation_history.append({"role": "user", "content": user_msg})
            tool_summary = ""
            if result.tool_calls:
                tool_summary = " [调用了: " + ", ".join(
                    f"{tc.tool}({json.dumps(tc.arguments, ensure_ascii=False)[:80]})"
                    for tc in result.tool_calls
                ) + "]"
            conversation_history.append({
                "role": "assistant",
                "content": (result.final_text or "OK") + tool_summary,
            })

        return results


ORDERING_AGENT_SYSTEM_PROMPT = """你是一个咖啡点单助手。用户会用自然语言告诉你想点什么，你需要使用提供的 MCP 工具来完成点单流程。

规则：
1. 星巴克杯型映射：中杯=tall, 大杯=grande, 超大杯=venti（不是 medium/large）
2. 下单前必须先调用 calculate_price 获取确认信息，展示给用户确认
3. 用户确认后才能调用 create_order，必须传入 confirmation_token 和 idempotency_key
4. 如果用户要修改已选商品的属性（杯型、温度等），更新对应字段，不要新增商品
5. 价格由服务端计算，不要自己编造价格
"""
