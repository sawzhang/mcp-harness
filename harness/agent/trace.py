"""
Agent Trace 数据模型

记录 Agent 执行一个场景的完整轨迹，供行为分析使用。
可从现有 DialogueResult 构建（无需修改 Adapter），也可由 Agent 框架直接填充。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """A single tool call within a turn."""

    tool: str
    arguments: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    latency_ms: float = 0
    was_error: bool = False


@dataclass
class TurnTrace:
    """Trace of one turn in a dialogue."""

    turn_number: int
    user_message: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    assistant_text: str = ""
    thought: str = ""  # Agent reasoning (if available from framework)


@dataclass
class AgentTrace:
    """Complete trace of an agent's execution on a scenario."""

    case_id: str
    agent_name: str
    model_name: str
    turns: list[TurnTrace] = field(default_factory=list)
    final_cart_state: dict = field(default_factory=dict)
    total_tool_calls: int = 0
    total_latency_ms: float = 0
    metadata: dict = field(default_factory=dict)

    @property
    def tool_sequence(self) -> list[str]:
        """Flat list of tool names in call order."""
        seq = []
        for turn in self.turns:
            for tc in turn.tool_calls:
                seq.append(tc.tool)
        return seq

    @property
    def unique_tools(self) -> set[str]:
        return set(self.tool_sequence)

    @property
    def error_tool_calls(self) -> list[ToolCallRecord]:
        errors = []
        for turn in self.turns:
            for tc in turn.tool_calls:
                if tc.was_error:
                    errors.append(tc)
        return errors


def trace_from_dialogue_result(
    case_id: str,
    agent_name: str,
    model_name: str,
    dialogue_result,
) -> AgentTrace:
    """Convert an existing DialogueResult into an AgentTrace for behavior analysis."""
    turns = []

    if dialogue_result.tool_call_timeline:
        # Use detailed timeline if available
        turns_by_num: dict[int, TurnTrace] = {}
        for entry in dialogue_result.tool_call_timeline:
            turn_num = entry.get("turn", 0)
            if turn_num not in turns_by_num:
                turns_by_num[turn_num] = TurnTrace(turn_number=turn_num)
            turn = turns_by_num[turn_num]
            turn.tool_calls.append(
                ToolCallRecord(
                    tool=entry.get("tool", ""),
                    arguments=entry.get("arguments", {}),
                    result=entry.get("result", {}),
                    latency_ms=entry.get("latency_ms", 0),
                    was_error=entry.get("was_error", False),
                )
            )
        turns = [turns_by_num[k] for k in sorted(turns_by_num)]
    else:
        # Fallback: build from flat tool_calls and turns
        for i, turn_data in enumerate(dialogue_result.turns):
            tc_records = []
            for tc_data in turn_data.get("tool_calls") or []:
                tc_records.append(
                    ToolCallRecord(
                        tool=tc_data.get("name", ""),
                        arguments=tc_data.get("arguments", {}),
                    )
                )
            turns.append(
                TurnTrace(
                    turn_number=i + 1,
                    assistant_text=turn_data.get("content", ""),
                    tool_calls=tc_records,
                )
            )

    # Fill assistant text from dialogue turns
    for i, turn in enumerate(turns):
        if i < len(dialogue_result.turns) and not turn.assistant_text:
            turn.assistant_text = dialogue_result.turns[i].get("content", "")

    return AgentTrace(
        case_id=case_id,
        agent_name=agent_name,
        model_name=model_name,
        turns=turns,
        total_tool_calls=len(dialogue_result.tool_calls),
        total_latency_ms=dialogue_result.total_latency_ms,
    )
