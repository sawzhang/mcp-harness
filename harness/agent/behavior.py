"""
Agent 行为分析 — 5 维评估

与 LLM 能力层正交，聚焦 Agent 框架的编排行为：
1. Planning: 规划步骤数、并行率、冗余搜索率
2. Recovery: 错误恢复成功率、额外步骤开销
3. Loop Detection: 搜索循环、依赖循环、确认循环
4. State: 购物车状态准确性（需 ground_truth）
5. Efficiency: optimal_calls / actual_calls
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .trace import AgentTrace


@dataclass
class PlanningMetrics:
    plan_step_count: int = 0
    unique_tool_count: int = 0
    parallel_tool_ratio: float = 0.0  # turns with >1 tool / total turns
    redundant_search_rate: float = 0.0  # repeated same-tool+args / total calls


@dataclass
class RecoveryMetrics:
    total_errors: int = 0
    recovery_attempts: int = 0
    recovery_success_rate: float = 0.0
    recovery_step_overhead: float = 0.0


@dataclass
class LoopDetection:
    search_loops: list[dict] = field(default_factory=list)
    # Each: {"tool": str, "count": int, "start_index": int}
    dependency_loops: list[dict] = field(default_factory=list)
    # Each: {"pattern": "A -> B -> A", "start_index": int}
    total_loops_detected: int = 0
    budget_exceeded: bool = False


@dataclass
class StateMetrics:
    cart_state_accuracy: float = 0.0
    cross_turn_reference_rate: float = 0.0


@dataclass
class EfficiencyMetrics:
    optimal_tool_count: int = 0
    actual_tool_count: int = 0
    chaining_efficiency: float = 0.0  # optimal / actual (1.0 = perfect)


@dataclass
class BehaviorReport:
    planning: PlanningMetrics = field(default_factory=PlanningMetrics)
    recovery: RecoveryMetrics = field(default_factory=RecoveryMetrics)
    loops: LoopDetection = field(default_factory=LoopDetection)
    state: StateMetrics = field(default_factory=StateMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)


def analyze_planning(trace: AgentTrace) -> PlanningMetrics:
    """Analyze planning behavior from a trace."""
    total_calls = len(trace.tool_sequence)
    unique = len(trace.unique_tools)

    # Parallel: turns with >1 tool call
    multi_tool_turns = sum(1 for t in trace.turns if len(t.tool_calls) > 1)
    total_turns = len(trace.turns) or 1
    parallel_ratio = multi_tool_turns / total_turns

    # Redundant: same tool + same args called more than once
    seen = set()
    redundant = 0
    for t in trace.turns:
        for tc in t.tool_calls:
            key = (tc.tool, str(sorted(tc.arguments.items())))
            if key in seen:
                redundant += 1
            seen.add(key)

    return PlanningMetrics(
        plan_step_count=total_calls,
        unique_tool_count=unique,
        parallel_tool_ratio=parallel_ratio,
        redundant_search_rate=redundant / total_calls if total_calls else 0,
    )


def detect_loops(trace: AgentTrace, max_repeat: int = 3) -> LoopDetection:
    """Detect pathological loop patterns in a trace."""
    loops = LoopDetection()
    seq = trace.tool_sequence

    # Pattern A: same tool called >= max_repeat times consecutively
    if len(seq) >= max_repeat:
        i = 0
        while i < len(seq):
            run_len = 1
            while i + run_len < len(seq) and seq[i + run_len] == seq[i]:
                run_len += 1
            if run_len >= max_repeat:
                loops.search_loops.append(
                    {"tool": seq[i], "count": run_len, "start_index": i}
                )
            i += run_len

    # Pattern C: dependency loop A -> B -> A
    for i in range(len(seq) - 2):
        if seq[i] == seq[i + 2] and seq[i] != seq[i + 1]:
            loops.dependency_loops.append(
                {"pattern": f"{seq[i]} -> {seq[i + 1]} -> {seq[i]}", "start_index": i}
            )

    loops.total_loops_detected = len(loops.search_loops) + len(loops.dependency_loops)

    # Budget check (simple heuristic)
    if len(seq) > 15:
        loops.budget_exceeded = True

    return loops


def analyze_recovery(trace: AgentTrace) -> RecoveryMetrics:
    """Analyze error recovery behavior."""
    errors = trace.error_tool_calls
    total_errors = len(errors)
    if total_errors == 0:
        return RecoveryMetrics()

    # Flatten all tool calls
    flat_calls = []
    for turn in trace.turns:
        flat_calls.extend(turn.tool_calls)

    recovery_attempts = 0
    recovery_successes = 0

    for i, tc in enumerate(flat_calls):
        if tc.was_error and i + 1 < len(flat_calls):
            recovery_attempts += 1
            next_tc = flat_calls[i + 1]
            if not next_tc.was_error:
                recovery_successes += 1

    return RecoveryMetrics(
        total_errors=total_errors,
        recovery_attempts=recovery_attempts,
        recovery_success_rate=(
            recovery_successes / total_errors if total_errors else 0
        ),
        recovery_step_overhead=(
            recovery_attempts / total_errors if total_errors else 0
        ),
    )


def analyze_efficiency(
    trace: AgentTrace, optimal_steps: int | None = None
) -> EfficiencyMetrics:
    """Analyze tool chaining efficiency."""
    actual = len(trace.tool_sequence)
    optimal = optimal_steps or actual
    return EfficiencyMetrics(
        optimal_tool_count=optimal,
        actual_tool_count=actual,
        chaining_efficiency=optimal / actual if actual else 1.0,
    )


def analyze_behavior(
    trace: AgentTrace, optimal_steps: int | None = None
) -> BehaviorReport:
    """Run all behavior analyses on a trace."""
    return BehaviorReport(
        planning=analyze_planning(trace),
        recovery=analyze_recovery(trace),
        loops=detect_loops(trace),
        efficiency=analyze_efficiency(trace, optimal_steps),
    )
