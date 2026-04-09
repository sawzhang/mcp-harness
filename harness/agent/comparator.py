"""
TraceComparator — 评估引擎核心

DTW 动态时间规整对齐两个 tool call 序列 + Cart 状态 Diff。

DTW 选型原因：不同 Agent 可能用不同步数完成同一任务，
动态时间规整做弹性序列对齐比 exact match 鲁棒，
避免因步数差异误判为失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace import AgentTrace


@dataclass
class AlignmentResult:
    """Result of DTW alignment between two tool sequences."""

    reference: list[str]
    candidate: list[str]
    alignment: list[tuple[int, int]]  # Pairs of (ref_idx, cand_idx)
    distance: float = 0.0
    normalized_distance: float = 0.0  # distance / max(len(ref), len(cand))

    @property
    def insertions(self) -> int:
        """Extra tools in candidate not matched to reference."""
        matched_cand = {a[1] for a in self.alignment}
        return len(self.candidate) - len(matched_cand)

    @property
    def deletions(self) -> int:
        """Tools in reference not matched to candidate."""
        matched_ref = {a[0] for a in self.alignment}
        return len(self.reference) - len(matched_ref)


def dtw_align(seq_a: list[str], seq_b: list[str]) -> AlignmentResult:
    """
    Dynamic Time Warping alignment of two tool call sequences.
    Cost: 0 for match, 1 for mismatch.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return AlignmentResult(
            reference=seq_a, candidate=seq_b, alignment=[], distance=0
        )
    if n == 0:
        return AlignmentResult(
            reference=seq_a, candidate=seq_b, alignment=[], distance=m,
            normalized_distance=1.0,
        )
    if m == 0:
        return AlignmentResult(
            reference=seq_a, candidate=seq_b, alignment=[], distance=n,
            normalized_distance=1.0,
        )

    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # match/mismatch
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
            )

    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1

    alignment.reverse()
    distance = dp[n][m]
    max_len = max(n, m) or 1

    return AlignmentResult(
        reference=seq_a,
        candidate=seq_b,
        alignment=alignment,
        distance=distance,
        normalized_distance=distance / max_len,
    )


@dataclass
class CartDiff:
    """Diff between expected and actual cart states."""

    missing_items: list[dict] = field(default_factory=list)
    extra_items: list[dict] = field(default_factory=list)
    modified_items: list[dict] = field(default_factory=list)
    # Each: {"item": str, "field": str, "expected": Any, "actual": Any}
    is_equivalent: bool = True


def diff_cart_state(expected: dict, actual: dict) -> CartDiff:
    """Compare expected vs actual cart/order state."""
    diff = CartDiff()
    expected_items = expected.get("items", [])
    actual_items = actual.get("items", [])

    expected_by_code = {
        item.get("product_code", f"_idx_{i}"): item
        for i, item in enumerate(expected_items)
    }
    actual_by_code = {
        item.get("product_code", f"_idx_{i}"): item
        for i, item in enumerate(actual_items)
    }

    for code, exp_item in expected_by_code.items():
        if code not in actual_by_code:
            diff.missing_items.append(exp_item)
            diff.is_equivalent = False
        else:
            act_item = actual_by_code[code]
            for key, exp_val in exp_item.items():
                act_val = act_item.get(key)
                if exp_val != act_val:
                    diff.modified_items.append(
                        {
                            "item": code,
                            "field": key,
                            "expected": exp_val,
                            "actual": act_val,
                        }
                    )
                    diff.is_equivalent = False

    for code in actual_by_code:
        if code not in expected_by_code:
            diff.extra_items.append(actual_by_code[code])
            diff.is_equivalent = False

    return diff


class TraceComparator:
    """
    Compare agent traces:
    - Two traces from different agents/models on the same case
    - A trace against ground truth order state
    """

    def compare_traces(
        self, reference: AgentTrace, candidate: AgentTrace
    ) -> dict:
        """Compare two traces from different agents/models."""
        alignment = dtw_align(reference.tool_sequence, candidate.tool_sequence)
        cart_diff = diff_cart_state(
            reference.final_cart_state, candidate.final_cart_state
        )

        return {
            "alignment": {
                "distance": alignment.distance,
                "normalized_distance": alignment.normalized_distance,
                "insertions": alignment.insertions,
                "deletions": alignment.deletions,
            },
            "cart_diff": {
                "is_equivalent": cart_diff.is_equivalent,
                "missing_items": len(cart_diff.missing_items),
                "extra_items": len(cart_diff.extra_items),
                "modified_fields": len(cart_diff.modified_items),
            },
            "efficiency_ratio": (
                len(reference.tool_sequence) / len(candidate.tool_sequence)
                if candidate.tool_sequence
                else 0
            ),
        }

    def compare_to_ground_truth(
        self, trace: AgentTrace, ground_truth: dict
    ) -> dict:
        """Compare a trace's final state against ground truth order."""
        cart_diff = diff_cart_state(ground_truth, trace.final_cart_state)
        return {
            "is_equivalent": cart_diff.is_equivalent,
            "missing_items": cart_diff.missing_items,
            "extra_items": cart_diff.extra_items,
            "modified_items": cart_diff.modified_items,
            "tool_count": len(trace.tool_sequence),
        }
