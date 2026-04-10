"""
Fingerprint Matrix — 多维 LLM/Agent 能力画像

不只给单一 pass_rate，而是为每个 Model 输出多维能力指纹，
直接指导 OmniRoute 路由权重和 Skill OS 补偿策略。

维度包括：
- 按 layer 分：tool_selection, param_mapping, multi_turn, safety, ...
- 按 Judge 分：product_recognition, tool_efficiency, modification_understanding, order_equivalence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DimensionScore:
    """Score for one evaluation dimension."""

    pass_rate: float = 0.0
    avg_score: float = 0.0  # For judge-scored dimensions (0-1)
    sample_count: int = 0


@dataclass
class ModelFingerprint:
    """Capability fingerprint for a single model."""

    model_name: str
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "overall": round(self.overall_score, 3),
            "dimensions": {
                k: {
                    "pass_rate": round(v.pass_rate, 3),
                    "avg_score": round(v.avg_score, 3),
                    "n": v.sample_count,
                }
                for k, v in self.dimensions.items()
            },
        }


@dataclass
class FingerprintMatrix:
    """Cross-model capability comparison matrix."""

    fingerprints: dict[str, ModelFingerprint] = field(default_factory=dict)

    def add_result(
        self,
        model: str,
        layer: str,
        passed: bool,
        judge_scores: dict[str, float] | None = None,
    ):
        """Accumulate a single eval result into the matrix."""
        if model not in self.fingerprints:
            self.fingerprints[model] = ModelFingerprint(model_name=model)
        fp = self.fingerprints[model]

        # Update layer dimension
        if layer not in fp.dimensions:
            fp.dimensions[layer] = DimensionScore()
        dim = fp.dimensions[layer]
        dim.sample_count += 1
        # Incremental average
        dim.pass_rate = (
            dim.pass_rate * (dim.sample_count - 1) + (1.0 if passed else 0.0)
        ) / dim.sample_count

        # Add judge scores if available
        if judge_scores:
            for dim_name, score in judge_scores.items():
                if dim_name not in fp.dimensions:
                    fp.dimensions[dim_name] = DimensionScore()
                d = fp.dimensions[dim_name]
                d.sample_count += 1
                d.avg_score = (
                    d.avg_score * (d.sample_count - 1) + score
                ) / d.sample_count

    def compute_overall(self):
        """Compute overall score for each model as average of layer pass_rates."""
        for fp in self.fingerprints.values():
            if fp.dimensions:
                layer_dims = [
                    d
                    for k, d in fp.dimensions.items()
                    if not k.startswith("product_")
                    and not k.startswith("tool_efficiency")
                    and not k.startswith("modification_")
                    and not k.startswith("order_equivalence")
                ]
                if layer_dims:
                    fp.overall_score = sum(d.pass_rate for d in layer_dims) / len(
                        layer_dims
                    )

    def to_dict(self) -> dict:
        self.compute_overall()
        return {
            model: fp.to_dict()
            for model, fp in sorted(self.fingerprints.items())
        }

    def get_model_fingerprint(self, model: str) -> ModelFingerprint | None:
        return self.fingerprints.get(model)
