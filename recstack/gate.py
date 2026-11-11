"""Promotion gate: the only place that decides whether a candidate may replace the champion.

It has no side effects, and every doubtful input blocks promotion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

METRIC = "ndcg@10"
SPLIT = "validation"
DEFAULT_TOLERANCE = 0.005


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    candidate_value: float | None = None
    champion_value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide(candidate: dict | None, champion: dict | None, tolerance: float = DEFAULT_TOLERANCE,
           floor: float | None = None) -> GateDecision:
    """Return a decision for evaluation records holding metric, split, value, users and eval_sha256.

    ``candidate=None`` means the evaluation failed. ``champion=None`` means no champion exists yet,
    and then the candidate must reach ``floor``, the popularity baseline on the same rows.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if candidate is None:
        return GateDecision(False, "evaluation_failed")
    value = candidate.get("value")
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        return GateDecision(False, "invalid_candidate_metric")
    if candidate.get("metric") != METRIC or candidate.get("split") != SPLIT:
        return GateDecision(False, "wrong_metric_or_split", value)
    if not candidate.get("users"):
        return GateDecision(False, "no_scored_users", value)
    if not candidate.get("eval_sha256"):
        return GateDecision(False, "missing_evaluation_hash", value)
    if champion is None:
        if floor is None or not math.isfinite(floor):
            return GateDecision(False, "missing_floor", value)
        if value < floor:
            return GateDecision(False, "below_first_champion_floor", value, None, floor)
        return GateDecision(True, "first_champion", value, None, floor)
    if champion.get("metric") != METRIC or champion.get("split") != SPLIT:
        return GateDecision(False, "champion_metric_not_comparable", value)
    if champion.get("eval_sha256") != candidate["eval_sha256"]:
        return GateDecision(False, "evaluation_rows_differ", value, champion.get("value"))
    champion_value = champion.get("value")
    if not isinstance(champion_value, int | float) or not math.isfinite(champion_value):
        return GateDecision(False, "invalid_champion_metric", value)
    threshold = champion_value - tolerance
    if value < threshold:
        return GateDecision(False, "below_champion_minus_tolerance", value, champion_value, threshold)
    return GateDecision(True, "within_tolerance_of_champion", value, champion_value, threshold)
