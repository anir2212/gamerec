"""The promotion rule, with every doubtful input blocking."""

import math

import pytest

from recstack.gate import decide


def record(value=0.30, **changes):
    return {"metric": "ndcg@10", "split": "validation", "value": value, "users": 100, "eval_sha256": "abc", **changes}


def test_candidate_at_the_tolerance_boundary_passes_and_just_below_fails():
    champion = record(0.40)
    assert decide(record(0.395), champion, 0.005).passed
    assert decide(record(0.3949), champion, 0.005).reason == "below_champion_minus_tolerance"
    assert decide(record(0.50), champion, 0.005).reason == "within_tolerance_of_champion"
    assert not decide(record(0.3999), champion, 0.0).passed


def test_blocked_decision_reports_the_values_it_compared():
    decision = decide(record(0.2), record(0.4), 0.005)
    assert (decision.candidate_value, decision.champion_value) == (0.2, 0.4)
    assert decision.threshold == pytest.approx(0.395)


def test_first_champion_needs_the_popularity_floor():
    assert decide(record(0.3), None, floor=0.2).reason == "first_champion"
    assert decide(record(0.1), None, floor=0.2).reason == "below_first_champion_floor"
    assert decide(record(0.3), None).reason == "missing_floor"
    assert decide(record(0.3), None, floor=math.nan).reason == "missing_floor"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.5, None, "0.4", True])
def test_invalid_candidate_metrics_block(bad):
    assert decide(record(bad), record(0.1)).reason == "invalid_candidate_metric"


def test_failed_evaluation_blocks_even_with_no_champion():
    assert decide(None, None, floor=0.0).reason == "evaluation_failed"
    assert decide(None, record(0.1)).reason == "evaluation_failed"


@pytest.mark.parametrize(("changes", "reason"), [
    ({"metric": "recall@10"}, "wrong_metric_or_split"),
    ({"split": "test"}, "wrong_metric_or_split"),
    ({"users": 0}, "no_scored_users"),
    ({"eval_sha256": ""}, "missing_evaluation_hash"),
    ({"eval_sha256": "other"}, "evaluation_rows_differ")])
def test_records_that_cannot_be_compared_block(changes, reason):
    assert decide(record(0.9, **changes), record(0.1)).reason == reason


def test_invalid_champion_records_block_and_bad_tolerance_raises():
    assert decide(record(0.9), record(0.1, split="test")).reason == "champion_metric_not_comparable"
    assert decide(record(0.9), record(math.nan)).reason == "invalid_champion_metric"
    with pytest.raises(ValueError, match="tolerance"):
        decide(record(), record(), tolerance=-0.1)
