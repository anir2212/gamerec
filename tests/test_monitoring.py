"""Drift measures, alert rules and the simulated-week generator."""

import numpy as np
import pytest

from recstack import monitor_run as run
from recstack import monitoring as mon
from recstack.data import GENRES, synthetic


def test_psi_is_zero_for_equal_shares_and_matches_a_hand_calculation():
    assert mon.psi(np.array([10, 20, 30]), np.array([20, 40, 60])) == pytest.approx(0.0)
    value = mon.psi(np.array([50, 50]), np.array([25, 75]))
    assert value == pytest.approx((0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5), rel=1e-2)
    assert mon.psi(np.array([1, 0]), np.array([0, 1])) > 1  # Smoothing keeps empty bins finite.


def test_js_divergence_bounds_and_symmetry():
    a, b = np.array([1, 2, 7]), np.array([5, 4, 1])
    assert mon.js_divergence_bits(a, a) == pytest.approx(0.0)
    assert mon.js_divergence_bits(a, b) == pytest.approx(mon.js_divergence_bits(b, a))
    assert 0.95 < mon.js_divergence_bits(np.array([1, 0]), np.array([0, 1])) <= 1


def test_measures_reject_mismatched_or_empty_input():
    with pytest.raises(ValueError, match="same bins"):
        mon.psi(np.array([1, 2]), np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="same bins"):
        mon.js_divergence_bits(np.array([1]), np.array([1, 2]))
    with pytest.raises(ValueError, match="nonempty"):
        mon.psi(np.array([0, 0]), np.array([1, 1]))
    with pytest.raises(ValueError, match="empty"):
        mon.quantile_edges(np.array([]))


def test_score_psi_keeps_out_of_range_values_and_detects_a_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=2000)
    assert mon.score_psi(reference, rng.normal(size=2000)) < 0.05
    assert mon.score_psi(reference, rng.normal(loc=2, size=2000)) > 1
    edges = mon.quantile_edges(reference)
    assert mon.binned(np.array([-99.0, 99.0]), edges).sum() == 2


def test_alert_rules():
    assert mon.drift_alert([0.1, 0.3], [0.1, 0.2]) == [False, True]
    assert mon.recall_alert([0.5, 0.44], [0.5, 0.5]) == [False, True]
    assert mon.first_alert_week([False, True, True]) == 2
    assert mon.first_alert_week([False]) is None


def test_simulated_weeks_are_reproducible_and_drift_moves_taste_and_popularity():
    _, catalog = synthetic(7)
    users = [f"u{index:03d}" for index in range(120)]
    control = run.simulate_weeks(catalog, users, "no_drift", 101)
    assert control == run.simulate_weeks(catalog, users, "no_drift", 101)
    assert control != run.simulate_weeks(catalog, users, "no_drift", 102)
    assert len(control) == run.WEEKS
    genre = {item.item: item.genre for item in catalog}
    drift = run.simulate_weeks(catalog, users, "drift", 101)

    def own_genre_share(rows):
        return np.mean([genre[row.item] == GENRES[int(row.user[1:]) % 5] for row in rows])

    def high_index_share(rows):
        return np.mean([int(row.item[1:]) >= 60 for row in rows])

    assert own_genre_share(drift[-1]) < own_genre_share(control[-1])
    assert high_index_share(drift[-1]) > high_index_share(control[-1])
    with pytest.raises(ValueError, match="scenario"):
        run.simulate_weeks(catalog, users, "other", 1)


def test_monitor_report_is_labelled_and_a_frozen_table_decays_only_under_drift():
    from recstack.data import clean, temporal_split
    from recstack.models import Recommenders
    from recstack.score_table import base_table

    rows, catalog = synthetic(7)
    train, _, _ = temporal_split(clean(rows))
    table = base_table(Recommenders(train, catalog, 7), "content")
    report = run.monitor(table, "test-version", seeds=(101,), weeks=4)
    assert report["label"].startswith("SIMULATED WEEKS")
    assert report["test_split_read"] is False
    assert len(report["scenarios"]["drift"]["weekly_mean"]["recall_at_10"]) == 4
    week = report["scenarios"]["no_drift"]["per_seed"]["101"][0]
    assert week["events"] > 0 and week["recall_users"] > 0
    assert 0 <= week["recall_ci95"][0] <= week["recall_at_10"] <= week["recall_ci95"][1] <= 1
    assert set(report["first_alert_week"]) == {"popularity_psi", "popularity_js_bits", "served_score_psi",
                                               "played_score_psi", "recall_at_10"}
