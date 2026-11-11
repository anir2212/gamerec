"""TB-1 tie rule behaviour: shared, deterministic, popularity-neutral, validation-only."""

from pathlib import Path

import numpy as np
import pytest

from recstack import reranker as rr
from recstack import tiebreak as tb
from recstack import tiebreak_run as tbr
from recstack import two_tower as tt
from recstack.data import clean, synthetic, temporal_split
from recstack.models import Recommenders

PROTOCOL = Path("docs/TIEBREAK_PROTOCOL.md")


@pytest.fixture(scope="module")
def corpus():
    rows, catalog = synthetic(seed=7, users=30, items=25)
    train, validation, test = temporal_split(clean(rows))
    return train, validation, test, catalog, Recommenders(train, catalog, seed=7)


def test_order_key_is_deterministic_and_differs_between_permutations(corpus):
    base = corpus[4]
    assert tb.order_key(base.item_ids, 3) == tb.order_key(base.item_ids, 3)
    assert tb.order_key(base.item_ids, 3) != tb.order_key(base.item_ids, 4)
    assert len(set(tb.order_key(base.item_ids, 0))) == len(base.item_ids)


def test_the_declared_salt_and_permutation_set_match_the_protocol_file():
    text = PROTOCOL.read_text()
    assert tb.SALT.decode().rstrip("|") in text
    assert tb.PERMUTATIONS == tuple(range(20))
    assert "0, 1, ..., 19" in text


def test_the_rule_never_reorders_distinct_scores(corpus):
    base = corpus[4]
    user = base.user_ids[0]
    assert int(np.sum(np.diff(np.sort(base.scores(user, "als"))) == 0)) == 0
    expected = base.recommend(user, "als", 10)
    for seed in (0, 7, 19):
        wrapped = tb.TieBrokenModels(base, {}, tb.order_key(base.item_ids, seed))
        assert wrapped.recommend(user, "als", 10) == expected


def test_a_fully_tied_scorer_spreads_over_items_instead_of_taking_the_lowest_ids(corpus):
    base = corpus[4]
    user = base.user_ids[0]
    flat = np.zeros((len(base.user_ids), len(base.item_ids)))
    firsts = {model.recommend(user, "flat", 1)[0]
              for model in tb.wrap(base, {"flat": flat})}
    assert len(firsts) > 5
    lowest_id = sorted(item for item in base.item_ids if item not in base.history[user])[0]
    assert firsts != {lowest_id}


def test_a_scorer_without_ties_is_untouched_by_every_permutation(corpus):
    base = corpus[4]
    user = base.user_ids[0]
    expected = base.recommend(user, "content", 10)
    assert {tuple(model.recommend(user, "content", 10)) for model in tb.wrap(base)} == {tuple(expected)}


def test_the_shared_helper_still_excludes_owned_games_and_rejects_bad_input(corpus):
    base = corpus[4]
    user = base.user_ids[0]
    wrapped = tb.wrap(base)[0]
    ranked = wrapped.recommend(user, "popularity", 10)
    assert not set(ranked) & base.history[user]
    assert len(set(ranked)) == len(ranked)
    with pytest.raises(ValueError):
        wrapped.recommend(user, "popularity", 0)
    with pytest.raises(ValueError):
        tb.TieBrokenModels(base, {}, [1, 2, 3])


def test_wrap_tables_is_seed_major_and_covers_every_permutation(corpus):
    base = corpus[4]
    tables = [np.zeros((len(base.user_ids), len(base.item_ids))) for _ in range(3)]
    models = tb.wrap_tables(base, "flat", tables, permutations=(0, 1))
    assert len(models) == 6
    assert models[0].tie_keys == models[2].tie_keys == tb.order_key(base.item_ids, 0)
    assert models[1].tie_keys == tb.order_key(base.item_ids, 1)


def test_the_key_is_uncorrelated_with_popularity_and_with_item_id_order(corpus):
    report = tbr.neutrality(corpus[4])
    assert len(report["per_permutation"]) == len(tb.PERMUTATIONS)
    assert report["mean_abs_spearman_key_vs_popularity"] < 0.5
    assert report["mean_abs_spearman_key_vs_item_index"] < 0.5


def test_the_expectation_over_permutations_is_reproducible(corpus):
    train, validation, _, _, base = corpus
    first = tbr.baseline_rows(base, validation, "text")[0]
    second = tbr.baseline_rows(base, validation, "text")[0]
    assert [row["ndcg"] for row in first] == [row["ndcg"] for row in second]


def test_selection_reads_validation_only_and_reports_both_rules(corpus, monkeypatch):
    train, validation, test, catalog, _ = corpus
    monkeypatch.setattr(tbr, "TOWER_GRID", [tt.TowerConfig(dim=8, epochs=2)])
    monkeypatch.setattr(tbr, "RERANK_GRID", [rr.RerankConfig(num_leaves=7, rounds=20, features="plan")])
    monkeypatch.setattr(tb, "PERMUTATIONS", (0, 1))
    selection = tbr.select(train, validation, catalog, corpus_seed=99)
    assert selection["test_split_read_by_selection"] is False
    assert "test" not in selection["split_sha256"]
    assert selection["validation_winner_tb1"] == next(iter(selection["validation_leaderboard_tb1"]))
    assert set(selection["validation_leaderboard_tb1"]) == set(selection["validation_leaderboard_lowest_id_recomputed"])
    assert selection["baselines_validation"]["content"]["ndcg"] == pytest.approx(
        selection["baselines_validation"]["content"]["ndcg_lowest_id_rule"])
    row = selection["baselines_validation"]["text"]
    assert row["tie_permutations"] == 2 and row["model_seeds"] == 1
    assert row["sd_between_tie_permutations"] >= 0.0


def test_a_text_winner_reports_no_paired_difference(corpus):
    train, validation, _, _, base = corpus
    out = tbr.winner_vs_text(base, validation, "text", {})
    assert out["tb1"] is None and out["lowest_id"] is None
