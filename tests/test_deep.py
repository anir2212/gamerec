"""Two-tower, re-ranker and multi-seed evaluation behaviour on small synthetic fixtures."""

import numpy as np
import pytest
import torch

from recstack import deep_run
from recstack import reranker as rr
from recstack import two_tower as tt
from recstack.data import Interaction, Item, clean, synthetic, temporal_split
from recstack.eval import evaluate, evaluate_seeds, paired_difference, seed_averaged_rows
from recstack.models import Recommenders
from recstack.score_table import ScoredModels, base_table

FAST = tt.TowerConfig(dim=8, epochs=2)


@pytest.fixture(scope="module")
def corpus():
    rows, catalog = synthetic(seed=7, users=30, items=25)
    train, validation, test = temporal_split(clean(rows))
    return train, validation, test, catalog, Recommenders(train, catalog, seed=7), tt.Features(train, catalog)


def test_content_vectors_drop_the_number_that_would_act_as_an_item_id():
    catalog = [Item("a", "strategy", "A strategy game with variant 41"),
               Item("b", "strategy", "A strategy game with variant 42"),
               Item("c", "racing", "A racing game with variant 43")]
    vectors = tt.content_vectors(catalog)
    assert np.array_equal(vectors[0], vectors[1])
    assert not np.array_equal(vectors[0], vectors[2])


def test_training_examples_never_contain_the_target_in_its_own_history(corpus):
    _, _, _, _, _, features = corpus
    examples = features.training_examples()
    rows = np.arange(len(examples["target"]))
    assert np.all(examples["history"][rows, examples["target"]] == 0)
    assert np.all(examples["recency"][rows, examples["target"]] == 0)
    assert np.isclose(examples["history"].sum(axis=1), 1).all()


def test_features_only_see_training_rows(corpus):
    train, _, test, catalog, _, features = corpus
    for user, rows in features.timelines.items():
        assert max(row.time for row in rows) <= max(row.time for row in train if row.user == user)
    held_out = {(row.user, row.item, row.time) for row in test}
    assert not held_out.intersection((row.user, row.item, row.time)
                                     for rows in features.timelines.values() for row in rows)
    assert features.counts.sum() == len(train)
    assert len(catalog) == features.n_items


def test_cold_item_vector_is_content_only():
    train = [Interaction("u", "warm", time, 30) for time in range(6)] + [Interaction("u", "cold", 7, 30)]
    catalog = [Item("warm", "strategy", "strategy game"), Item("cold", "strategy", "strategy game"),
               Item("cold2", "strategy", "strategy game")]
    features = tt.Features(train, catalog)
    assert features.cold.tolist() == [False, True, True]
    model = tt.TwoTower(features, FAST)
    with torch.no_grad():
        model.eval()
        before = model.item_vectors()
        model.id_embedding.weight[1:] += 5.0  # Changing a cold item's ID embedding must not matter.
        after = model.item_vectors()
        assert torch.equal(before[1:], after[1:])
        assert torch.equal(after[1], after[2])  # Same copy and genre: identical vectors.
        model.id_embedding.weight[0] += 5.0
        assert not torch.equal(after[0], model.item_vectors()[0])


def test_variants_remove_the_channel_they_claim_to_remove(corpus):
    _, _, _, _, _, features = corpus
    content = tt.TwoTower(features, tt.TowerConfig(dim=8, variant="content_only"))
    ids = tt.TwoTower(features, tt.TowerConfig(dim=8, variant="id_only"))
    with torch.no_grad():
        content.eval()
        ids.eval()
        before = content.item_vectors()
        content.id_embedding.weight += 1.0
        assert torch.equal(before, content.item_vectors())
        before = ids.item_vectors()
        ids.text_projection.weight += 1.0
        ids.genre_embedding.weight += 1.0
        assert torch.equal(before, ids.item_vectors())
    with pytest.raises(ValueError, match="variant"):
        tt.TowerConfig(variant="unknown")


def test_in_batch_loss_removes_duplicate_and_owned_negatives():
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    none = torch.zeros((3, 3), dtype=torch.bool)
    distinct = tt.in_batch_loss(users, positives, torch.tensor([0, 1, 2]), none, 1.0)
    # Row 0 and row 1 now share a target, so each stops treating the other's copy as a negative.
    duplicate = tt.in_batch_loss(users, positives, torch.tensor([0, 0, 2]), none, 1.0)
    assert duplicate < distinct
    owned = none.clone()
    owned[0, 1] = owned[0, 2] = True
    masked = tt.in_batch_loss(users, positives, torch.tensor([0, 1, 2]), owned, 1.0)
    assert masked < distinct


def test_logq_correction_raises_the_logit_of_a_frequently_sampled_negative():
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    none = torch.zeros((2, 2), dtype=torch.bool)
    # Item 1 is very frequent. Row 0's negative gets log q subtracted, which makes it harder to beat.
    log_q = torch.log(torch.tensor([0.01, 0.99]))
    plain = tt.in_batch_loss(users[:1].repeat(2, 1), positives, torch.tensor([0, 1]), none, 1.0)
    corrected = tt.in_batch_loss(users[:1].repeat(2, 1), positives, torch.tensor([0, 1]), none, 1.0, log_q)
    assert corrected > plain


def test_fit_is_deterministic_for_a_seed_and_differs_between_seeds(corpus):
    _, _, _, _, _, features = corpus
    first = tt.score_table(tt.fit(features, FAST, 1), features)
    again = tt.score_table(tt.fit(features, FAST, 1), features)
    other = tt.score_table(tt.fit(features, FAST, 2), features)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    content = tt.score_table(tt.fit(features, tt.TowerConfig(dim=8, epochs=1, variant="content_only"), 1), features)
    same_genre = np.flatnonzero(features.genre_ids == features.genre_ids[0])
    assert np.all(content[:, same_genre] == content[:, [same_genre[0]]])  # Exact ties, not rounding noise.
    assert first.shape == (len(features.user_ids), features.n_items)
    assert np.isfinite(first).all()


def test_score_table_recommendations_exclude_owned_and_reject_bad_tables(corpus):
    train, _, _, _, base, features = corpus
    table = tt.score_table(tt.fit(features, FAST, 1), features)
    scored = ScoredModels(base, {"two_tower": table})
    user = train[0].user
    ranked = scored.recommend(user, "two_tower", 10)
    assert not base.history[user].intersection(ranked)
    assert len(set(ranked)) == len(ranked) == 10
    assert np.array_equal(scored.scores(user, "itemknn"), base.scores(user, "itemknn"))
    with pytest.raises(ValueError, match="shape"):
        ScoredModels(base, {"bad": table[:1]})
    broken = table.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ScoredModels(base, {"bad": broken})


def test_new_slices_partition_the_overall_denominator(corpus):
    _, _, test, _, base, _ = corpus
    from recstack.eval import _per_user
    everyone = {row["user"] for row in _per_user(base, test, "itemknn", 10)[0]}
    new = {row["user"] for row in _per_user(base, test, "itemknn", 10, "new_user")[0]}
    warm = {row["user"] for row in _per_user(base, test, "itemknn", 10, "warm_user")[0]}
    cold = {row["user"] for row in _per_user(base, test, "itemknn", 10, "cold_item")[0]}
    warm_item = {row["user"] for row in _per_user(base, test, "itemknn", 10, "warm_item")[0]}
    assert new | warm == everyone and not new & warm
    assert cold | warm_item == everyone  # A user can hold targets in both item slices.


def test_evaluate_seeds_with_one_seed_matches_evaluate(corpus):
    _, _, test, _, base, _ = corpus
    single = evaluate(base, test, "itemknn", 10, bootstrap_samples=50)
    seeded = evaluate_seeds([base], test, "itemknn", 10, bootstrap_samples=50)
    assert seeded["users"] == single["users"]
    assert seeded["metrics"]["ndcg"] == pytest.approx(single["metrics"]["ndcg"])
    assert seeded["ci95"]["ndcg"] == pytest.approx(single["ci95"]["ndcg"])
    assert seeded["seed_sd_ndcg"] == 0


def test_evaluate_seeds_averages_and_reports_seed_spread(corpus):
    _, _, test, _, base, features = corpus
    scored = [ScoredModels(base, {"m": tt.score_table(tt.fit(features, FAST, seed), features)}) for seed in (1, 2, 3)]
    result = evaluate_seeds(scored, test, "m", 10, bootstrap_samples=50)
    assert result["seeds"] == 3
    assert result["metrics"]["ndcg"] == pytest.approx(sum(result["per_seed_ndcg"]) / 3)
    assert result["seed_sd_ndcg"] > 0


def test_paired_difference_on_a_known_fixture_and_user_mismatch():
    a = [{"user": "x", "ndcg": 1.0}, {"user": "y", "ndcg": 0.5}, {"user": "z", "ndcg": 0.5}]
    b = [{"user": "x", "ndcg": 0.0}, {"user": "y", "ndcg": 0.5}, {"user": "z", "ndcg": 0.25}]
    result = paired_difference(a, b, bootstrap_samples=500)
    assert result["mean_difference"] == pytest.approx(1.25 / 3)
    assert (result["users_a_better"], result["users_b_better"], result["users_tied"]) == (2, 0, 1)
    assert 0 <= result["ci95"][0] <= result["ci95"][1] <= 1
    with pytest.raises(ValueError, match="identical users"):
        paired_difference(a, b[:2])


def test_seed_rows_reject_disagreeing_user_sets(corpus):
    train, _, test, catalog, base, _ = corpus
    other = Recommenders(train[:len(train) // 2] + train[len(train) // 2:][:5], catalog, seed=7)
    with pytest.raises(ValueError, match="disagree"):
        seed_averaged_rows([base, other], test, "itemknn", 10)


def test_reranker_uses_training_features_and_novel_validation_labels(corpus):
    train, validation, _, _, base, features = corpus
    table = tt.score_table(tt.fit(features, FAST, 1), features)
    arrays = rr.feature_arrays(base, table, train)
    assert set(rr.PLUS_FEATURES) <= set(arrays)
    targets = rr.novel_targets(base, validation)
    for user, items in targets.items():
        assert not items & base.history[user]
    shares = arrays["genre_match"]
    position = base.user_index[train[0].user]
    assert np.unique(shares[position]).sum() == pytest.approx(1.0)
    config = rr.RerankConfig(rounds=5, min_data_in_leaf=5)
    ranker = rr.fit_ranker(base, arrays, targets, sorted(targets), config, 3)
    predicted = rr.predict_table(base, arrays, ranker, config)
    again = rr.predict_table(base, arrays, rr.fit_ranker(base, arrays, targets, sorted(targets), config, 3), config)
    assert np.array_equal(predicted, again)
    user = sorted(targets)[0]
    owned_positions = [base.item_index[item] for item in base.history[user]]
    assert np.all(predicted[base.user_index[user], owned_positions] == -1e6)


def test_out_of_fold_never_fits_a_user_it_scores(corpus, monkeypatch):
    train, validation, _, _, base, features = corpus
    arrays = rr.feature_arrays(base, tt.score_table(tt.fit(features, FAST, 1), features), train)
    targets = rr.novel_targets(base, validation)
    fitted_sets, real_fit, real_predict = [], rr.fit_ranker, rr.predict_table

    def spy_fit(base_, arrays_, targets_, users, config, seed):
        fitted_sets.append(set(users))
        return real_fit(base_, arrays_, targets_, users, config, seed)

    def spy_predict(base_, arrays_, ranker, config, users=None, into=None):
        assert not fitted_sets[-1] & set(users)
        return real_predict(base_, arrays_, ranker, config, users, into)

    monkeypatch.setattr(rr, "fit_ranker", spy_fit)
    monkeypatch.setattr(rr, "predict_table", spy_predict)
    table = rr.out_of_fold_table(base, arrays, targets, rr.RerankConfig(rounds=3, min_data_in_leaf=5), 3)
    assert len(fitted_sets) == 5
    for user in targets:
        assert table[base.user_index[user]].max() > -1e6


def test_select_reads_validation_only_and_compare_reports_all_models(corpus, monkeypatch):
    train, validation, test, catalog, _, _ = corpus
    monkeypatch.setattr(deep_run, "NEURAL_SEEDS", (1, 2))
    monkeypatch.setattr(deep_run, "TOWER_GRID", [tt.TowerConfig(dim=8, epochs=1), tt.TowerConfig(dim=8, epochs=2)])
    monkeypatch.setattr(deep_run, "RERANK_GRID", [rr.RerankConfig(rounds=3, min_data_in_leaf=5)])
    monkeypatch.setattr(deep_run, "tower_tables", lambda features, config, seeds=(1, 2): [
        tt.score_table(tt.fit(features, config, seed), features) for seed in seeds])
    selection = deep_run.select(train, validation, catalog, corpus_seed=7)
    assert selection["test_split_read"] is False
    assert "test" not in selection["split_sha256"]
    best = max(selection["two_tower_grid"], key=lambda row: row["ndcg"])
    assert selection["two_tower_selected"] == best["config"]
    assert selection["validation_winner"] == next(iter(selection["validation_leaderboard"]))
    report = deep_run.compare(train, validation, test, catalog, selection, corpus_seed=7)
    assert "known fixture" in report["test_status"]
    assert {row["model"] for row in report["test"]} == {
        *deep_run.BASELINES, "two_tower", "two_tower_id_only", "two_tower_content_only", "reranker"}
    assert set(report["paired_test_ndcg10"]) == {entry[0] for entry in deep_run.PAIRED}
    assert base_table(Recommenders(train, catalog, 7), "text").shape[1] == len(catalog)
