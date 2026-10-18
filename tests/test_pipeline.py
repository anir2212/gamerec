import pytest

from recstack.data import Interaction, assert_no_temporal_leakage, clean, split_stats, synthetic, temporal_split
from recstack.eval import evaluate
from recstack.models import Recommenders


def test_temporal_split_rejects_a_future_training_event():
    rows, _ = synthetic(seed=7)
    train, validation, test = temporal_split(clean(rows))
    assert_no_temporal_leakage(train, validation, test)
    user = test[0].user
    future = next(row for row in test if row.user == user)
    corrupted_train = train + [Interaction(user, future.item, future.time, future.minutes)]
    with pytest.raises(ValueError, match="temporal leakage"):
        assert_no_temporal_leakage(corrupted_train, validation, test)


def test_clean_caps_outliers_and_cold_subsets_are_present():
    rows, catalog = synthetic(seed=7)
    assert clean([Interaction("u", "g", 1, 0), Interaction("u", "g", 2, 5000)]) == [
        Interaction("u", "g", 2, 1000)
    ]
    train, val, test = temporal_split(clean(rows))
    stats = split_stats(train, val, test, catalog)
    assert stats["new_users_at_most_3_train"] > 0
    assert stats["test_events_on_cold_items"] > 0


def test_models_exclude_owned_and_evaluation_has_a_valid_denominator():
    rows, catalog = synthetic(seed=7, users=30, items=25)
    train, _, test = temporal_split(clean(rows))
    models = Recommenders(train, catalog, seed=7)
    user = train[0].user
    owned = models.history[user]
    for method in ("popularity", "itemknn", "als", "bpr", "content", "text"):
        ranked = models.recommend(user, method, 10)
        assert not owned.intersection(ranked)
        assert len(set(ranked)) == len(ranked)
        result = evaluate(models, test, method, 10, bootstrap_samples=30)
        assert result["users"] > 0
        assert 0 <= result["metrics"]["recall"] <= 1
        assert result["ci95"]["ndcg"][0] <= result["ci95"]["ndcg"][1]


def test_text_model_uses_descriptions_for_unseen_same_genre_items():
    from recstack.data import Item

    catalog = [
        Item("played", "strategy", "orbital colony management"),
        Item("similar", "strategy", "space colony management"),
        Item("different", "strategy", "tactical knight battles"),
    ]
    train = [Interaction("u", "played", 1, 30)]
    models = Recommenders(train, catalog, seed=7)
    scores = models.scores("u", "text")
    assert scores[models.item_index["similar"]] > scores[models.item_index["different"]]
    assert models.recommend("u", "text", 2)[0] == "similar"
