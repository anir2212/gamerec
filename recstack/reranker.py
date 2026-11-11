"""LightGBM LambdaRank re-ranker over every unowned catalog item (synthetic corpus only).

The plan's top-200 candidate list cannot exist in an 80-item catalog, so a query
holds all of the user's unowned items. Base models are fitted on training rows; the
ranker's labels are validation plays, so base-model scores and labels never share rows.
Hyperparameters are selected on out-of-fold validation users.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np

from recstack.data import Interaction
from recstack.models import Recommenders
from recstack.score_table import base_table

PLAN_FEATURES = ("als", "two_tower", "log_popularity", "genre_match", "recent_genre_match")
PLUS_FEATURES = PLAN_FEATURES + ("itemknn", "bpr", "text")
RECENT_PLAYS = 3


@dataclass(frozen=True)
class RerankConfig:
    num_leaves: int = 7
    rounds: int = 50
    features: str = "plan"
    learning_rate: float = 0.05
    min_data_in_leaf: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    def names(self) -> tuple[str, ...]:
        if self.features == "plan":
            return PLAN_FEATURES
        if self.features == "plus":
            return PLUS_FEATURES
        raise ValueError(f"unknown feature set {self.features}")


def feature_arrays(base: Recommenders, tower_table: np.ndarray, train: list[Interaction]) -> dict[str, np.ndarray]:
    """Build (users, items) feature arrays from training data and base-model scores."""
    users, items = len(base.user_ids), len(base.item_ids)
    genres = sorted({base.genre[item] for item in base.item_ids})
    genre_index = np.array([genres.index(base.genre[item]) for item in base.item_ids])
    all_share = np.zeros((users, len(genres)))
    recent_share = np.zeros((users, len(genres)))
    timelines: dict[str, list[Interaction]] = {}
    for row in sorted(train, key=lambda r: (r.user, r.time, r.item)):
        timelines.setdefault(row.user, []).append(row)
    for position, user in enumerate(base.user_ids):
        rows = timelines[user]
        for row in rows:
            all_share[position, genres.index(base.genre[row.item])] += 1 / len(rows)
        recent = rows[-RECENT_PLAYS:]
        for row in recent:
            recent_share[position, genres.index(base.genre[row.item])] += 1 / len(recent)
    arrays = {"two_tower": tower_table,
              "log_popularity": np.tile(np.log1p(np.asarray(base.popularity, dtype=np.float64)), (users, 1)),
              "genre_match": all_share[:, genre_index],
              "recent_genre_match": recent_share[:, genre_index]}
    for method in ("als", "itemknn", "bpr", "text"):
        arrays[method] = base_table(base, method).astype(np.float64)
    for name, array in arrays.items():
        if array.shape != (users, items):
            raise ValueError(f"feature {name} has shape {array.shape}")
    return arrays


def novel_targets(base: Recommenders, rows: list[Interaction]) -> dict[str, set[str]]:
    """Each user's distinct held-out games not already owned in training."""
    targets: dict[str, set[str]] = {}
    for row in rows:
        if row.item not in base.history.get(row.user, set()):
            targets.setdefault(row.user, set()).add(row.item)
    return targets


def _query_rows(base: Recommenders, arrays: dict[str, np.ndarray], names: tuple[str, ...], user: str):
    position = base.user_index[user]
    candidates = [index for index, item in enumerate(base.item_ids) if item not in base.history[user]]
    return candidates, np.column_stack([arrays[name][position, candidates] for name in names])


def fit_ranker(base: Recommenders, arrays: dict[str, np.ndarray], targets: dict[str, set[str]],
               users: list[str], config: RerankConfig, seed: int) -> lgb.Booster:
    names = config.names()
    matrices, labels, groups = [], [], []
    for user in users:
        candidates, matrix = _query_rows(base, arrays, names, user)
        matrices.append(matrix)
        labels.extend(int(base.item_ids[index] in targets[user]) for index in candidates)
        groups.append(len(candidates))
    if not groups:
        raise ValueError("re-ranker needs at least one query")
    params = {"objective": "lambdarank", "lambdarank_truncation_level": 10, "metric": "none",
              "num_leaves": config.num_leaves, "learning_rate": config.learning_rate,
              "min_data_in_leaf": config.min_data_in_leaf, "seed": seed, "deterministic": True,
              "force_row_wise": True, "num_threads": 1, "verbosity": -1}
    data = lgb.Dataset(np.vstack(matrices), label=labels, group=groups, feature_name=list(names),
                       free_raw_data=False)
    return lgb.train(params, data, num_boost_round=config.rounds)


def predict_table(base: Recommenders, arrays: dict[str, np.ndarray], ranker: lgb.Booster,
                  config: RerankConfig, users: list[str] | None = None,
                  into: np.ndarray | None = None) -> np.ndarray:
    """Fill a (users, items) table with ranker scores. Owned items stay at the floor value."""
    names = config.names()
    table = np.full((len(base.user_ids), len(base.item_ids)), -1e6, dtype=np.float64) if into is None else into
    for user in base.user_ids if users is None else users:
        candidates, matrix = _query_rows(base, arrays, names, user)
        table[base.user_index[user], candidates] = ranker.predict(matrix)
    return table


def out_of_fold_table(base: Recommenders, arrays: dict[str, np.ndarray], targets: dict[str, set[str]],
                      config: RerankConfig, seed: int, folds: int = 5) -> np.ndarray:
    """Score each validation user with a ranker fitted on the other folds' users."""
    users = sorted(targets)
    table = np.full((len(base.user_ids), len(base.item_ids)), -1e6, dtype=np.float64)
    for fold in range(folds):
        held_out = users[fold::folds]
        fitted = [user for user in users if user not in set(held_out)]
        ranker = fit_ranker(base, arrays, targets, fitted, config, seed)
        predict_table(base, arrays, ranker, config, held_out, table)
    return table
