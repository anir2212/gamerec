"""User-level offline ranking metrics with bootstrap intervals.

The slices are ``all``, ``new_user`` (at most three training events),
``cold_item`` (targets with at most five training events), and the complements
added for the M3 comparison: ``warm_user`` and ``warm_item``.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from recstack.data import Interaction
from recstack.models import Recommenders


def _per_user(models: Recommenders, test: list[Interaction], method: str,
              k: int, subset: str = "all") -> tuple[list[dict], set[str]]:
    by_user: dict[str, set[str]] = defaultdict(set)
    counts = models.train_counts()
    for row in test:
        if row.item in models.history.get(row.user, set()):
            continue  # Already owned/repeated items cannot be recommended.
        if subset == "cold_item" and counts[row.item] > 5:
            continue
        if subset == "warm_item" and counts[row.item] <= 5:
            continue
        by_user[row.user].add(row.item)
    if subset == "new_user":
        by_user = {user: relevant for user, relevant in by_user.items()
                   if models.user_interactions[user] <= 3}
    if subset == "warm_user":
        by_user = {user: relevant for user, relevant in by_user.items()
                   if models.user_interactions[user] > 3}
    users, recommended = [], set()
    total_confidence = sum(models.popularity)
    for user, relevant in sorted(by_user.items()):
        if not relevant:
            continue
        ranked = models.recommend(user, method, k)
        recommended.update(ranked)
        hits = [int(item in relevant) for item in ranked]
        count = sum(hits)
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
        running = 0
        average_precision = 0.0
        for index, hit in enumerate(hits, start=1):
            running += hit
            if hit:
                average_precision += running / index
        novelty = sum(-math.log2((models.popularity[models.item_index[item]] + 1) /
                                  (total_confidence + len(models.item_ids)))
                      for item in ranked) / len(ranked) if ranked else 0.0
        users.append({"user": user, "recall": count / len(relevant), "ndcg": dcg / ideal if ideal else 0.0,
                      "map": average_precision / min(len(relevant), k),
                      "hit_rate": float(count > 0), "novelty": novelty})
    return users, recommended


def evaluate(models: Recommenders, test: list[Interaction], method: str, k: int,
             subset: str = "all", bootstrap_samples: int = 400, seed: int = 17) -> dict:
    users, recommended = _per_user(models, test, method, k, subset)
    if not users:
        return {"model": method, "k": k, "subset": subset, "users": 0,
                "metrics": None, "ci95": None}
    keys = ("recall", "ndcg", "map", "hit_rate", "novelty")
    metrics = {name: sum(row[name] for row in users) / len(users) for name in keys}
    metrics["catalog_coverage"] = len(recommended) / len(models.item_ids)
    rng = random.Random(seed)
    distributions = {name: [] for name in keys}
    for _ in range(bootstrap_samples):
        sample = [rng.choice(users) for _ in users]
        for name in keys:
            distributions[name].append(sum(row[name] for row in sample) / len(sample))
    ci95 = {}
    for name, values in distributions.items():
        values.sort()
        ci95[name] = [values[int(0.025 * len(values))],
                      values[min(len(values) - 1, int(0.975 * len(values)))]]
    return {"model": method, "k": k, "subset": subset, "users": len(users),
            "metrics": metrics, "ci95": ci95}


METRIC_KEYS = ("recall", "ndcg", "map", "hit_rate", "novelty")


def _percentile_ci(values: list[float]) -> list[float]:
    values = sorted(values)
    return [values[int(0.025 * len(values))],
            values[min(len(values) - 1, int(0.975 * len(values)))]]


def seed_averaged_rows(models_by_seed: list[Recommenders], test: list[Interaction], method: str,
                       k: int, subset: str = "all") -> tuple[list[dict], list[dict]]:
    """Return one row per user averaged over model seeds, plus one summary per seed.

    Every seed shares a training split, so every seed must score the same users.
    """
    if not models_by_seed:
        raise ValueError("need at least one seeded model")
    per_seed, summaries = [], []
    for models in models_by_seed:
        users, recommended = _per_user(models, test, method, k, subset)
        per_seed.append(users)
        summaries.append({"users": len(users),
                          "catalog_coverage": len(recommended) / len(models.item_ids),
                          **{name: sum(row[name] for row in users) / len(users) if users else 0.0
                             for name in METRIC_KEYS}})
    names = [row["user"] for row in per_seed[0]]
    if any([row["user"] for row in users] != names for users in per_seed):
        raise ValueError("seeds disagree on the evaluated users")
    averaged = [{"user": name,
                 **{key: sum(users[index][key] for users in per_seed) / len(per_seed)
                    for key in METRIC_KEYS}}
                for index, name in enumerate(names)]
    return averaged, summaries


def evaluate_seeds(models_by_seed: list[Recommenders], test: list[Interaction], method: str, k: int,
                   subset: str = "all", bootstrap_samples: int = 400, seed: int = 17) -> dict:
    """Metrics over several model seeds with user-level intervals.

    The point value is the mean over seeds. The interval resamples users after
    averaging each user's metric over seeds, so it reflects user sampling, and
    ``seed_sd_ndcg`` separately reports the spread between training seeds.
    """
    users, summaries = seed_averaged_rows(models_by_seed, test, method, k, subset)
    if not users:
        return {"model": method, "k": k, "subset": subset, "users": 0, "seeds": len(summaries),
                "metrics": None, "ci95": None}
    metrics = {name: sum(row[name] for row in users) / len(users) for name in METRIC_KEYS}
    metrics["catalog_coverage"] = sum(row["catalog_coverage"] for row in summaries) / len(summaries)
    rng = random.Random(seed)
    distributions = {name: [] for name in METRIC_KEYS}
    for _ in range(bootstrap_samples):
        sample = [rng.choice(users) for _ in users]
        for name in METRIC_KEYS:
            distributions[name].append(sum(row[name] for row in sample) / len(sample))
    ndcgs = [row["ndcg"] for row in summaries]
    mean = sum(ndcgs) / len(ndcgs)
    spread = math.sqrt(sum((value - mean) ** 2 for value in ndcgs) / (len(ndcgs) - 1)) if len(ndcgs) > 1 else 0.0
    return {"model": method, "k": k, "subset": subset, "users": len(users), "seeds": len(summaries),
            "metrics": metrics, "ci95": {name: _percentile_ci(values) for name, values in distributions.items()},
            "per_seed_ndcg": ndcgs, "seed_sd_ndcg": spread}


def paired_difference(rows_a: list[dict], rows_b: list[dict], metric: str = "ndcg",
                      bootstrap_samples: int = 2000, seed: int = 17) -> dict:
    """Paired user bootstrap of mean(a - b); both inputs must cover the same users."""
    if [row["user"] for row in rows_a] != [row["user"] for row in rows_b]:
        raise ValueError("paired comparison needs identical users")
    if not rows_a:
        raise ValueError("paired comparison needs at least one user")
    differences = [a[metric] - b[metric] for a, b in zip(rows_a, rows_b, strict=True)]
    rng = random.Random(seed)
    resampled = []
    for _ in range(bootstrap_samples):
        sample = [rng.choice(differences) for _ in differences]
        resampled.append(sum(sample) / len(sample))
    low, high = _percentile_ci(resampled)
    return {"users": len(differences), "metric": metric,
            "mean_difference": sum(differences) / len(differences), "ci95": [low, high],
            "ci_excludes_zero": low > 0 or high < 0,
            "users_a_better": sum(value > 0 for value in differences),
            "users_b_better": sum(value < 0 for value in differences),
            "users_tied": sum(value == 0 for value in differences)}
