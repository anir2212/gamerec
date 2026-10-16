"""User-level offline ranking metrics with bootstrap intervals."""

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
        by_user[row.user].add(row.item)
    if subset == "new_user":
        by_user = {user: relevant for user, relevant in by_user.items()
                   if models.user_interactions[user] <= 3}
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
        users.append({"recall": count / len(relevant), "ndcg": dcg / ideal if ideal else 0.0,
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
