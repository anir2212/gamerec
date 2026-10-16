"""Temporal recommendation baselines and experiment calculations."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict


def sample_data(seed=7):
    rng = random.Random(seed)
    genres = {f"g{i}": ("strategy" if i % 3 == 0 else "arcade" if i % 3 == 1 else "puzzle") for i in range(18)}
    rows = []
    for u in range(30):
        preferred = rng.choice(tuple(sorted(set(genres.values()))))
        for t in range(12):
            items = [g for g, genre in genres.items() if genre == preferred] if rng.random() < .8 else list(genres)
            rows.append({"user": f"u{u}", "item": rng.choice(items), "time": t, "minutes": rng.randint(1, 200)})
    return rows, genres


def temporal_split(rows, holdout=2):
    by_user = defaultdict(list)
    for row in rows:
        if row["minutes"] > 0:
            by_user[row["user"]].append(row)
    train, test = [], []
    for user_rows in by_user.values():
        ordered = sorted(user_rows, key=lambda x: x["time"])
        train.extend(ordered[:-holdout])
        test.extend(ordered[-holdout:])
    return train, test


def recommend(user, train, genres, k=5, method="itemknn"):
    by_user = defaultdict(set)
    popularity = Counter()
    for row in train:
        by_user[row["user"]].add(row["item"])
        popularity[row["item"]] += math.log1p(row["minutes"])
    owned = by_user[user]
    scores = defaultdict(float)
    if method == "itemknn" and owned:
        for other in by_user.values():
            common = len(owned & other)
            if common:
                for item in other - owned:
                    scores[item] += common / math.sqrt(len(owned) * len(other))
    preferred = Counter(genres[item] for item in owned).most_common(1)
    for item, pop in popularity.items():
        if item not in owned:
            scores[item] += .001 * pop
    if preferred and method == "itemknn":
        for item, genre in genres.items():
            if item not in owned and genre == preferred[0][0]:
                scores[item] += .05
    return [item for item, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def recall_at_k(train, test, genres, k=5, method="itemknn"):
    users = sorted({r["user"] for r in test})
    scores = []
    for user in users:
        relevant = {r["item"] for r in test if r["user"] == user}
        if relevant:
            scores.append(len(relevant & set(recommend(user, train, genres, k, method))) / len(relevant))
    return sum(scores) / len(scores) if scores else 0


def assign(user_id, salt="experiment-1"):
    return int.from_bytes(hashlib.sha256(f"{salt}:{user_id}".encode()).digest()[:8], "big") % 2


def srm(count_a, count_b, tolerance_z=3.29):
    total = count_a + count_b
    return total > 0 and abs(count_a - total/2) / math.sqrt(total/4) > tolerance_z


def cuped(outcomes, preperiod):
    if len(outcomes) != len(preperiod) or not outcomes:
        raise ValueError("paired nonempty arrays required")
    avg_x = sum(preperiod) / len(preperiod)
    avg_y = sum(outcomes) / len(outcomes)
    var_x = sum((x-avg_x)**2 for x in preperiod)
    theta = sum((x-avg_x)*(y-avg_y) for x, y in zip(preperiod, outcomes)) / var_x if var_x else 0
    return [y - theta*(x-avg_x) for x, y in zip(preperiod, outcomes)]


def sample_size(p0, lift, z_alpha=1.96, z_power=.84):
    p1 = p0 + lift
    if not 0 < p0 < 1 or not 0 < p1 < 1:
        raise ValueError("probabilities must be in (0,1)")
    pooled = (p0+p1)/2
    return math.ceil(((z_alpha*math.sqrt(2*pooled*(1-pooled)) + z_power*math.sqrt(p0*(1-p0)+p1*(1-p1))) / lift)**2)


def analyze_binary(a_success, a_total, b_success, b_total):
    if not 0 <= a_success <= a_total or not 0 <= b_success <= b_total or min(a_total, b_total) == 0:
        raise ValueError("valid positive group sizes required")
    pa, pb = a_success/a_total, b_success/b_total
    delta = pb-pa
    standard_error = math.sqrt(pa*(1-pa)/a_total + pb*(1-pb)/b_total)
    pooled = (a_success+b_success)/(a_total+b_total)
    null_se = math.sqrt(pooled*(1-pooled)*(1/a_total+1/b_total))
    z = delta/null_se if null_se else 0
    return {"control_rate": pa, "treatment_rate": pb, "difference": delta, "ci95": [delta-1.96*standard_error, delta+1.96*standard_error], "p_value": math.erfc(abs(z)/math.sqrt(2)), "srm_flag": srm(a_total,b_total)}


def simulated_experiment(train, test, genres):
    groups = {0: [], 1: []}
    for user in sorted({r["user"] for r in test}):
        arm = assign(user)
        relevant = {r["item"] for r in test if r["user"] == user}
        recommendation = recommend(user, train, genres, method="popularity" if arm == 0 else "itemknn")
        groups[arm].append(int(bool(relevant & set(recommendation))))
    return analyze_binary(sum(groups[0]), len(groups[0]), sum(groups[1]), len(groups[1]))


def main():
    rows, genres = sample_data()
    train, test = temporal_split(rows)
    result = {m: recall_at_k(train, test, genres, method=m) for m in ("popularity", "itemknn")}
    print(json.dumps({"interactions": len(rows), "train": len(train), "test": len(test), "recall_at_5": result, "synthetic_experiment": simulated_experiment(train, test, genres)}, indent=2))


if __name__ == "__main__":
    main()
