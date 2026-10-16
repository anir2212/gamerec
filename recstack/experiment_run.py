"""Run seeded statistical diagnostics and a declared offline click-proxy experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recstack.data import clean, synthetic, temporal_split
from recstack.experiment import assign, binary_result, cuped, peeking_false_positive, sample_size_per_arm
from recstack.models import Recommenders


def run(output: Path = Path("runs/experiment_baseline.json")) -> dict:
    rows, catalog = synthetic(seed=7)
    train, _, test = temporal_split(clean(rows))
    models = Recommenders(train, catalog, seed=7)
    relevant = {}
    for row in test:
        if row.item not in models.history[row.user]:
            relevant.setdefault(row.user, set()).add(row.item)
    groups: dict[str, list[int]] = {"A": [], "B": []}
    for user, items in sorted(relevant.items()):
        arm = assign(user, "gamerec-offline-proxy-v1")
        method = "als" if arm == "A" else "content"
        # A hit among held-out games is a proxy; no actual impression or click occurred.
        groups[arm].append(int(bool(items.intersection(models.recommend(user, method, 10)))))
    proxy = binary_result(groups["A"], groups["B"], bootstrap=1000)
    rng = np.random.default_rng(7)
    pre = rng.normal(size=2000)
    outcomes = 0.75 * pre + rng.normal(size=2000)
    adjusted, theta = cuped(outcomes.tolist(), pre.tolist())
    variance_reduction = 1 - float(np.var(adjusted) / np.var(outcomes))
    report = {"simulated_offline_hit_proxy": proxy,
              "power_example": {"baseline": 0.2, "absolute_lift": 0.01,
                                "sample_size_per_arm": sample_size_per_arm(0.2, 0.01)},
              "cuped_correlated_fixture": {"n": 2000, "theta": theta,
                                            "variance_reduction": variance_reduction},
              "peeking_null_simulation": peeking_false_positive()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
