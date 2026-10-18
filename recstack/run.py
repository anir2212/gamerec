"""Train local baselines on a seeded synthetic corpus and write measured tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recstack.data import clean, split_stats, synthetic, temporal_split
from recstack.eval import evaluate
from recstack.models import Recommenders


def run(seed: int, output: Path) -> dict:
    rows, catalog = synthetic(seed)
    train, validation, test = temporal_split(clean(rows))
    models = Recommenders(train, catalog, seed)
    methods = ("popularity", "itemknn", "als", "bpr", "content", "text")
    validation_scores = [evaluate(models, validation, method, 10, bootstrap_samples=200)
                         for method in methods]
    # Select on validation, then report untouched test results for every method.
    best = max(validation_scores, key=lambda result: result["metrics"]["ndcg"])["model"]
    test_scores = [evaluate(models, test, method, k, subset)
                   for method in methods for k in (10, 20)
                   for subset in ("all", "new_user", "cold_item")]
    report = {"dataset": "seeded synthetic only", "seed": seed,
              "split": split_stats(train, validation, test, catalog),
              "validation_selection_metric": "ndcg@10", "validation_winner": best,
              "validation": validation_scores, "test": test_scores}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic_baselines.json"))
    args = parser.parse_args()
    result = run(args.seed, args.output)
    print(json.dumps({"split": result["split"], "winner": result["validation_winner"],
                      "test_ndcg10": {row["model"]: row["metrics"]["ndcg"]
                                      for row in result["test"]
                                      if row["k"] == 10 and row["subset"] == "all"}}, indent=2))
