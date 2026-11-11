"""M3 comparison on the seed-7 synthetic corpus, as fixed by docs/M3_PROTOCOL.md.

``--phase select`` reads validation only and writes the chosen settings.
``--phase compare`` refits those settings and reports validation and test next to the
six baselines. The test split is a known fixture: its baseline numbers had already been
read before this comparison was designed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from recstack import reranker as rr
from recstack import two_tower as tt
from recstack.contract import SYNTHETIC_CONTRACT, rows_digest, validate_records
from recstack.data import clean, synthetic, temporal_split, to_records
from recstack.eval import evaluate, evaluate_seeds, paired_difference, seed_averaged_rows
from recstack.models import Recommenders
from recstack.score_table import ScoredModels, base_table

BASELINES = ("popularity", "itemknn", "als", "bpr", "content", "text")
NEURAL_SEEDS = (11, 23, 37, 41, 53)
TOWER_GRID = [tt.TowerConfig(dim=dim, epochs=epochs, logq=logq)
              for dim, epochs, logq in itertools.product((16, 32), (5, 10, 20), (False, True))]
RERANK_GRID = [rr.RerankConfig(num_leaves=leaves, rounds=rounds, features=features)
               for features, leaves, rounds in itertools.product(("plan", "plus"), (7, 15), (50, 150))]
SLICES = ("all", "cold_item", "warm_item", "new_user", "warm_user")
PROTOCOL = Path("docs/M3_PROTOCOL.md")
# (name, model A, model B, slice): every paired test comparison the protocol names.
PAIRED = (("two_tower_vs_text_overall", "two_tower", "text", "all"),
          ("reranker_vs_text_overall", "reranker", "text", "all"),
          ("reranker_vs_two_tower_overall", "reranker", "two_tower", "all"),
          ("two_tower_vs_itemknn_cold_item", "two_tower", "itemknn", "cold_item"),
          ("content_only_vs_id_only_cold_item", "two_tower_content_only", "two_tower_id_only", "cold_item"),
          ("hybrid_vs_id_only_cold_item", "two_tower", "two_tower_id_only", "cold_item"),
          ("id_only_vs_content_only_warm_item", "two_tower_id_only", "two_tower_content_only", "warm_item"),
          ("itemknn_vs_content_warm_user", "itemknn", "content", "warm_user"))


def protocol_record() -> dict:
    """Tie the output to the exact protocol text and the commit that first added it."""
    if not PROTOCOL.exists():
        return {"path": str(PROTOCOL), "sha256": None, "first_commit": None}
    first = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", str(PROTOCOL)],
                           capture_output=True, text=True, check=False).stdout.strip()
    return {"path": str(PROTOCOL), "sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "first_commit": first or None}


def assert_protocol_committed() -> None:
    """Refuse a reported seed-7 test comparison unless the protocol is committed and unedited."""
    record = protocol_record()
    edited = subprocess.run(["git", "status", "--porcelain", "--", str(PROTOCOL)],
                            capture_output=True, text=True, check=False).stdout.strip()
    if record["first_commit"] is None or edited:
        raise SystemExit(f"{PROTOCOL} must be committed and unedited before scoring seed-7 test")


def load_splits(corpus_seed: int):
    rows, catalog = synthetic(corpus_seed)
    rows = validate_records(to_records(rows), SYNTHETIC_CONTRACT, temporal_claim=True, catalog=catalog)
    train, validation, test = temporal_split(clean(rows))
    return train, validation, test, catalog


def tower_tables(features: tt.Features, config: tt.TowerConfig, seeds=NEURAL_SEEDS) -> list[np.ndarray]:
    return [tt.score_table(tt.fit(features, config, seed), features) for seed in seeds]


def select(train, validation, catalog, corpus_seed: int, model_seed: int = 7) -> dict:
    """Choose settings from validation. This function never receives the test split."""
    base = Recommenders(train, catalog, model_seed)
    features = tt.Features(train, catalog)
    baselines = {method: evaluate(base, validation, method, 10, bootstrap_samples=400) for method in BASELINES}
    tower_results = []
    for config in TOWER_GRID:
        tables = tower_tables(features, config)
        scored = [ScoredModels(base, {"two_tower": table}) for table in tables]
        result = evaluate_seeds(scored, validation, "two_tower", 10)
        tower_results.append({"config": config.to_dict(), "ndcg": result["metrics"]["ndcg"],
                              "ci95": result["ci95"]["ndcg"], "per_seed_ndcg": result["per_seed_ndcg"],
                              "seed_sd_ndcg": result["seed_sd_ndcg"], "novelty": result["metrics"]["novelty"],
                              "catalog_coverage": result["metrics"]["catalog_coverage"]})
    # max keeps the first of equal values, and the grid runs from the smallest setting up.
    best_tower = max(range(len(tower_results)), key=lambda index: tower_results[index]["ndcg"])
    tables = tower_tables(features, TOWER_GRID[best_tower])
    arrays_by_seed = [rr.feature_arrays(base, table, train) for table in tables]
    targets = rr.novel_targets(base, validation)
    rerank_results = []
    for config in RERANK_GRID:
        scored = [ScoredModels(base, {"reranker": rr.out_of_fold_table(base, arrays, targets, config, seed)})
                  for seed, arrays in zip(NEURAL_SEEDS, arrays_by_seed, strict=True)]
        result = evaluate_seeds(scored, validation, "reranker", 10)
        rerank_results.append({"config": config.to_dict(), "out_of_fold_ndcg": result["metrics"]["ndcg"],
                               "ci95": result["ci95"]["ndcg"], "per_seed_ndcg": result["per_seed_ndcg"],
                               "seed_sd_ndcg": result["seed_sd_ndcg"]})
    best_rerank = max(range(len(rerank_results)), key=lambda index: rerank_results[index]["out_of_fold_ndcg"])
    leaderboard = {method: result["metrics"]["ndcg"] for method, result in baselines.items()}
    leaderboard["two_tower"] = tower_results[best_tower]["ndcg"]
    leaderboard["reranker"] = rerank_results[best_rerank]["out_of_fold_ndcg"]
    return {"data": "seeded synthetic only", "corpus_seed": corpus_seed, "model_seed": model_seed,
            "neural_seeds": list(NEURAL_SEEDS), "protocol": protocol_record(),
            "split_sha256": {"train": rows_digest(train), "validation": rows_digest(validation)},
            "test_split_read": False, "selection_metric": "validation ndcg@10, mean over neural seeds",
            "baselines_validation": {method: {"users": result["users"], "ndcg": result["metrics"]["ndcg"],
                                              "ci95": result["ci95"]["ndcg"]}
                                     for method, result in baselines.items()},
            "two_tower_grid": tower_results, "two_tower_selected": TOWER_GRID[best_tower].to_dict(),
            "reranker_grid": rerank_results, "reranker_selected": RERANK_GRID[best_rerank].to_dict(),
            "validation_leaderboard": dict(sorted(leaderboard.items(), key=lambda pair: -pair[1])),
            "validation_winner": max(leaderboard, key=leaderboard.get)}


def tie_break_diagnostic(base: Recommenders, method_table: np.ndarray, test, shuffles: int = 20) -> dict:
    """Recompute NDCG@10 when exact score ties are broken at random rather than by lowest item ID.

    Low IDs are the generator's popular items, so the ID tie-break can act as a hidden popularity prior.
    """
    gaps = np.abs(np.diff(np.sort(method_table, axis=1), axis=1))
    # Smallest noise that cannot reorder distinct scores.
    positive = gaps[gaps > 1e-12]
    scale = (positive.min() if positive.size else 1.0) * 0.25
    results = {}
    for subset in ("all", "cold_item"):
        original = evaluate(ScoredModels(base, {"m": method_table}), test, "m", 10, subset, bootstrap_samples=1)
        values = []
        for shuffle in range(shuffles):
            noise = np.random.default_rng(shuffle).uniform(0, scale, method_table.shape)
            scored = ScoredModels(base, {"m": method_table + noise})
            values.append(evaluate(scored, test, "m", 10, subset, bootstrap_samples=1)["metrics"]["ndcg"])
        results[subset] = {"users": original["users"], "ndcg_id_tie_break": original["metrics"]["ndcg"],
                           "ndcg_random_tie_break_mean": sum(values) / len(values),
                           "ndcg_random_tie_break_min": min(values), "ndcg_random_tie_break_max": max(values)}
    ties = sum(int(np.sum(np.isclose(np.diff(np.sort(row)), 0.0, atol=1e-12))) for row in method_table)
    results["adjacent_tied_score_pairs"] = ties
    results["adjacent_score_pairs"] = int(method_table.shape[0] * (method_table.shape[1] - 1))
    return results


def compare(train, validation, test, catalog, selection: dict, corpus_seed: int, model_seed: int = 7) -> dict:
    """Refit the validation-selected settings and report them beside the baselines on validation and test."""
    base = Recommenders(train, catalog, model_seed)
    features = tt.Features(train, catalog)
    tower_config = tt.TowerConfig(**selection["two_tower_selected"])
    rerank_config = rr.RerankConfig(**selection["reranker_selected"])
    variants = {"two_tower": tower_config,
                "two_tower_id_only": tt.TowerConfig(**{**tower_config.to_dict(), "variant": "id_only"}),
                "two_tower_content_only": tt.TowerConfig(**{**tower_config.to_dict(), "variant": "content_only"})}
    seed_tables = {name: tower_tables(features, config) for name, config in variants.items()}
    arrays_by_seed = [rr.feature_arrays(base, table, train) for table in seed_tables["two_tower"]]
    targets = rr.novel_targets(base, validation)
    rankers = [rr.fit_ranker(base, arrays, targets, sorted(targets), rerank_config, seed)
               for seed, arrays in zip(NEURAL_SEEDS, arrays_by_seed, strict=True)]
    seed_tables["reranker"] = [rr.predict_table(base, arrays, ranker, rerank_config)
                               for arrays, ranker in zip(arrays_by_seed, rankers, strict=True)]
    # Validation labels fitted these rankers, so the reranker's validation row is out-of-fold.
    out_of_fold = [rr.out_of_fold_table(base, arrays, targets, rerank_config, seed)
                   for seed, arrays in zip(NEURAL_SEEDS, arrays_by_seed, strict=True)]
    models = {name: [ScoredModels(base, {name: table}) for table in tables] for name, tables in seed_tables.items()}
    models_validation = dict(models)
    models_validation["reranker"] = [ScoredModels(base, {"reranker": table}) for table in out_of_fold]
    for method in BASELINES:
        models[method] = [base]
        models_validation[method] = [base]

    def report(rows, registry) -> list[dict]:
        output = []
        for name, scored in registry.items():
            for k, subset in itertools.product((10, 20), SLICES):
                if name in BASELINES:
                    output.append({**evaluate(base, rows, name, k, subset), "seeds": 1})
                else:
                    output.append(evaluate_seeds(scored, rows, name, k, subset))
        return output

    paired = {}
    for name, left, right, subset in PAIRED:
        rows_left, _ = seed_averaged_rows(models[left], test, left, 10, subset)
        rows_right, _ = seed_averaged_rows(models[right], test, right, 10, subset)
        paired[name] = {"a": left, "b": right, "subset": subset, **paired_difference(rows_left, rows_right)}
    importance = {}
    names = rerank_config.names()
    gains = np.mean([ranker.feature_importance("gain") for ranker in rankers], axis=0)
    total = float(gains.sum())
    for name, gain in zip(names, gains, strict=True):
        importance[name] = float(gain) / total if total else 0.0
    single_digit = [index for index, item in enumerate(base.item_ids) if int(item[1:]) < 10]
    top_share = {}
    for name in ("text", "content", "two_tower", "reranker"):
        hits = []
        scored = models[name][0]
        for user in base.user_ids:
            ranked = scored.recommend(user, name, 10)
            hits.append(sum(base.item_index[item] in single_digit for item in ranked) / max(1, len(ranked)))
        top_share[name] = sum(hits) / len(hits)
    return {"data": "seeded synthetic only", "corpus_seed": corpus_seed, "model_seed": model_seed,
            "neural_seeds": list(NEURAL_SEEDS), "protocol": protocol_record(),
            "test_status": "known fixture: baseline test metrics were already read before this comparison",
            "split_sha256": {"train": rows_digest(train), "validation": rows_digest(validation),
                             "test": rows_digest(test)},
            "two_tower_config": tower_config.to_dict(), "reranker_config": rerank_config.to_dict(),
            "validation_winner_from_selection": selection["validation_winner"],
            "validation": report(validation, models_validation), "test": report(test, models),
            "paired_test_ndcg10": paired, "reranker_feature_gain_share": importance,
            "tie_break_diagnostic_test": {
                "text": tie_break_diagnostic(base, base_table(base, "text"), test),
                "content": tie_break_diagnostic(base, base_table(base, "content"), test),
                "two_tower_content_only": tie_break_diagnostic(base, seed_tables["two_tower_content_only"][0], test)},
            # Added after the first compare run; it does not feed selection or any other result.
            "tie_break_diagnostic_validation_post_hoc": {
                "text": tie_break_diagnostic(base, base_table(base, "text"), validation),
                "content": tie_break_diagnostic(base, base_table(base, "content"), validation),
                "two_tower_content_only": tie_break_diagnostic(base, seed_tables["two_tower_content_only"][0], validation),
                "two_tower": tie_break_diagnostic(base, seed_tables["two_tower"][0], validation)},
            "top10_share_of_single_digit_item_ids": top_share, "training_examples": len(features.training_examples()["target"]),
            "cold_items": int(features.cold.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "compare"), required=True)
    parser.add_argument("--corpus-seed", type=int, default=7)
    parser.add_argument("--selection", type=Path, default=Path("evidence/m3_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/m3_test_comparison.json"))
    args = parser.parse_args()
    random.seed(0)
    if args.corpus_seed == 7:
        assert_protocol_committed()
    train, validation, test, catalog = load_splits(args.corpus_seed)
    if args.phase == "select":
        del test
        result = select(train, validation, catalog, args.corpus_seed)
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        args.selection.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"two_tower": result["two_tower_selected"], "reranker": result["reranker_selected"],
                          "leaderboard": result["validation_leaderboard"]}, indent=2))
        return
    selection = json.loads(args.selection.read_text())
    if selection["corpus_seed"] != args.corpus_seed or selection["split_sha256"]["validation"] != rows_digest(validation):
        raise SystemExit("selection file does not match this corpus")
    if args.corpus_seed == 7 and selection["protocol"]["sha256"] != protocol_record()["sha256"]:
        raise SystemExit("the protocol changed after selection; do not reuse that selection")
    result = compare(train, validation, test, catalog, selection, args.corpus_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({row["model"]: round(row["metrics"]["ndcg"], 4) for row in result["test"]
                      if row["k"] == 10 and row["subset"] == "all"}, indent=2))


if __name__ == "__main__":
    main()
