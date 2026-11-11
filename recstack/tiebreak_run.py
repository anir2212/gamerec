"""TB-1 reselection on the seed-7 synthetic corpus, as fixed by docs/TIEBREAK_PROTOCOL.md.

``--phase select`` reads train and validation only and rewrites the M3 model selection
with exact score ties ordered by the protocol's popularity-neutral key instead of lowest
item ID. ``--phase fixture`` reads the same eight methods on the test split, which is a
known, already-inspected fixture and not a held-out estimate.

Nothing here changes ``recstack/models.py`` or ``recstack/deep_run.py`` output, so the
6 November lowest-ID results stay reproducible and keep their own date.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np

from recstack import reranker as rr
from recstack import tiebreak as tb
from recstack import two_tower as tt
from recstack.contract import rows_digest
from recstack.deep_run import BASELINES, NEURAL_SEEDS, RERANK_GRID, TOWER_GRID, load_splits, tower_tables
from recstack.eval import evaluate, evaluate_seeds, paired_difference, seed_averaged_rows
from recstack.models import Recommenders
from recstack.score_table import ScoredModels, base_table

PROTOCOL = Path("docs/TIEBREAK_PROTOCOL.md")
SLICES = ("all", "cold_item", "warm_item", "new_user", "warm_user")
RULE = ("exact float score ties ordered by blake2b(salt|permutation|item_id); "
        "metric is the mean over the 20 declared permutations")


def protocol_record() -> dict:
    """Tie the output to the exact protocol text and the commit that first added it."""
    if not PROTOCOL.exists():
        return {"path": str(PROTOCOL), "sha256": None, "first_commit": None}
    first = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", str(PROTOCOL)],
                           capture_output=True, text=True, check=False).stdout.strip()
    return {"path": str(PROTOCOL), "sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "first_commit": first or None, "rule": RULE, "salt": tb.SALT.decode(),
            "permutations": list(tb.PERMUTATIONS)}


def assert_protocol_committed() -> None:
    """Refuse a reported seed-7 run unless the protocol is committed and unedited."""
    record = protocol_record()
    edited = subprocess.run(["git", "status", "--porcelain", "--", str(PROTOCOL)],
                            capture_output=True, text=True, check=False).stdout.strip()
    if record["first_commit"] is None or edited:
        raise SystemExit(f"{PROTOCOL} must be committed and unedited before a scored TB-1 run")


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarise(result: dict, model_seeds: int, lowest_id_ndcg: float) -> dict:
    """Point estimate, user interval and the two separate spreads the protocol requires."""
    permutations = len(tb.PERMUTATIONS)
    values = result["per_seed_ndcg"]
    grid = [values[index * permutations:(index + 1) * permutations] for index in range(model_seeds)]
    seed_means = [sum(row) / len(row) for row in grid]
    permutation_means = [sum(row[column] for row in grid) / len(grid) for column in range(permutations)]
    return {"users": result["users"], "ndcg": result["metrics"]["ndcg"], "ci95": result["ci95"]["ndcg"],
            "model_seeds": model_seeds, "tie_permutations": permutations,
            "sd_between_model_seeds": _sd(seed_means), "sd_between_tie_permutations": _sd(permutation_means),
            "ndcg_min_over_runs": min(values), "ndcg_max_over_runs": max(values),
            "ndcg_lowest_id_rule": lowest_id_ndcg,
            "novelty": result["metrics"]["novelty"], "catalog_coverage": result["metrics"]["catalog_coverage"]}


def baseline_rows(base: Recommenders, rows, method: str, k: int = 10, subset: str = "all"):
    """Per-user expectations for a baseline: no score table, so only the tie order changes."""
    return seed_averaged_rows(tb.wrap(base), rows, method, k, subset)


def select(train, validation, catalog, corpus_seed: int, model_seed: int = 7) -> dict:
    """Reselect from validation under TB-1. This function never receives the test split."""
    base = Recommenders(train, catalog, model_seed)
    features = tt.Features(train, catalog)
    baselines = {}
    for method in BASELINES:
        result = evaluate_seeds(tb.wrap(base), validation, method, 10)
        old = evaluate(base, validation, method, 10, bootstrap_samples=1)["metrics"]["ndcg"]
        baselines[method] = summarise(result, 1, old)
    tower_results = []
    for config in TOWER_GRID:
        tables = tower_tables(features, config)
        result = evaluate_seeds(tb.wrap_tables(base, "two_tower", tables), validation, "two_tower", 10)
        old = evaluate_seeds([ScoredModels(base, {"two_tower": table}) for table in tables],
                             validation, "two_tower", 10, bootstrap_samples=1)["metrics"]["ndcg"]
        tower_results.append({"config": config.to_dict(), **summarise(result, len(tables), old)})
    # max keeps the first of equal values, and the grid runs from the smallest setting up.
    best_tower = max(range(len(tower_results)), key=lambda index: tower_results[index]["ndcg"])
    tables = tower_tables(features, TOWER_GRID[best_tower])
    arrays_by_seed = [rr.feature_arrays(base, table, train) for table in tables]
    targets = rr.novel_targets(base, validation)
    rerank_results = []
    for config in RERANK_GRID:
        folds = [rr.out_of_fold_table(base, arrays, targets, config, seed)
                 for seed, arrays in zip(NEURAL_SEEDS, arrays_by_seed, strict=True)]
        result = evaluate_seeds(tb.wrap_tables(base, "reranker", folds), validation, "reranker", 10)
        old = evaluate_seeds([ScoredModels(base, {"reranker": table}) for table in folds],
                             validation, "reranker", 10, bootstrap_samples=1)["metrics"]["ndcg"]
        rerank_results.append({"config": config.to_dict(), **summarise(result, len(folds), old)})
    best_rerank = max(range(len(rerank_results)), key=lambda index: rerank_results[index]["ndcg"])
    leaderboard = {method: row["ndcg"] for method, row in baselines.items()}
    leaderboard["two_tower"] = tower_results[best_tower]["ndcg"]
    leaderboard["reranker"] = rerank_results[best_rerank]["ndcg"]
    old_leaderboard = {method: row["ndcg_lowest_id_rule"] for method, row in baselines.items()}
    old_leaderboard["two_tower"] = tower_results[best_tower]["ndcg_lowest_id_rule"]
    old_leaderboard["reranker"] = rerank_results[best_rerank]["ndcg_lowest_id_rule"]
    # Exact ties across methods are broken alphabetically, as the protocol declares.
    winner = min(sorted(leaderboard), key=lambda method: (-leaderboard[method], method))
    winner_tables = {"two_tower": tables,
                     "reranker": [rr.out_of_fold_table(base, arrays, targets, RERANK_GRID[best_rerank], seed)
                                  for seed, arrays in zip(NEURAL_SEEDS, arrays_by_seed, strict=True)]}
    paired = winner_vs_text(base, validation, winner, winner_tables)
    return {"data": "seeded synthetic only, no real record used", "corpus_seed": corpus_seed,
            "model_seed": model_seed, "neural_seeds": list(NEURAL_SEEDS), "protocol": protocol_record(),
            "split_sha256": {"train": rows_digest(train), "validation": rows_digest(validation)},
            "test_split_read_by_selection": False,
            "selection_metric": "expected validation ndcg@10 under TB-1, mean over tie permutations and model seeds",
            "baselines_validation": baselines, "two_tower_grid": tower_results,
            "two_tower_selected": TOWER_GRID[best_tower].to_dict(), "reranker_grid": rerank_results,
            "reranker_selected": RERANK_GRID[best_rerank].to_dict(),
            "validation_leaderboard_tb1": dict(sorted(leaderboard.items(), key=lambda pair: -pair[1])),
            "validation_leaderboard_lowest_id_recomputed": dict(sorted(old_leaderboard.items(),
                                                                       key=lambda pair: -pair[1])),
            "validation_winner_tb1": winner,
            "validation_winner_lowest_id_recomputed": min(sorted(old_leaderboard),
                                                          key=lambda m: (-old_leaderboard[m], m)),
            "winner_minus_text_validation_ndcg10_descriptive": paired,
            "reranker_note": "the re-ranker fits validation labels; its validation row is out of fold",
            "popularity_neutrality": neutrality(base)}


def winner_vs_text(base: Recommenders, validation, winner: str, winner_tables: dict) -> dict:
    """Descriptive paired bootstrap of winner minus text on the split that selected the winner.

    Computed under both rules. It is optimistic by construction and is not a hypothesis test.
    """
    if winner == "text":
        return {"tb1": None, "lowest_id": None,
                "note": "the TB-1 winner is text itself, so there is no difference to report"}
    paired = {}
    for rule, wrapped in (("tb1", True), ("lowest_id", False)):
        rows = {}
        for method in (winner, "text"):
            if method in BASELINES:
                rows[method] = (baseline_rows(base, validation, method)[0] if wrapped
                                else seed_averaged_rows([base], validation, method, 10)[0])
            else:
                models = (tb.wrap_tables(base, method, winner_tables[method]) if wrapped
                          else [ScoredModels(base, {method: table}) for table in winner_tables[method]])
                rows[method] = seed_averaged_rows(models, validation, method, 10)[0]
        paired[rule] = {"a": winner, "b": "text", "subset": "all",
                        **paired_difference(rows[winner], rows["text"])}
    return paired


def neutrality(base: Recommenders) -> dict:
    """Evidence that the declared key is unrelated to popularity or to item ID order."""
    popularity = np.asarray(base.popularity, dtype=float)
    index = np.arange(len(base.item_ids), dtype=float)
    correlations = []
    for seed in tb.PERMUTATIONS:
        ranks = np.argsort(np.argsort(np.asarray(tb.order_key(base.item_ids, seed), dtype=object))).astype(float)
        correlations.append({"permutation": seed,
                             "spearman_key_vs_popularity": float(np.corrcoef(
                                 ranks, np.argsort(np.argsort(popularity)).astype(float))[0, 1]),
                             "spearman_key_vs_item_index": float(np.corrcoef(
                                 ranks, np.argsort(np.argsort(index)).astype(float))[0, 1])})
    return {"per_permutation": correlations,
            "mean_abs_spearman_key_vs_popularity": float(np.mean(
                [abs(row["spearman_key_vs_popularity"]) for row in correlations])),
            "mean_abs_spearman_key_vs_item_index": float(np.mean(
                [abs(row["spearman_key_vs_item_index"]) for row in correlations]))}


def single_digit_share(models: list, base: Recommenders, method: str, k: int = 10) -> float:
    """Share of each top-k list held by the generator's popular single-digit item IDs."""
    single = {item for item in base.item_ids if int(item[1:]) < 10}
    hits = [sum(item in single for item in model.recommend(user, method, k)) / k
            for model in models for user in base.user_ids]
    return sum(hits) / len(hits)


def fixture(train, validation, test, catalog, selection: dict, corpus_seed: int, model_seed: int = 7) -> dict:
    """Read all eight methods on the test split under TB-1. Test is a known fixture, not a holdout."""
    base = Recommenders(train, catalog, model_seed)
    features = tt.Features(train, catalog)
    tower_config = tt.TowerConfig(**selection["two_tower_selected"])
    rerank_config = rr.RerankConfig(**selection["reranker_selected"])
    tables = tower_tables(features, tower_config)
    arrays_by_seed = [rr.feature_arrays(base, table, train) for table in tables]
    targets = rr.novel_targets(base, validation)
    rankers = [rr.fit_ranker(base, arrays, targets, sorted(targets), rerank_config, seed)
               for seed, arrays in zip(NEURAL_SEEDS, arrays_by_seed, strict=True)]
    neural = {"two_tower": tables,
              "reranker": [rr.predict_table(base, arrays, ranker, rerank_config)
                           for arrays, ranker in zip(arrays_by_seed, rankers, strict=True)]}
    rows = []
    for method in BASELINES:
        for k, subset in itertools.product((10, 20), SLICES):
            result = evaluate_seeds(tb.wrap(base), test, method, k, subset)
            old = evaluate(base, test, method, k, subset, bootstrap_samples=1)
            rows.append({"model": method, "k": k, "subset": subset, "users": result["users"],
                         "metrics": result["metrics"], "ci95": result["ci95"],
                         "ndcg_lowest_id_rule": None if old["metrics"] is None else old["metrics"]["ndcg"]})
    for method, seed_tables in neural.items():
        for k, subset in itertools.product((10, 20), SLICES):
            result = evaluate_seeds(tb.wrap_tables(base, method, seed_tables), test, method, k, subset)
            old = evaluate_seeds([ScoredModels(base, {method: table}) for table in seed_tables],
                                 test, method, k, subset, bootstrap_samples=1)
            rows.append({"model": method, "k": k, "subset": subset, "users": result["users"],
                         "metrics": result["metrics"], "ci95": result["ci95"],
                         "ndcg_lowest_id_rule": None if old["metrics"] is None else old["metrics"]["ndcg"]})
    share = {method: single_digit_share(tb.wrap(base), base, method) for method in ("text", "content")}
    for method, seed_tables in neural.items():
        share[method] = single_digit_share(tb.wrap(base, {method: seed_tables[0]}), base, method)
    return {"data": "seeded synthetic only, no real record used", "corpus_seed": corpus_seed,
            "model_seed": model_seed, "neural_seeds": list(NEURAL_SEEDS), "protocol": protocol_record(),
            "test_status": ("KNOWN FIXTURE, ALREADY INSPECTED: these 97 test users were scored and read on "
                            "6 November 2026, before this protocol existed. Not a held-out estimate."),
            "selection_protocol_sha256": selection["protocol"]["sha256"],
            "validation_winner_tb1": selection["validation_winner_tb1"],
            "split_sha256": {"train": rows_digest(train), "validation": rows_digest(validation),
                             "test": rows_digest(test)},
            "test_rows": rows, "top10_share_of_single_digit_item_ids_tb1": share}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "fixture"), required=True)
    parser.add_argument("--corpus-seed", type=int, default=7)
    parser.add_argument("--selection", type=Path, default=Path("evidence/m3_tiebreak_reselection.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/m3_tiebreak_test_fixture.json"))
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
        print(json.dumps({"winner_tb1": result["validation_winner_tb1"],
                          "leaderboard_tb1": {name: round(value, 4) for name, value
                                              in result["validation_leaderboard_tb1"].items()},
                          "two_tower": result["two_tower_selected"],
                          "reranker": result["reranker_selected"]}, indent=2))
        return
    selection = json.loads(args.selection.read_text())
    if selection["corpus_seed"] != args.corpus_seed or selection["split_sha256"]["validation"] != rows_digest(validation):
        raise SystemExit("selection file does not match this corpus")
    if args.corpus_seed == 7 and selection["protocol"]["sha256"] != protocol_record()["sha256"]:
        raise SystemExit("the protocol changed after selection; do not reuse that selection")
    result = fixture(train, validation, test, catalog, selection, args.corpus_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({row["model"]: round(row["metrics"]["ndcg"], 4) for row in result["test_rows"]
                      if row["k"] == 10 and row["subset"] == "all"}, indent=2))


if __name__ == "__main__":
    main()
