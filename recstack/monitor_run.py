"""Monitor a frozen champion over simulated weeks. All weeks are authored simulations (docs/M7_PROTOCOL.md).

The simulation continues the seed-7 fixture's generator, with and without authored drift. It uses
no real traffic, exposure log or live experiment, and the test split is never read.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from recstack import monitoring as mon
from recstack.data import GENRES, Interaction, Item, clean, synthetic, temporal_split
from recstack.eval import evaluate
from recstack.models import Recommenders
from recstack.registry import Registry
from recstack.score_table import ScoredModels

WEEKS = 8
SIM_SEEDS = (101, 102, 103, 104, 105)
SCENARIOS = ("no_drift", "drift")
ACTIVE_PROBABILITY = 0.8
OWN_GENRE_PROBABILITY = 0.74
TASTE_RATE = 0.1
POPULARITY_RATE = 0.08


def simulate_weeks(catalog: list[Item], user_ids: list[str], scenario: str, seed: int,
                   weeks: int = WEEKS) -> list[list[Interaction]]:
    """Return one list of simulated plays per week, reproducible from the seed."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    rng = random.Random(seed)
    switch_order = list(user_ids)
    rng.shuffle(switch_order)
    switch_rank = {user: rank for rank, user in enumerate(switch_order)}
    count = len(catalog)
    forward = [1 / (1 + index) ** 1.15 for index in range(count)]
    reverse = [1 / (1 + (count - 1 - index)) ** 1.15 for index in range(count)]
    by_genre = {genre: [index for index, item in enumerate(catalog) if item.genre == genre] for genre in GENRES}
    output = []
    for week in range(1, weeks + 1):
        drifting = scenario == "drift"
        moved = min(1.0, TASTE_RATE * week) * len(user_ids) if drifting else 0
        reversed_probability = POPULARITY_RATE * week if drifting else 0.0
        rows = []
        for user in user_ids:
            if rng.random() >= ACTIVE_PROBABILITY:
                continue
            genre_position = int(user[1:]) % len(GENRES)
            if switch_rank[user] < moved:
                genre_position = (genre_position + 1) % len(GENRES)
            for play in range(rng.randint(1, 4)):
                pool = by_genre[GENRES[genre_position]] if rng.random() < OWN_GENRE_PROBABILITY else range(count)
                weights = reverse if rng.random() < reversed_probability else forward
                index = rng.choices(list(pool), weights=[weights[i] for i in pool])[0]
                rows.append(Interaction(user, catalog[index].item, week * 1_000_000 + play, rng.randint(5, 240)))
        output.append(rows)
    return output


def _item_counts(models: Recommenders, rows: list[Interaction]) -> np.ndarray:
    counts = np.zeros(len(models.item_ids))
    for row in rows:
        counts[models.item_index[row.item]] += 1
    return counts


def _played_scores(models: Recommenders, rows: list[Interaction], table: np.ndarray) -> np.ndarray:
    return np.array([table[models.user_index[row.user], models.item_index[row.item]] for row in rows])


def _served_scores(models: ScoredModels, users, table: np.ndarray) -> np.ndarray:
    values = []
    for user in users:
        for item in models.recommend(user, "champion", 10):
            values.append(table[models.user_index[user], models.item_index[item]])
    return np.array(values)


def week_metrics(models: ScoredModels, table: np.ndarray, train: list[Interaction], reference_played: np.ndarray,
                 reference_served: np.ndarray, rows: list[Interaction]) -> dict:
    train_counts = _item_counts(models, train)
    week_counts = _item_counts(models, rows)
    recall = evaluate(models, rows, "champion", 10, bootstrap_samples=400)
    active = sorted({row.user for row in rows})
    return {"events": len(rows), "active_users": len(active), "recall_users": recall["users"],
            "popularity_psi": mon.psi(train_counts, week_counts),
            "popularity_js_bits": mon.js_divergence_bits(train_counts, week_counts),
            "served_score_psi": mon.score_psi(reference_served, _served_scores(models, active, table)),
            "played_score_psi": mon.score_psi(reference_played, _played_scores(models, rows, table)),
            "recall_at_10": recall["metrics"]["recall"], "recall_ci95": recall["ci95"]["recall"]}


def monitor(table: np.ndarray, version: str | None, corpus_seed: int = 7, seeds=SIM_SEEDS, weeks: int = WEEKS,
            users: int = 120, items: int = 80) -> dict:
    rows, catalog = synthetic(corpus_seed, users, items)
    train, validation, _ = temporal_split(clean(rows))
    base = Recommenders(train, catalog, corpus_seed)
    models = ScoredModels(base, {"champion": table})
    reference_played = _played_scores(models, validation, table)
    reference_served = _served_scores(models, base.user_ids, table)
    names = ("popularity_psi", "popularity_js_bits", "served_score_psi", "played_score_psi", "recall_at_10")
    scenarios: dict = {}
    for scenario in SCENARIOS:
        per_seed = defaultdict(list)
        for seed in seeds:
            per_seed[seed] = [week_metrics(models, table, train, reference_played, reference_served, week)
                              for week in simulate_weeks(catalog, base.user_ids, scenario, seed, weeks)]
        scenarios[scenario] = {
            "per_seed": {str(seed): values for seed, values in per_seed.items()},
            "weekly_mean": {name: [mon.mean([per_seed[seed][week][name] for seed in seeds]) for week in range(weeks)]
                            for name in names},
            "weekly_min": {name: [min(per_seed[seed][week][name] for seed in seeds) for week in range(weeks)]
                           for name in names},
            "weekly_max": {name: [max(per_seed[seed][week][name] for seed in seeds) for week in range(weeks)]
                           for name in names}}
    control, drift = scenarios["no_drift"]["weekly_mean"], scenarios["drift"]["weekly_mean"]
    alerts = {name: mon.drift_alert(drift[name], control[name]) for name in names if name != "recall_at_10"}
    alerts["recall_at_10"] = mon.recall_alert(drift["recall_at_10"], control["recall_at_10"])
    control_alerts = {"recall_at_10": mon.recall_alert(control["recall_at_10"], control["recall_at_10"])}
    return {"label": "SIMULATED WEEKS on the seed-7 synthetic fixture; authored drift, no real traffic or exposure log",
            "champion_version": version, "corpus_seed": corpus_seed, "simulation_seeds": list(seeds), "weeks": weeks,
            "test_split_read": False, "scenarios": scenarios,
            "alerts_by_week": alerts, "first_alert_week": {name: mon.first_alert_week(flags) for name, flags in alerts.items()},
            "control_recall_alerts_by_week": control_alerts["recall_at_10"],
            "rule_of_thumb_psi_0_25_weeks_exceeded_in_control": {
                name: sum(value >= 0.25 for value in control[name])
                for name in ("popularity_psi", "served_score_psi", "played_score_psi")},
            "rule_of_thumb_psi_0_25_weeks_exceeded_in_drift": {
                name: sum(value >= 0.25 for value in drift[name])
                for name in ("popularity_psi", "served_score_psi", "played_score_psi")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("runs/registry_demo"))
    parser.add_argument("--output", type=Path, default=Path("evidence/m7_monitoring_simulated.json"))
    args = parser.parse_args()
    registry = Registry(args.registry)
    pointer = registry.champion()
    if pointer is None:
        raise SystemExit(f"no champion in {args.registry}; run `make pipeline-demo` first")
    table, _, _, _ = registry.load(pointer["version"])
    report = monitor(table, pointer["version"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"first_alert_week": report["first_alert_week"],
                      "weekly_mean": {scenario: report["scenarios"][scenario]["weekly_mean"] for scenario in SCENARIOS}},
                     indent=1))


if __name__ == "__main__":
    main()
