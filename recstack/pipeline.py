"""Local retraining pipeline with a promotion gate (seed-7 synthetic corpus, no deployed service).

Steps: extract -> features -> train -> evaluate -> gate -> register -> deploy.
Each step reads and writes files in a work directory, so one can run alone (the Airflow DAG file
does this) or all can run in order. A blocked gate leaves the registry's champion untouched;
as a CLI step it exits with status 3, which stops downstream tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

from recstack import gate as gate_rules
from recstack.contract import SYNTHETIC_CONTRACT, rows_digest, validate_records
from recstack.data import Interaction, Item, clean, synthetic, temporal_split, to_records
from recstack.eval import evaluate
from recstack.models import Recommenders
from recstack.registry import Registry
from recstack.score_table import ScoredModels, base_table

STEPS = ("extract", "features", "train", "evaluate", "gate", "register", "deploy")
BLOCKED_EXIT_CODE = 3
FAULTS = ("none", "degrade_candidate", "evaluation_error")


class EvaluationFailure(RuntimeError):
    """The evaluation step could not produce a trustworthy metric."""


@dataclass(frozen=True)
class PipelineConfig:
    work_dir: Path
    registry_dir: Path
    corpus_seed: int = 7
    users: int = 120
    items: int = 80
    model_seed: int = 11
    # M3 validation-selected settings; see evidence/m3_selection.json.
    tower: dict = field(default_factory=lambda: {"dim": 32, "epochs": 5, "logq": True})
    tolerance: float = gate_rules.DEFAULT_TOLERANCE
    fault: str = "none"
    reload_url: str | None = None


def _read(config: PipelineConfig, name: str) -> dict:
    return json.loads((config.work_dir / f"{name}.json").read_text())


def _write(config: PipelineConfig, name: str, payload: dict) -> dict:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    (config.work_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _splits(config: PipelineConfig):
    payload = _read(config, "extract")
    catalog = [Item(**row) for row in payload["catalog"]]
    return ([Interaction(**row) for row in payload["train"]],
            [Interaction(**row) for row in payload["validation"]], catalog)


def extract(config: PipelineConfig) -> dict:
    """Generate and validate rows. The test split is dropped here, so later steps cannot read it."""
    rows, catalog = synthetic(config.corpus_seed, config.users, config.items)
    rows = validate_records(to_records(rows), SYNTHETIC_CONTRACT, temporal_claim=True, catalog=catalog)
    train, validation, _ = temporal_split(clean(rows))
    return _write(config, "extract", {
        "data": "seeded synthetic only", "corpus_seed": config.corpus_seed,
        "train": to_records(train), "validation": to_records(validation),
        "catalog": [{"item": item.item, "genre": item.genre, "description": item.description} for item in catalog],
        "split_sha256": {"train": rows_digest(train), "validation": rows_digest(validation)}})


def features(config: PipelineConfig) -> dict:
    from recstack.two_tower import Features  # PyTorch is only needed by the training steps.

    train, _, catalog = _splits(config)
    built = Features(train, catalog)
    np.savez(config.work_dir / "features.npz", text=built.text, counts=built.counts, cold=built.cold)
    return _write(config, "features", {"items": built.n_items, "text_columns": int(built.text.shape[1]),
                                       "cold_items": int(built.cold.sum()),
                                       "training_examples": int(len(built.training_examples()["target"]))})


def train_models(config: PipelineConfig) -> dict:
    from recstack import two_tower as tt

    train, _, catalog = _splits(config)
    with np.load(config.work_dir / "features.npz", allow_pickle=False) as data:
        built = tt.Features(train, catalog, text=data["text"])
    base = Recommenders(train, catalog, config.corpus_seed)
    tower_config = tt.TowerConfig(**config.tower)
    tower = tt.score_table(tt.fit(built, tower_config, config.model_seed), built)
    np.savez(config.work_dir / "tables.npz", als=base_table(base, "als"), two_tower=tower,
             user_ids=np.array(base.user_ids), item_ids=np.array(base.item_ids))
    return _write(config, "train", {"models": ["als", "two_tower"], "model_seed": config.model_seed,
                                    "two_tower_config": tower_config.to_dict()})


def run_evaluation(config: PipelineConfig) -> dict:
    """Score both models on validation and pick the better as the candidate."""
    # An earlier run's files must not survive a failure here and reach the gate or registration.
    for name in ("evaluation.json", "gate.json", "candidate.npz"):
        (config.work_dir / name).unlink(missing_ok=True)
    if config.fault == "evaluation_error":
        raise EvaluationFailure("injected evaluation failure")
    train, validation, catalog = _splits(config)
    base = Recommenders(train, catalog, config.corpus_seed)
    with np.load(config.work_dir / "tables.npz", allow_pickle=False) as data:
        tables = {"als": data["als"], "two_tower": data["two_tower"]}
    if config.fault == "degrade_candidate":
        # Fault injection for the gate test: each user's scores are shuffled, which destroys the ranking.
        rng = np.random.default_rng(config.model_seed)
        tables = {name: rng.permuted(table, axis=1) for name, table in tables.items()}
    scored = ScoredModels(base, tables)
    results = {name: evaluate(scored, validation, name, 10, bootstrap_samples=200) for name in tables}
    floor = evaluate(base, validation, "popularity", 10, bootstrap_samples=1)["metrics"]["ndcg"]
    winner = max(results, key=lambda name: results[name]["metrics"]["ndcg"])
    np.savez(config.work_dir / "candidate.npz", table=tables[winner], user_ids=np.array(base.user_ids),
             item_ids=np.array(base.item_ids))
    return _write(config, "evaluation", {
        "split": gate_rules.SPLIT, "candidate_model": winner, "popularity_floor": floor, "fault": config.fault,
        "models": {name: {"ndcg": row["metrics"]["ndcg"], "ci95": row["ci95"]["ndcg"], "users": row["users"]}
                   for name, row in results.items()},
        "candidate": {"metric": gate_rules.METRIC, "split": gate_rules.SPLIT, "value": results[winner]["metrics"]["ndcg"],
                      "users": results[winner]["users"], "eval_sha256": rows_digest(validation)}})


def run_gate(config: PipelineConfig) -> dict:
    registry = Registry(config.registry_dir)
    champion = registry.champion()
    path = config.work_dir / "evaluation.json"
    evaluation = json.loads(path.read_text()) if path.exists() else {}
    decision = gate_rules.decide(evaluation.get("candidate"), champion["evaluation"] if champion else None,
                                 config.tolerance, evaluation.get("popularity_floor"))
    return _write(config, "gate", {"decision": decision.to_dict(), "tolerance": config.tolerance,
                                   "champion_version_at_decision": champion["version"] if champion else None,
                                   "candidate_model": evaluation.get("candidate_model")})


def register(config: PipelineConfig) -> dict:
    """Add and promote a version only after a passing gate file. A rerun of the same artifact is a no-op."""
    gate = _read(config, "gate")
    if not gate["decision"]["passed"]:
        raise PermissionError("the gate did not pass; refusing to register")
    registry = Registry(config.registry_dir)
    registry.root.mkdir(parents=True, exist_ok=True)
    evaluation = _read(config, "evaluation")
    if gate["decision"]["candidate_value"] != evaluation["candidate"]["value"]:
        raise PermissionError("the gate decision does not belong to this evaluation; refusing to register")
    with np.load(config.work_dir / "candidate.npz", allow_pickle=False) as data:
        table, user_ids, item_ids = data["table"], data["user_ids"].tolist(), data["item_ids"].tolist()
    champion = registry.champion()
    if champion is not None:
        current, _, _, _ = registry.load(champion["version"])
        if current.shape == table.shape and np.array_equal(current, table):
            return _write(config, "register", {"outcome": "no_change", "version": champion["version"]})
    version = registry.add_version(table, user_ids, item_ids, {
        "data": "seeded synthetic only", "model": evaluation["candidate_model"], "evaluation": evaluation["candidate"],
        "model_seed": config.model_seed, "split_sha256": _read(config, "extract")["split_sha256"],
        "gate": gate["decision"]})
    registry.promote(version)
    return _write(config, "register", {"outcome": "promoted", "version": version})


def deploy(config: PipelineConfig, client=None) -> dict:
    """Ask a local API to reload the champion. Without a URL or client, no service is contacted."""
    if config.reload_url is None and client is None:
        return _write(config, "deploy", {"outcome": "skipped_no_api", "reloaded_version": None})
    try:
        response = client.post("/admin/reload") if client is not None else httpx.post(config.reload_url, timeout=10)
    except httpx.HTTPError as error:
        return _write(config, "deploy", {"outcome": "reload_failed", "reloaded_version": None, "error": type(error).__name__})
    if response.status_code != 200:
        return _write(config, "deploy", {"outcome": "reload_failed", "reloaded_version": None,
                                         "status_code": response.status_code})
    return _write(config, "deploy", {"outcome": "reloaded", "reloaded_version": response.json()["champion_version"]})


def run_step(step: str, config: PipelineConfig, client=None) -> dict:
    if step == "deploy":
        return deploy(config, client)
    return {"extract": extract, "features": features, "train": train_models, "evaluate": run_evaluation,
            "gate": run_gate, "register": register}[step](config)


def run_pipeline(config: PipelineConfig, client=None) -> dict:
    """Run every step. An evaluation failure or a failed gate ends the run before registration."""
    registry = Registry(config.registry_dir)
    before = registry.champion()
    for name in ("register", "deploy"):
        (config.work_dir / f"{name}.json").unlink(missing_ok=True)
    for step in ("extract", "features", "train"):
        run_step(step, config)
    error = None
    try:
        run_step("evaluate", config)
    except EvaluationFailure as failure:
        error = str(failure)
    gate = run_gate(config)
    summary = {"fault": config.fault, "model_seed": config.model_seed, "evaluation_error": error,
               "gate": gate["decision"], "candidate_model": gate["candidate_model"],
               "champion_before": before["version"] if before else None}
    if gate["decision"]["passed"]:
        summary["register"] = register(config)
        summary["deploy"] = deploy(config, client)
        summary["status"] = summary["register"]["outcome"]
    else:
        summary["status"] = "blocked"
    after = registry.champion()
    summary["champion_after"] = after["version"] if after else None
    registry.record_decision({key: summary[key] for key in ("status", "fault", "model_seed", "gate", "candidate_model",
                                                            "champion_before", "champion_after")})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=(*STEPS, "all"), default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/pipeline_work"))
    parser.add_argument("--registry", type=Path, default=Path("runs/registry"))
    parser.add_argument("--model-seed", type=int, default=11)
    parser.add_argument("--fault", choices=FAULTS, default="none")
    parser.add_argument("--reload-url", default=os.environ.get("GAMEREC_RELOAD_URL"))
    args = parser.parse_args()
    config = PipelineConfig(args.work_dir, args.registry, model_seed=args.model_seed, fault=args.fault,
                            reload_url=args.reload_url)
    if args.step == "all":
        config.work_dir.mkdir(parents=True, exist_ok=True)
        result = run_pipeline(config)
        print(json.dumps(result, indent=2))
        return BLOCKED_EXIT_CODE if result["status"] == "blocked" else 0
    result = run_step(args.step, config)
    if args.step == "gate":
        print(json.dumps(result["decision"]))
        return 0 if result["decision"]["passed"] else BLOCKED_EXIT_CODE
    print(json.dumps({"step": args.step, "done": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
