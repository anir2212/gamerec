"""Exercise the gate end to end on a fresh file registry and save what happened (synthetic data, local only).

Sequence: first champion, identical rerun, shuffled candidate, failed evaluation, retrain under another seed.
The API is the in-process FastAPI test client; no separate service is deployed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from recstack.gate import DEFAULT_TOLERANCE
from recstack.pipeline import PipelineConfig, run_pipeline
from recstack.registry import Registry, sha256_file
from recstack.serve import app


def snapshot(registry: Registry) -> dict:
    """Hashes that must not move when promotion is blocked."""
    pointer = registry.champion()
    if pointer is None:
        return {"champion": None}
    version_dir = registry.versions / pointer["version"]
    return {"champion": pointer["version"], "pointer_sha256": sha256_file(registry.pointer),
            "scores_sha256": sha256_file(version_dir / "scores.npz"),
            "metadata_sha256": sha256_file(version_dir / "metadata.json")}


def served(client: TestClient) -> dict:
    response = client.get("/recommend", params={"user_id": "u015", "k": 5, "method": "champion"})
    if response.status_code != 200:
        return {"status_code": response.status_code, "items": None, "version": None}
    body = response.json()
    return {"status_code": 200, "items": [item["item_id"] for item in body["items"]], "version": body["champion_version"]}


def run_demo(root: Path) -> dict:
    shutil.rmtree(root, ignore_errors=True)
    registry_dir = root
    registry = Registry(registry_dir)
    os.environ["GAMEREC_REGISTRY"] = str(registry_dir)
    work = root.parent / f"{root.name}_work"
    shutil.rmtree(work, ignore_errors=True)
    client = TestClient(app)
    stages = []
    for name, seed, fault in (("first_champion", 11, "none"),
                              ("same_seed_rerun", 11, "none"),
                              ("degraded_candidate", 11, "degrade_candidate"),
                              ("failed_evaluation", 11, "evaluation_error"),
                              ("retrain_other_seed", 23, "none")):
        before = snapshot(registry)
        served_before = served(client)
        config = PipelineConfig(work, registry_dir, model_seed=seed, fault=fault)
        summary = run_pipeline(config, client)
        after = snapshot(registry)
        stages.append({"stage": name, "summary": summary, "registry_before": before, "registry_after": after,
                       "registry_unchanged": before == after, "served_before": served_before,
                       "served_after": served(client), "health_version": client.get("/health").json()["champion_version"]})
    return {"label": "seeded synthetic data, local file registry, in-process API client; no deployed service",
            "tolerance": DEFAULT_TOLERANCE, "stages": stages,
            "decisions_logged": len(registry.decisions())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/registry_demo"))
    parser.add_argument("--output", type=Path, default=Path("evidence/m7_pipeline_demo.json"))
    args = parser.parse_args()
    report = run_demo(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{key: stage[key] for key in ("stage", "registry_unchanged", "served_after", "health_version")}
                      | {"status": stage["summary"]["status"], "gate": stage["summary"]["gate"]}
                      for stage in report["stages"]], indent=1))


if __name__ == "__main__":
    main()
