"""End-to-end gate, registry and API reload behaviour on the seed-7 synthetic corpus."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recstack import pipeline, serve
from recstack.pipeline import PipelineConfig, run_pipeline
from recstack.pipeline_demo import snapshot
from recstack.registry import Registry, RegistryError
from recstack.serve import app, trained_models


@pytest.fixture(autouse=True)
def fresh_api_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMEREC_REGISTRY", str(tmp_path / "registry"))
    serve._champion.update(version=None, table=None)
    yield
    serve._champion.update(version=None, table=None)


def config_for(tmp_path, **changes):
    return PipelineConfig(tmp_path / "work", tmp_path / "registry", **changes)


def recommended(client, method="champion", user="u015"):
    response = client.get("/recommend", params={"user_id": user, "k": 5, "method": method})
    assert response.status_code == 200
    return [item["item_id"] for item in response.json()["items"]]


def add_table(registry_dir, table, value=0.4):
    base = trained_models()
    registry = Registry(registry_dir)
    registry.root.mkdir(parents=True, exist_ok=True)
    version = registry.add_version(table, base.user_ids, base.item_ids, {
        "evaluation": {"metric": "ndcg@10", "split": "validation", "value": value, "users": 100, "eval_sha256": "x"}})
    registry.promote(version)
    return version


def ranked_table(first_item_index: int):
    base = trained_models()
    table = np.zeros((len(base.user_ids), len(base.item_ids)))
    table[:, first_item_index] = 1.0
    return table


def test_failing_candidate_does_not_change_champion_files_or_served_recommendations(tmp_path):
    client = TestClient(app)
    first = run_pipeline(config_for(tmp_path), client)
    assert first["status"] == "promoted" and first["gate"]["reason"] == "first_champion"
    assert first["deploy"] == {"outcome": "reloaded", "reloaded_version": "v0001"}
    registry = Registry(tmp_path / "registry")
    before = snapshot(registry)
    served_before = recommended(client)

    blocked = run_pipeline(config_for(tmp_path, fault="degrade_candidate"), client)

    assert blocked["status"] == "blocked"
    assert blocked["gate"]["reason"] == "below_champion_minus_tolerance"
    assert blocked["gate"]["candidate_value"] < blocked["gate"]["champion_value"] - 0.005
    assert blocked["champion_after"] == "v0001"
    assert snapshot(registry) == before
    assert [path.name for path in registry.versions.iterdir()] == ["v0001"]
    assert "deploy" not in blocked
    assert recommended(client) == served_before
    assert client.get("/health").json()["champion_version"] == "v0001"
    assert [row["status"] for row in registry.decisions()] == ["promoted", "blocked"]


def test_register_itself_refuses_after_a_failed_gate(tmp_path):
    config = config_for(tmp_path)
    run_pipeline(config)
    run_pipeline(config_for(tmp_path, fault="degrade_candidate"))
    before = snapshot(Registry(tmp_path / "registry"))
    with pytest.raises(PermissionError, match="did not pass"):
        pipeline.register(config)
    assert snapshot(Registry(tmp_path / "registry")) == before


def test_register_refuses_a_gate_file_from_another_evaluation(tmp_path):
    config = config_for(tmp_path)
    run_pipeline(config)
    gate = json.loads((config.work_dir / "gate.json").read_text())
    gate["decision"]["candidate_value"] = 0.123456
    (config.work_dir / "gate.json").write_text(json.dumps(gate))
    with pytest.raises(PermissionError, match="does not belong"):
        pipeline.register(config)


def test_failed_evaluation_blocks_and_clears_stale_files(tmp_path):
    client = TestClient(app)
    run_pipeline(config_for(tmp_path), client)
    registry = Registry(tmp_path / "registry")
    before = snapshot(registry)
    blocked = run_pipeline(config_for(tmp_path, fault="evaluation_error"), client)
    assert blocked["status"] == "blocked"
    assert blocked["gate"]["reason"] == "evaluation_failed"
    assert blocked["evaluation_error"] == "injected evaluation failure"
    assert snapshot(registry) == before
    assert not (tmp_path / "work" / "candidate.npz").exists()


def test_degraded_first_candidate_cannot_become_champion_when_below_floor(tmp_path):
    summary = run_pipeline(config_for(tmp_path, fault="degrade_candidate"))
    assert summary["status"] == "blocked"
    assert summary["gate"]["reason"] == "below_first_champion_floor"
    assert summary["champion_after"] is None
    assert Registry(tmp_path / "registry").champion() is None


def test_identical_rerun_is_a_no_change_and_passing_retrain_gets_a_new_version(tmp_path):
    run_pipeline(config_for(tmp_path))
    again = run_pipeline(config_for(tmp_path))
    assert again["status"] == "no_change" and again["champion_after"] == "v0001"
    relaxed = run_pipeline(config_for(tmp_path, model_seed=23, tolerance=0.5))
    assert relaxed["status"] == "promoted" and relaxed["champion_after"] == "v0002"
    assert relaxed["gate"]["reason"] == "within_tolerance_of_champion"


def test_extract_leaves_the_test_split_out_and_gate_never_reads_it(tmp_path):
    payload = pipeline.extract(config_for(tmp_path))
    assert set(payload) == {"data", "corpus_seed", "train", "validation", "catalog", "split_sha256"}
    assert set(payload["split_sha256"]) == {"train", "validation"}
    assert payload["data"] == "seeded synthetic only"


def test_steps_can_run_one_at_a_time_and_gate_cli_exit_blocks(tmp_path, monkeypatch, capsys):
    work, registry = tmp_path / "work", tmp_path / "registry"
    arguments = ["--work-dir", str(work), "--registry", str(registry)]
    for step in pipeline.STEPS:
        monkeypatch.setattr("sys.argv", ["pipeline", "--step", step, *arguments])
        assert pipeline.main() == 0
    assert Registry(registry).champion()["version"] == "v0001"
    for step in ("evaluate", "gate"):
        monkeypatch.setattr("sys.argv", ["pipeline", "--step", step, "--fault", "degrade_candidate", *arguments])
        code = pipeline.main()
    assert code == pipeline.BLOCKED_EXIT_CODE
    monkeypatch.setattr("sys.argv", ["pipeline", "--fault", "degrade_candidate", *arguments])
    assert pipeline.main() == pipeline.BLOCKED_EXIT_CODE
    capsys.readouterr()


def test_reload_serves_a_newly_promoted_version(tmp_path):
    client = TestClient(app)
    add_table(tmp_path / "registry", ranked_table(40))
    assert client.post("/admin/reload").json()["champion_version"] == "v0001"
    base = trained_models()
    assert recommended(client)[0] == base.item_ids[40]
    add_table(tmp_path / "registry", ranked_table(41))
    assert recommended(client)[0] == base.item_ids[40]  # Promotion alone does not change the process.
    assert client.post("/admin/reload").json()["champion_version"] == "v0002"
    assert recommended(client)[0] == base.item_ids[41]
    assert client.get("/health").json()["champion_version"] == "v0002"
    response = client.get("/recommend", params={"user_id": "u015", "k": 5, "method": "champion"})
    assert response.json()["champion_version"] == "v0002"


def test_reload_refuses_a_corrupt_artifact_and_keeps_serving_the_old_table(tmp_path):
    client = TestClient(app)
    add_table(tmp_path / "registry", ranked_table(40))
    client.post("/admin/reload")
    served = recommended(client)
    version = add_table(tmp_path / "registry", ranked_table(41))
    path = tmp_path / "registry" / "versions" / version / "scores.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    response = client.post("/admin/reload")
    assert response.status_code == 409 and "hash mismatch" in response.json()["detail"]
    assert recommended(client) == served
    assert client.get("/health").json()["champion_version"] == "v0001"


def test_reload_refuses_other_corpus_and_missing_champion(tmp_path):
    client = TestClient(app)
    assert client.post("/admin/reload").status_code == 409
    registry = Registry(tmp_path / "registry")
    registry.root.mkdir(parents=True)
    version = registry.add_version(np.zeros((1, 2)), ["other"], ["a", "b"],
                                   {"evaluation": {"metric": "ndcg@10"}})
    registry.promote(version)
    response = client.post("/admin/reload")
    assert response.status_code == 409 and "corpus" in response.json()["detail"]
    assert client.get("/health").json()["champion_version"] is None


def test_reload_rejects_a_non_local_client_and_champion_needs_a_load(tmp_path):
    add_table(tmp_path / "registry", ranked_table(40))
    assert TestClient(app, client=("10.0.0.5", 5000)).post("/admin/reload").status_code == 403
    client = TestClient(app)
    response = client.get("/recommend", params={"user_id": "u015", "method": "champion"})
    assert response.status_code == 409


def test_champion_route_excludes_owned_games_and_keeps_the_unknown_user_fallback(tmp_path):
    client = TestClient(app)
    add_table(tmp_path / "registry", ranked_table(0))
    client.post("/admin/reload")
    assert not set(recommended(client)).intersection(trained_models().history["u015"])
    unknown = client.get("/recommend", params={"user_id": "new-user", "method": "champion", "genres": "racing"})
    assert unknown.status_code == 200
    assert unknown.json()["method"] == "genre-popularity-fallback"
    assert unknown.json()["items"][0]["genre"] == "racing"


def test_deploy_reports_a_failed_reload_without_claiming_success(tmp_path):
    client = TestClient(app)
    config = config_for(tmp_path)
    assert pipeline.deploy(config)["outcome"] == "skipped_no_api"
    assert pipeline.deploy(config, client)["outcome"] == "reload_failed"
    unreachable = config_for(tmp_path, reload_url="http://127.0.0.1:1/admin/reload")
    assert pipeline.deploy(unreachable)["outcome"] == "reload_failed"


def test_registry_rejects_bad_tables_and_tampered_files(tmp_path):
    registry = Registry(tmp_path)
    with pytest.raises(RegistryError, match="shape"):
        registry.add_version(np.zeros((2, 2)), ["u"], ["a", "b"], {})
    with pytest.raises(RegistryError, match="non-finite"):
        registry.add_version(np.array([[np.nan, 0.0]]), ["u"], ["a", "b"], {})
    version = registry.add_version(np.ones((1, 2)), ["u"], ["a", "b"], {"evaluation": {}})
    (tmp_path / "versions" / version / "scores.npz").write_bytes(b"no")
    with pytest.raises(RegistryError, match="hash"):
        registry.promote(version)
    assert registry.champion() is None
    with pytest.raises(RegistryError, match="unknown"):
        registry.load("v9999")


def test_pipeline_settings_match_the_m3_selection():
    selected = json.loads(Path("evidence/m3_selection.json").read_text())["two_tower_selected"]
    for key, value in PipelineConfig(Path("w"), Path("r")).tower.items():
        assert selected[key] == value


def test_dag_file_steps_match_the_pipeline_without_needing_airflow():
    tree = ast.parse(Path("dags/gamerec_pipeline.py").read_text())
    steps = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and node.targets[0].id == "STEPS")
    assert steps == pipeline.STEPS
