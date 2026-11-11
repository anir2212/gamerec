"""Airflow DAG import and structure. Skipped outside the Airflow environment; `make dag-check` runs it there."""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="gamerec-airflow-"))
airflow_models = pytest.importorskip("airflow.models")

STEPS = ("extract", "features", "train", "evaluate", "gate", "register", "deploy")


def test_dag_imports_with_the_gate_before_register_and_deploy():
    bag = airflow_models.DagBag(dag_folder=str(Path("dags").resolve()), include_examples=False)
    assert bag.import_errors == {}
    dag = bag.dags["gamerec_retrain_local"]
    assert dag.schedule_interval is None
    assert {task.task_id for task in dag.tasks} == set(STEPS)
    for earlier, later in zip(STEPS, STEPS[1:], strict=False):
        assert dag.get_task(earlier).downstream_task_ids == {later}
    assert dag.get_task("deploy").downstream_task_ids == set()
    for step in STEPS:
        assert f"--step {step}" in dag.get_task(step).bash_command
