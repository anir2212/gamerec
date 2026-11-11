"""Airflow DAG file for the local GameRec pipeline.

Only import and structure are tested (`make dag-check`); this DAG has never been run by a
scheduler. Each task runs one `recstack.pipeline` step in the project's own environment, so parsing
the DAG imports none of the model code. The gate step exits non-zero when it blocks, so `register`
and `deploy` do not run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = Path(__file__).resolve().parents[1]
STEPS = ("extract", "features", "train", "evaluate", "gate", "register", "deploy")

with DAG(dag_id="gamerec_retrain_local", start_date=datetime(2026, 11, 1), schedule=None, catchup=False,
         tags=["synthetic", "local"], doc_md=__doc__) as dag:
    previous = None
    for step in STEPS:
        task = BashOperator(task_id=step, cwd=str(PROJECT),
                            bash_command=f".venv/bin/python -m recstack.pipeline --step {step}")
        if previous is not None:
            previous >> task
        previous = task
