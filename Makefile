PYTHON := .venv/bin/python
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1

.PHONY: all test data eval experiment serve bench manifest manifest-check deep tiebreak pipeline pipeline-demo monitor dag-check
all: test manifest-check eval experiment pipeline-demo monitor

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -m recstack.run --seed 7

eval:
	$(PYTHON) -m recstack.run --seed 7

manifest:
	$(PYTHON) -m recstack.manifest --seed 7 --protocol per_user
	$(PYTHON) -m recstack.manifest --seed 7 --protocol global

manifest-check:
	$(PYTHON) -m recstack.manifest --seed 7 --protocol per_user --check
	$(PYTHON) -m recstack.manifest --seed 7 --protocol global --check

experiment:
	$(PYTHON) -m recstack.experiment_run

# Two-tower and LambdaRank comparison on the synthetic corpus; takes under a minute on one thread.
deep:
	$(PYTHON) -m recstack.deep_run --phase select
	$(PYTHON) -m recstack.deep_run --phase compare

# TB-1 tie-break reselection (validation only), then the labelled known-fixture test reading.
# Does not touch the earlier lowest-ID evidence in evidence/m3_selection.json.
tiebreak:
	$(PYTHON) -m recstack.tiebreak_run --phase select
	$(PYTHON) -m recstack.tiebreak_run --phase fixture

# One pipeline run against the working registry in runs/registry.
pipeline:
	$(PYTHON) -m recstack.pipeline

# Fresh demo registry: first champion, blocked candidates, reload. Writes evidence/m7_pipeline_demo.json.
pipeline-demo:
	$(PYTHON) -m recstack.pipeline_demo

# Simulated weeks against the demo champion. Writes evidence/m7_monitoring_simulated.json.
monitor:
	$(PYTHON) -m recstack.monitor_run

# Needs the separate Airflow environment described in the README. Import and structure only.
dag-check:
	AIRFLOW_HOME=$(CURDIR)/runs/airflow_home .venv-airflow/bin/python -m pytest -q tests/test_dag_import.py

serve:
	$(PYTHON) -m uvicorn recstack.serve:app --host 127.0.0.1 --port 18081 --no-access-log

bench:
	$(PYTHON) -m bench.serve_bench --seconds 5
