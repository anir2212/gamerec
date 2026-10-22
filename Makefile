PYTHON := .venv/bin/python
export OPENBLAS_NUM_THREADS := 1

.PHONY: all test data eval experiment serve bench manifest manifest-check
all: test manifest-check eval experiment

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

serve:
	$(PYTHON) -m uvicorn recstack.serve:app --host 127.0.0.1 --port 18081 --no-access-log

bench:
	$(PYTHON) -m bench.serve_bench --seconds 5
