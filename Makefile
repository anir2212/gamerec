PYTHON := .venv/bin/python
export OPENBLAS_NUM_THREADS := 1

.PHONY: all test data eval experiment serve bench
all: test eval experiment

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -m recstack.run --seed 7

eval:
	$(PYTHON) -m recstack.run --seed 7

experiment:
	$(PYTHON) -m recstack.experiment_run

serve:
	$(PYTHON) -m uvicorn recstack.serve:app --host 127.0.0.1 --port 18081 --no-access-log

bench:
	$(PYTHON) -m bench.serve_bench --seconds 5
