# GameRec

Recommendation baselines and offline experiment analysis.

A recommendation lab comparing popularity, item similarity, ALS, BPR, and content scoring. It uses temporal splits, reports cold-start slices, serves a local ranking endpoint, and includes an A/B statistics exercise.

## Why this project

New games need a sensible ranking before there is enough interaction history for collaborative filtering. This project compares simple and learned recommenders under the same temporal split, with cold-start results visible instead of hidden in one aggregate score.

## Run locally

You need Python 3.12.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make all
make serve
```

`make all` builds the seeded dataset, runs evaluation, and writes experiment artifacts. [Experiments](EXPERIMENTS.md) records the iterations; [results](RESULTS.md) records measured scores and denominators.

## Repository map

`recstack/` contains data generation, ranking models, evaluation, and serving; `tests/` covers the pipeline; `EXPERIMENTS.md` records iteration choices.

## Scope

Users and games are synthetic. Offline ranking scores and the A/B exercise do not establish live product lift.
