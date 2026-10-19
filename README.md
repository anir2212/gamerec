# GameRec

A small recommendation lab for the awkward period when a game is new and collaborative signals are sparse. It compares popularity, item-neighbour similarity, ALS, BPR, genre preferences, and description-based TF-IDF under the same temporal split. A local API serves the rankings; a separate toolkit exercises experiment statistics.

## Why this project

New games need a sensible ranking before there is enough interaction history for collaborative filtering. I wanted to see whether catalog copy could help that case without pretending that a synthetic hit proxy is a live click. The evaluation reports low-count items separately and keeps model selection on validation data.

## Milestones

| Plan milestone | Where it stands |
|---|---|
| M1–M2: data and ranking baselines | A 3,000-interaction seed-7 fixture is split by each user's time; six methods are evaluated with validation-only selection. |
| M3: deep and content models | Genre and description TF-IDF baselines run. Text reached 0.2764 cold-item NDCG@10 versus 0.2169 for genre on 21 synthetic users. Learned deep models remain open. |
| M4: serving | The local API filters owned games and handles an unknown user with declared genres. |
| M5: benchmarking and iteration | A five-second localhost run handled 1,000 requests at about 200/s with zero HTTP errors. The iteration log records feature decisions; no LLM-suggested feature was used. |
| M6: A/B toolkit | Assignment, SRM, CUPED and repeated-peeking calculations run on simulated data. The offline hit proxy is not a live product result. |
| M7: pipeline and monitoring | A production pipeline and monitoring loop remain open. |

The next useful comparison needs data with clear rights, real event times and richer game descriptions. A live experiment also needs logged exposures and predeclared success and safety metrics. [RESULTS.md](RESULTS.md) has denominators and limitations.

## What has run

- The seed-7 fixture contains 120 fictional users, 80 games, and 3,000 interactions. Every user's later plays are held out.
- TF-IDF text won validation NDCG@10 (0.3875 versus 0.3703 for genre content), but genre content was slightly higher on the untouched overall test slice (0.4026 versus 0.3979). On the 21-user cold-item slice, text scored 0.2764 NDCG@10 versus genre's 0.2169. Those descriptions mostly restate genre, so this is a diagnostic result, not a claim about real players.
- A FastAPI route supports each method, removes already-owned games, and limits genre concentration. Nine local tests pass, including a same-genre fixture showing that text changes an unplayed item's rank.
- The A/B toolkit covers assignment checks, power, CUPED, confidence intervals, and the cost of repeated uncorrected peeks. Its reported uplift is an offline synthetic proxy.

## Run locally

You need Python 3.12.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make all
make serve
```

`make all` runs tests, writes `runs/synthetic_baselines.json`, and writes `runs/experiment_baseline.json`. To try the text method after `make serve` starts, request `http://127.0.0.1:18081/recommend?user_id=u015&k=10&method=text`. The API trains the small seeded model on its first request. [Results](RESULTS.md) gives denominators and limitations; [Experiments](EXPERIMENTS.md) records the changes and decisions.

## Repository map

`recstack/data.py` builds the fixture and temporal splits. `recstack/models.py` fits the six scoring methods, including a TF-IDF item matrix and training-play-weighted user text profiles. `recstack/eval.py` computes user-level ranking metrics and bootstrap intervals; `recstack/run.py` selects on validation and reports all methods on test. `recstack/serve.py` exposes the local API. `tests/` covers ranking, serving, and experiment calculations. The preserved `gamerec.py` is an earlier standard-library prototype.

## Scope

Users, games, descriptions, and plays are synthetic. The text here is lexical TF-IDF, not a learned semantic encoder or two-tower model. Offline ranking scores and the A/B exercise do not establish live product lift. Real timestamped interactions, richer catalog copy, exposure logs, and a licensed experiment dataset are still needed.
