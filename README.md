# GameRec

A recommendation lab for the awkward period when a game is new and collaborative signals are sparse. It compares six ranking baselines, a two-tower model and a LambdaRank re-ranker under the same temporal split. A local API serves the rankings, a file-based promotion gate controls its champion, and a separate toolkit exercises experiment statistics.

## Why this project

New games need a sensible ranking before there is enough interaction history for collaborative filtering. I wanted to see whether catalog copy could help that case without pretending that a synthetic hit proxy is a live click. The evaluation reports low-count items separately and keeps model selection on validation data.

## Milestones

| Plan milestone | Where it stands |
|---|---|
| M1–M2: data and ranking baselines | A 3,000-interaction seed-7 fixture is split by each user's time; six methods are evaluated with validation-only selection. A dataset contract rejects untimed or malformed rows before any temporal claim, and committed split manifests record cut-offs, counts, row hashes and leakage checks. |
| M3: deep and content models | Two-tower and LambdaRank models ran on the synthetic corpus under a protocol committed before scoring. Under the original lowest-ID tie rule, text won validation at 0.3875. A separately declared, popularity-neutral tie rule changed the validation winner to genre content at 0.3703 on 7 November 2026. The deep models did not win either selection. The test split is a known fixture, so its scores are diagnostic. |
| M4: serving | The local API filters owned games, handles an unknown user with declared genres, and reloads the validated registry champion through a loopback-only admin route. |
| M5: benchmarking and iteration | A five-second localhost run handled 1,000 requests at about 200/s with zero HTTP errors. The iteration log records feature decisions; no LLM-suggested feature was used. |
| M6: A/B toolkit | Assignment, SRM, CUPED and repeated-peeking calculations run on simulated data. The offline hit proxy is not a live product result. |
| M7: pipeline and monitoring | A local plain-Python pipeline, promotion gate and file registry run. Tests show a failing candidate leaves the champion byte-identical. Monitors ran on authored simulated weeks; the Airflow DAG passed an import check but was not run by a scheduler. |

The next useful comparison needs data with clear rights, real event times and richer game descriptions. A 20 October audit of candidate real sources is in [data/SOURCES.md](data/SOURCES.md): the CC0-labelled Kaggle Steam set has descriptions, but the meaning of its date column and its 2.2 GB size need a decision first, and the UCSD Steam pages state no reuse licence. Nothing was downloaded. A live experiment also needs logged exposures and predeclared success and safety metrics. [RESULTS.md](RESULTS.md) has denominators and limitations.

## What has run

- The seed-7 fixture contains 120 fictional users, 80 games, and 3,000 interactions. Every user's later plays are held out.
- Under the original lowest-item-ID tie rule, text won validation NDCG@10 (0.3875 versus 0.3703 for genre content); on the already-known test fixture, genre content scored 0.4026 versus text's 0.3979. The declared fair tie rule changed text's validation score to 0.3096 and selected genre content instead. See [Results](RESULTS.md) for the two dated protocols and their limitations.
- A two-tower model scored 0.4367 and a LambdaRank re-ranker 0.4292 overall test NDCG@10 under the original tie rule, but paired intervals against text include zero. Content helped over ID-only embeddings on the 21-user low-count-item slice; no never-seen test item was measured.
- The FastAPI route removes owned games and limits genre concentration. Its champion reload checks the registry hash and corpus; a failed reload leaves the prior table in place. The admin route is loopback-only, not authenticated.
- The pipeline gate blocks a shuffled or failed candidate without changing the champion. Simulated-week monitors raised alerts under authored drift; item-popularity PSI also exceeded its rule-of-thumb threshold in all eight control weeks.
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

`make all` runs tests, checks that `data/manifests/*.json` match a fresh computation, writes `runs/synthetic_baselines.json` and `runs/experiment_baseline.json`, and reproduces the committed M7 demo and monitoring evidence. `make deep` and `make tiebreak` reproduce their separately dated, committed protocols. `make pipeline-demo` exercises a fresh local registry; `make monitor` uses it to replay simulated weeks. `make manifest` rewrites split manifests after an intended change. To try the text method after `make serve` starts, request `http://127.0.0.1:18081/recommend?user_id=u015&k=10&method=text`. The API trains the small seeded model on its first request. [Results](RESULTS.md) gives denominators and limitations; [Experiments](EXPERIMENTS.md) records the changes and decisions.

## Repository map

`recstack/data.py` builds the fixture and temporal splits. `recstack/models.py` fits the six baseline methods; `recstack/two_tower.py` and `recstack/reranker.py` fit the added models. `recstack/eval.py` computes user-level ranking metrics and bootstrap intervals; `recstack/deep_run.py` and `recstack/tiebreak_run.py` reproduce the dated comparisons. `recstack/pipeline.py`, `recstack/gate.py`, `recstack/registry.py` and `recstack/monitoring.py` implement the local promotion and simulated monitoring path. `recstack/serve.py` exposes the API. `dags/` contains an import-checked Airflow DAG, not a scheduler run. The preserved `gamerec.py` is an earlier standard-library prototype.

## Scope

Users, games, descriptions, and plays are synthetic. The text baseline is lexical TF-IDF, not a learned semantic encoder. The two-tower uses generated genre-level content, and the monitored drift is authored. Offline ranking scores and the A/B exercise do not establish live product lift. There is no deployed service, Airflow scheduler run, real traffic, or exposure log. Real timestamped interactions, richer catalog copy, and an authorized experiment dataset are still needed.
