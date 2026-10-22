# GameRec measured results

All numbers on this page come from the **seed-7 synthetic corpus** and local scripts. They are development measurements, not claims about real players. `runs/` contains the generated JSON files and is ignored by Git; the commands in the README recreate them. The split manifests in `data/manifests/` are committed evidence.

## Data and offline ranking

`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m recstack.run --seed 7` produced 3,000 interactions from 120 fictional users and 80 games. Per-user temporal splitting gave **1,808 train**, **596 validation**, and **596 test** interactions. Ten users had at most three training interactions; 22 catalog items had at most five, with 27 test events on those items. A user is scored only if they have a novel test item not already owned in training, leaving **97 overall**, **7 short-history**, and **21 cold-item** evaluation users. These small cold denominators matter.

With the added TF-IDF description baseline, validation NDCG@10 selected **text** (0.3875 versus 0.3703 for genre content). The table below reports untouched test results; intervals are 95% percentile bootstrap intervals over users with 400 resamples. Genre content means genre preference plus a small popularity tie-breaker. Text means TF-IDF of the generated item description, weighted by training-play confidence; it is not a learned semantic encoder.

| Model | Overall NDCG@10 (95% CI) | Overall Recall@10 | Cold-item NDCG@10 | Cold-item Recall@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.2510 (0.1942–0.3196) | 0.3393 | 0.0000 | 0.0000 |
| Item cosine kNN | 0.3854 (0.3223–0.4492) | 0.5679 | 0.2170 | 0.6429 |
| Implicit ALS | 0.1741 (0.1297–0.2111) | 0.3333 | 0.0717 | 0.1905 |
| BPR | 0.3598 (0.2957–0.4282) | 0.5438 | 0.1996 | 0.5714 |
| Genre content | **0.4026 (0.3446–0.4662)** | 0.5971 | 0.2169 | 0.6905 |
| Description TF-IDF | 0.3979 (0.3424–0.4572) | **0.6143** | **0.2764** | **0.7857** |

The validation-selected text model did **not** have the highest overall test NDCG: genre content was 0.4026 versus text 0.3979. Text's observed cold-item NDCG was 0.2764 versus genre content's 0.2169, but the cold slice has only 21 users and the generated descriptions largely repeat genre labels; this is a development-fixture result, not proof of cold-start gain on real games. ALS is weak under the current fixed settings and was not tuned. The JSON report also contains MAP, hit rate, coverage, novelty and @20 values. No Steam, learned text encoder, two-tower, LambdaRank or FAISS result exists yet.

## Dataset contract and split manifests (synthetic, 20 October 2026)

`make manifest` wrote the two committed files [`data/manifests/synthetic_seed7_per_user.json`](data/manifests/synthetic_seed7_per_user.json) and [`data/manifests/synthetic_seed7_global.json`](data/manifests/synthetic_seed7_global.json). `make all` now recomputes them and fails on any difference. Both use the seed-7 fixture under `SYNTHETIC_CONTRACT` (`time_semantics = synthetic_ordinal`). All 3,000 rows passed validation and none were dropped by cleaning.

| Protocol | Cut-offs | Train / validation / test rows | Users in test | Disclosed overlaps |
|---|---|---:|---:|---|
| `per_user` (used for every ranking result above) | each user's last 20% is test and the 20% before it is validation; boundaries hashed per user | 1,808 / 596 / 596 | 120 | 100 test rows are earlier than the latest training event of some *other* user; 410 test rows repeat a game the user already owns in training |
| `global` (prepared for real data; not used to train or score models) | validation from time 15,110, test from time 21,066 | 1,800 / 600 / 600 | 92 | 0 users without training history; 410 test rows repeat an owned game |

The leakage checks passed for both protocols: time order under the protocol's rule, no row in two splits, and an exact partition of the validated input. The per-user split hashes in the manifest equal the `split_sha256` values that `recstack.run` now writes into `runs/synthetic_baselines.json`. This confirms that the ranking table above used exactly the manifest's rows. The cross-user overlap is a property of the per-user protocol and is now stated rather than hidden. The ranking and experiment numbers did not change in this checkpoint. These are fixture properties, not a real-data result: no real records were imported (see [data/SOURCES.md](data/SOURCES.md)).

## Local serving

The FastAPI `TestClient` test passed for owned-item exclusion with the text method, a declared-genre unknown-user fallback, five unique returned items, and invalid-method rejection. A separate loopback server benchmark warmed model loading and then sent open-loop requests for five seconds at each rate:

| Target rate | Requests | Achieved rate | Errors | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 50/s | 250 | 50.17/s | 0 | 1.78 ms | 4.93 ms | 19.18 ms |
| 200/s | 1,000 | 200.10/s | 0 | 1.67 ms | 2.58 ms | 7.86 ms |

These are one short local run each, with a tiny 80-item catalog and no Redis/network deployment. The p95 inversion between rates shows ordinary run noise; it is not evidence that higher load improves latency. The server process was stopped after the run.

## Experiment toolkit

`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m recstack.experiment_run` measured:

- An **offline held-out hit proxy** with deterministic user assignment: ALS control 27/54 = 50.0%, genre-content treatment 35/43 = 81.4%, observed difference **+31.40 percentage points**, bootstrap 95% CI **+13.35 to +48.06 points**, two-sided pooled z-test p = 0.00138. SRM chi-square p = 0.264. This is a synthetic, retrospective proxy with 97 users and no impressions, randomization in a live product, or actual clicks; it cannot justify a launch decision.
- A power example at 20% baseline and one percentage point absolute lift gave **25,583 users per arm** for two-sided alpha 0.05 and 80% power under the normal approximation.
- CUPED on a 2,000-row correlated synthetic fixture reduced outcome variance by **35.86%** (theta 0.7576). This is a unit fixture, not measured variance reduction in an experiment.
- In 500 seeded null simulations with ten daily uncorrected looks, the fixed-horizon false-positive fraction was **7.4%** and the fraction with any daily p < 0.05 was **22.6%**. Monte Carlo noise applies, but this illustrates the peeking problem.

On 20 October 2026, `make all` passed **31 tests** (9 before this checkpoint), checked both saved split manifests, and reran the seeded ranking and experiment reports. The ranking and experiment values above were unchanged. The suite includes contract rejection of untimed, mistyped, duplicate and unknown-catalog rows; rejection of temporal claims for `snapshot`/`none` sources; manifest determinism and drift detection; global cut-off and tie rules; corrupted-split leakage failures; a deliberate temporal-leakage failure, a same-genre text discrimination fixture, owned-item exclusion, API behavior, SRM, bootstrap inference and CUPED variance reduction. The historical prototype tests remain included. No real Cookie Cats CSV was used because the inspected data card lists its licence as unknown. No Steam or Kaggle record was downloaded or used.
