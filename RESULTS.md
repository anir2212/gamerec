# GameRec measured results

All numbers on this page come from the **seed-7 synthetic corpus** and local scripts; the monitoring section adds **authored simulated weeks**. They are development measurements, not claims about real players. `runs/` contains the generated JSON files and is ignored by Git; the commands in the README recreate them. The split manifests in `data/manifests/` and the JSON in `evidence/` are committed evidence.

## Data and offline ranking

`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m recstack.run --seed 7` produced 3,000 interactions from 120 fictional users and 80 games. Per-user temporal splitting gave **1,808 train**, **596 validation**, and **596 test** interactions. Ten users had at most three training interactions; 22 catalog items had at most five, with 27 test events on those items. A user is scored only if they have a novel test item not already owned in training, leaving **97 overall**, **7 short-history**, and **21 cold-item** evaluation users. These small cold denominators matter.

With the added TF-IDF description baseline, validation NDCG@10 selected **text** (0.3875 versus 0.3703 for genre content) **under the lowest-item-ID tie rule used up to 6 November 2026**. Under the fair tie rule pre-registered on 7 November, that winner changes to genre content; see the TB-1 section below. The 6 November numbers on this page keep their original rule and date. The table below reports test results, which were first read at that baseline checkpoint and so are a known fixture for anything later; intervals are 95% percentile bootstrap intervals over users with 400 resamples. Genre content means genre preference plus a small popularity tie-breaker. Text means TF-IDF of the generated item description, weighted by training-play confidence; it is not a learned semantic encoder.

| Model | Overall NDCG@10 (95% CI) | Overall Recall@10 | Cold-item NDCG@10 | Cold-item Recall@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.2510 (0.1942–0.3196) | 0.3393 | 0.0000 | 0.0000 |
| Item cosine kNN | 0.3854 (0.3223–0.4492) | 0.5679 | 0.2170 | 0.6429 |
| Implicit ALS | 0.1741 (0.1297–0.2111) | 0.3333 | 0.0717 | 0.1905 |
| BPR | 0.3598 (0.2957–0.4282) | 0.5438 | 0.1996 | 0.5714 |
| Genre content | **0.4026 (0.3446–0.4662)** | 0.5971 | 0.2169 | 0.6905 |
| Description TF-IDF | 0.3979 (0.3424–0.4572) | **0.6143** | **0.2764** | **0.7857** |

The validation-selected text model did **not** have the highest overall test NDCG: genre content was 0.4026 versus text 0.3979. Text's observed cold-item NDCG was 0.2764 versus genre content's 0.2169, but the cold slice has only 21 users and the generated descriptions largely repeat genre labels; this is a development-fixture result, not proof of cold-start gain on real games. ALS is weak under the current fixed settings and was not tuned. The JSON report also contains MAP, hit rate, coverage, novelty and @20 values. The 6 November tie-break check below shows that text's overall score leans on how exact ties are broken, and the 7 November TB-1 reselection below acts on that finding with a declared rule. Two-tower and LambdaRank results are in the next section. No Steam, learned text encoder or FAISS result exists.

## Two-tower and LambdaRank, plan M3 (synthetic, 6 November 2026)

**Outcome: the deep models did not win selection, and no overall gain over the text baseline was shown.** Validation again selected text, under this protocol's lowest-item-ID tie rule. (The separately pre-registered TB-1 reselection of 7 November 2026, in the next section, changes that winner to genre content; the deep models still do not win. The numbers in this section are unchanged and keep their 6 November date and rule.) On test, the two-tower had the highest observed overall NDCG@10, but its paired interval against text includes zero.

**Order of work.** [`docs/M3_PROTOCOL.md`](docs/M3_PROTOCOL.md) was committed alone (`bc42bfb`) before any new model was scored on seed 7, and the driver refuses to run on seed 7 otherwise. An AI agent chose the settings after reading the baseline test table, and after a smoke run on a separate seed-11 corpus that set the epoch range. **Test is therefore a known fixture, not an untouched holdout.** There was one scored compare run. Selection reads validation only. A rerun reproduced `evidence/m3_selection.json` byte for byte and all the original fields of the comparison file.

**Selected settings (validation NDCG@10, five seeds: 11, 23, 37, 41, 53).** Two-tower: `dim` 32, `epochs` 5, `logq` on, out of 12 settings. Five epochs is the lowest value in the grid, so the search hit its edge. Re-ranker: `plan` features, 7 leaves, 50 rounds, out of 8 settings, scored out of fold across users.

| Validation NDCG@10 (100 users) | Value |
|---|---:|
| **Text (selected)** | **0.3875** |
| Genre content | 0.3703 |
| LambdaRank re-ranker, out of fold | 0.3679 |
| Item kNN | 0.3533 |
| Two-tower | 0.3531 |
| BPR | 0.3080 |
| Popularity | 0.2176 |
| ALS | 0.1603 |

Test results, 95% user bootstrap intervals. Baseline rows match the table above exactly. Neural rows are the mean over five seeds; **SD** is the between-seed spread of overall NDCG@10.

| Model | Overall NDCG@10 (97 users) | Overall Recall@10 | Cold-item NDCG@10 (21 users) | Cold-item Recall@10 | SD |
|---|---:|---:|---:|---:|---:|
| Two-tower (hybrid) | 0.4367 (0.3765–0.4921) | 0.6452 | 0.2342 (0.1699–0.2983) | 0.6524 | 0.0112 |
| LambdaRank re-ranker | 0.4292 (0.3763–0.4798) | 0.6713 | 0.0895 (0.0400–0.1382) | 0.2905 | 0.0300 |
| Two-tower, content only | 0.4100 (0.3531–0.4664) | 0.6308 | 0.2773 (0.2121–0.3386) | 0.7857 | 0.0034 |
| Genre content | 0.4026 (0.3446–0.4662) | 0.5971 | 0.2169 (0.1477–0.2746) | 0.6905 | – |
| Description TF-IDF | 0.3979 (0.3424–0.4572) | 0.6143 | 0.2764 (0.2121–0.3377) | 0.7857 | – |
| Two-tower, ID only | 0.3879 (0.3269–0.4421) | 0.5930 | 0.0499 (0.0226–0.0859) | 0.1524 | 0.0060 |
| Item kNN | 0.3854 (0.3223–0.4492) | 0.5679 | 0.2170 (0.1551–0.2807) | 0.6429 | – |
| BPR | 0.3598 (0.2957–0.4282) | 0.5438 | 0.1996 (0.1161–0.2792) | 0.5714 | – |
| Popularity | 0.2510 (0.1942–0.3196) | 0.3393 | 0.0000 (0.0000–0.0000) | 0.0000 | – |
| ALS | 0.1741 (0.1297–0.2111) | 0.3333 | 0.0717 (0.0143–0.1516) | 0.1905 | – |

Paired test comparisons named in the protocol (NDCG@10, A minus B, 2,000 user resamples, no multiplicity correction):

| A minus B | Slice (users) | Difference (95% CI) | A better / B better / tied | Read under the protocol |
|---|---|---:|---:|---|
| Two-tower − text | All (97) | +0.0388 (−0.0263 to +0.1076) | 42 / 46 / 9 | No difference shown |
| Re-ranker − text | All (97) | +0.0313 (−0.0185 to +0.0818) | 46 / 41 / 10 | No difference shown |
| Re-ranker − two-tower | All (97) | −0.0075 (−0.0533 to +0.0372) | 42 / 47 / 8 | No difference shown |
| Two-tower − item kNN | Cold item (21) | +0.0173 (−0.0134 to +0.0544) | 10 / 5 / 6 | No difference shown |
| Content only − ID only | Cold item (21) | **+0.2274 (+0.1779 to +0.2759)** | 17 / 0 / 4 | Content helps, on this fixture |
| Hybrid − ID only | Cold item (21) | **+0.1844 (+0.1290 to +0.2384)** | 15 / 0 / 6 | Content helps, on this fixture |
| ID only − content only | Warm item (90) | +0.0188 (−0.0605 to +0.0964) | 38 / 44 / 8 | CF win not shown |
| Item kNN − genre content | Warm user (90) | −0.0115 (−0.0432 to +0.0182) | 22 / 33 / 35 | CF win not shown |

**Where content helps and where CF wins.**
- *Content helps cold items.* With few plays to learn from, an ID-only embedding almost never surfaced those games. Adding content gave the large paired gain above. All 21 users are on a synthetic fixture, and "cold" means at most five training events: only one catalog item (`g070`) has none, and no test event lands on it. So **no never-seen item is measured**. Twenty-one users can show a gap this large exists here and support little else.
- *CF did not win on warm users or items here.* The plan expected it to, and neither comparison shows it. The likely reason is the fixture: the generator draws 74% of plays from one preferred genre, so genre alone carries most of the signal available. That is an explanation, not something this run tested.
- *"Content" here is genre-level information only.* After removing variant numbers, which act as hidden item IDs, same-genre descriptions have identical vectors.
- *The re-ranker gave up cold items.* It fell to 0.0895 on the cold slice against the two-tower's 0.2342, though no paired interval was registered for that pair. Mean gain share by feature: two-tower 62%, log popularity 14%, ALS 13%, genre match 10%, recent genre match 0.7%. It also saw validation labels at fit time, which the baselines did not.
- *Validation and test disagree on order.* Validation picked text and two-tower ranked fifth; test put two-tower first. Nothing was reselected from test.
- *Log-frequency correction.* At every `dim` and `epochs` value, validation NDCG@10 was higher with it on (0.3238–0.3531) than off (0.2984–0.3203). Recommendations also became more popular: novelty fell from 7.80–7.95 to 6.37–6.61, and catalog coverage from 0.982–1.000 to 0.847–0.900. Since popular games are heavily rewarded by this fixture's generator, that does not show the correction is right for real data.
- *Recency is expected to be inert.* The generator draws plays independently of order. Recent genre match took 0.7% of gain.
- `new_user` has only 7 test users, so its intervals are very wide (for example two-tower 0.5274, 0.2658–0.7825). Full numbers for `new_user`, `warm_user`, `warm_item`, @20 and validation are in the evidence.

**Tie-break check (declared in the protocol; validation rows added after the first run).** `recommend` breaks exact score ties by lowest item ID, and the generator makes low IDs popular. Most text and content-only scores are exact ties (8,108 and 9,000 of 9,480 adjacent pairs). Breaking ties at random, over 20 shuffles:

| Model | Slice (users) | NDCG@10, ID tie-break | NDCG@10, random tie-break: mean (min–max) |
|---|---|---:|---:|
| Description TF-IDF | Test all (97) | 0.3979 | 0.3043 (0.2792–0.3341) |
| Description TF-IDF | Test cold item (21) | 0.2764 | 0.3303 (0.2448–0.3943) |
| Description TF-IDF | Validation all (100) | 0.3875 | 0.3159 (0.2991–0.3296) |
| Two-tower content only, seed 11 | Test all (97) | 0.4123 | 0.2956 (0.2617–0.3189) |
| Two-tower content only, seed 11 | Test cold item (21) | 0.2773 | 0.3426 (0.2607–0.4185) |
| Genre content | Test all (97) | 0.4026 | 0.4026 (no ties) |
| Two-tower hybrid, seed 11 | Validation all (100) | 0.3375 | 0.3359 (0.3325–0.3391) |

So **text's overall score, including its validation win, depends on a hidden popularity prior in the tie-break**. On cold items the same tie-break works the other way: random ties raised the mean, because low IDs are rarely cold. Under random ties, text's validation 0.3159 would fall below genre content's 0.3703. The selection rule was fixed in advance, so this M3 report does not reselect, and the main tables above keep the ID tie-break. A declared tie rule and a validation-only reselection followed on 7 November 2026 under their own pre-registered protocol; see the TB-1 section below. The content-only and text results should be read with this in mind. For contrast, in one seed across all users, single-digit IDs filled 34.9% of two-tower top-10 lists and 7.8% of text's.

**Evidence:** [`evidence/m3_selection.json`](evidence/m3_selection.json), [`evidence/m3_test_comparison.json`](evidence/m3_test_comparison.json). **Declared deviations** (TF-IDF for sentence-transformers, no tags or price, all unowned items for top-200, derived segment and recency) are in the protocol.

## Fair tie-breaking and a validation-only reselection, protocol TB-1 (synthetic, 7 November 2026)

**Outcome: the selected model changed. Under the declared tie rule, validation selects genre content (0.3703) and the description TF-IDF text model falls from 0.3875 to 0.3096, from first place to fifth.** The 6 November selection above keeps its own lowest-ID rule, its own numbers and its own date; this section is a second, separately pre-registered selection on the same seed-7 synthetic corpus, not a correction of those numbers.

**Read this first.** [`docs/TIEBREAK_PROTOCOL.md`](docs/TIEBREAK_PROTOCOL.md) was committed alone (`b3dc679`) before anything was scored under it, and the runner refuses a seed-7 run if it is uncommitted or edited. But the same AI agent that wrote the rule had already read the 6 November tie-break diagnostic, so **it already knew the likely direction of this result: under any popularity-neutral tie rule, genre content was expected to finish above text.** This reselection confirms an expected direction rather than discovering one, and it should be read as the correction of a known artefact, not as independent evidence that genre content is the better model. There is no independent reviewer. The mitigations are only those the protocol lists: the rule was fixed before the first scored run, it is popularity-neutral by construction rather than tuned, one shared helper applies it identically to all eight methods, every possible winner was handled in advance, and **the first scored run is the one reported** — nothing was retuned after seeing it.

**The rule (TB-1).** Candidates are ordered by descending score exactly as before; items whose scores are *exactly equal as floats* are ordered by `blake2b("p04-gamerec/tiebreak/v1|" + p + "|" + item_id)` read as a big-endian integer; and a reported value is the mean over the 20 declared permutations `p = 0..19`, each user's metric being averaged over permutations (and, for the neural methods, over the five model seeds) before the mean over users. The key cannot see popularity or item-ID order: its mean absolute Spearman correlation with popularity rank was 0.102 and with item index 0.081 across the 20 permutations. `recstack/models.py` is untouched, so every earlier lowest-ID number stays reproducible.

| Validation NDCG@10 (100 users) | TB-1, mean over 20 permutations (95% CI over users) | Same model, lowest-ID rule |
|---|---:|---:|
| **Genre content (TB-1 winner)** | **0.3703 (0.3175–0.4219)** | 0.3703 |
| LambdaRank re-ranker, out of fold | 0.3693 (0.3183–0.4261) | 0.3679 |
| Item cosine kNN | 0.3533 (0.2997–0.4031) | 0.3533 |
| Two-tower | 0.3516 (0.3005–0.4068) | 0.3531 |
| Description TF-IDF | 0.3096 (0.2680–0.3522) | **0.3875** |
| BPR | 0.3080 (0.2574–0.3625) | 0.3080 |
| Popularity | 0.2176 (0.1709–0.2661) | 0.2176 |
| Implicit ALS | 0.1603 (0.1223–0.2026) | 0.1603 |

Only text moves materially: it is the only method with many exact ties (8,108 of 9,480 adjacent score pairs). Genre content, item kNN, BPR, popularity and ALS have no exact ties, so TB-1 leaves them bit-for-bit unchanged. The winning margin over the re-ranker is **0.0010**, far inside the user interval, and the re-ranker fits validation labels while the other seven methods do not, so its row is out of fold and not comparable on equal terms.

**Separate spreads, not one narrow number.** For the two-tower the SD between the five model seeds was 0.0143 and the SD between the 20 tie permutations 0.0018; for the re-ranker, 0.0110 and 0.0005; for text, 0 between seeds (it has none) and 0.0114 between permutations. Averaging over permutations narrows the permutation spread only, and the user interval above is what carries the real uncertainty.

**The grid choices did not change.** Both grids were rescored in full under TB-1 — the two-tower over the same 12 settings and the re-ranker over the same 8 — and both selected the same entry as on 6 November: two-tower `dim` 32, `epochs` 5, `logq` on; re-ranker `plan` features, 7 leaves, 50 rounds. Only the cross-method winner moved.

**What TB-1 does not fix.** It removes the popularity prior that entered through *tie-breaking* only. The genre content scorer is `genre_count + 0.001 * popularity`, so it carries an explicit popularity term **inside its own score** and has zero ties, which TB-1 cannot touch. So part of the reason genre content now finishes above text is a popularity prior that genre content declares in its scoring function and text does not have. Under TB-1, single-digit (popular) item IDs filled 7.8% of text's top-10 lists and 7.0% of genre content's, against 34.9% for the two-tower and 33.3% for the re-ranker.

**One descriptive paired comparison** (validation, 2,000 user resamples). Genre content minus text was **+0.0608 (+0.0269 to +0.0966)** under TB-1 (42 users better / 36 worse / 22 tied) and −0.0172 (−0.0354 to +0.0021) under the lowest-ID rule (19 / 34 / 47). This is computed on the split that selected the winner, so it is optimistic by construction and is **not** a hypothesis test; it is shown only to indicate that the reselection moved the ranking by more than user-level noise.

**Test reading — a known fixture, not a holdout.** These 97 test users were scored and read on 6 November, before this protocol existed. The numbers below have no role in selection, no acceptance rule and no claim of generalisation.

| Model | Test NDCG@10 under TB-1 (95% CI), known fixture | Same model, lowest-ID rule |
|---|---:|---:|
| Two-tower | 0.4357 (0.3768–0.4908) | 0.4367 |
| LambdaRank re-ranker | 0.4273 (0.3747–0.4783) | 0.4292 |
| Genre content | 0.4026 (0.3446–0.4662) | 0.4026 |
| Item cosine kNN | 0.3854 (0.3223–0.4492) | 0.3854 |
| BPR | 0.3598 (0.2957–0.4282) | 0.3598 |
| Description TF-IDF | 0.3112 (0.2687–0.3533) | 0.3979 |
| Popularity | 0.2510 (0.1942–0.3196) | 0.2510 |
| Implicit ALS | 0.1741 (0.1297–0.2111) | 0.1741 |

Validation and this fixture still disagree: validation selects genre content, while the two-tower has the highest observed fixture value. Nothing was reselected from the fixture.

**Older results stay reproducible.** The same run recomputed every lowest-ID validation number alongside the TB-1 ones, and the recomputed leaderboard reproduces the 6 November table exactly (text 0.3875, genre content 0.3703, re-ranker 0.3679, item kNN 0.3533, two-tower 0.3531, BPR 0.3080, popularity 0.2176, ALS 0.1603). `make all` and `make deep` are unchanged.

**Carry-forward.** Genre content is the model this project would carry forward from here on validation evidence, with the caveats above. Nothing was promoted into the M7 registry under this protocol; the registry champion is out of scope for it.

**Evidence:** [`evidence/m3_tiebreak_reselection.json`](evidence/m3_tiebreak_reselection.json), [`evidence/m3_tiebreak_test_fixture.json`](evidence/m3_tiebreak_test_fixture.json). Reproduce with `make tiebreak`.

## Pipeline, gate and simulated monitoring, plan M7 (synthetic, 6 November 2026)

[`docs/M7_PROTOCOL.md`](docs/M7_PROTOCOL.md) was committed alone (`94e6bbc`) before the first gate decision or monitoring number on the reported corpus. This is a plain-Python local pipeline with a file-based registry. **It is not a deployed service, and no MLflow or Airflow run took place.**

**Gate demo** (`evidence/m7_pipeline_demo.json`; validation NDCG@10 only, tolerance 0.005, test never read):

| Stage | Candidate NDCG@10 | Gate reason | Outcome | Champion | Champion files unchanged |
|---|---:|---|---|---|---|
| First champion, seed 11 | 0.3375 | `first_champion` (floor 0.2176) | Promoted, API reloaded | none → v0001 | n/a |
| Identical rerun, seed 11 | 0.3375 | `within_tolerance_of_champion` | No change | v0001 | Yes |
| Shuffled scores (fault injected) | 0.0765 | `below_champion_minus_tolerance` | **Blocked** | v0001 | **Yes** |
| Evaluation raises (fault injected) | none | `evaluation_failed` | **Blocked** | v0001 | **Yes** |
| Retrain, seed 23 | 0.3407 | `within_tolerance_of_champion` | Promoted, API reloaded | v0001 → v0002 | n/a |

The API in that table is the in-process FastAPI test client. Separately, one loopback `uvicorn` check (`evidence/m7_loopback_reload.json`) returned 409 for `method=champion` before reload, then served v0002 after `POST /admin/reload`. Tests also cover direct `register` refusal after a failed gate, a stale gate file, a corrupt artifact refused at reload with the old table still serving, a registry from another corpus, a non-local client, and a non-zero gate exit code. As a check on the tests, temporarily disabling the threshold comparison made 4 tests fail; the code was then restored. Between-seed SD of the selected two-tower on validation was 0.0137, larger than the 0.005 tolerance, so a retrain could be blocked by seed noise alone.

**Airflow.** `dags/gamerec_pipeline.py` passes a `DagBag` import and task-order test under Airflow 2.10.5 in a separate environment (`make dag-check`, 1 test). **The DAG has never been executed by a scheduler.** The same test is skipped in the main environment.

**Monitoring (`evidence/m7_monitoring_simulated.json`) — SIMULATED weeks, authored drift.** Frozen champion v0002, 8 simulated weeks, 5 simulation seeds, across-seed means. Drift and control share their random draws, so both have the same active users each week: 82–104 per seed-week, with 202–269 events (mean 240, below the protocol's rough estimate of 300). Recall denominators per seed-week are 52–72 users in control and 64–97 under drift, because drifted users play more games they do not already own.

| Monitor | Control range, weeks 1–8 | Drift, week 1 | Drift, week 8 | First alert week |
|---|---:|---:|---:|---:|
| Item popularity PSI | 0.4532–0.5490 | 0.6303 | 2.4070 | 1 |
| Item popularity JS (bits) | 0.0642–0.0756 | 0.0881 | 0.3203 | 1 |
| Served top-10 score PSI | 0.0021–0.0044 | 0.0023 | 0.0025 | **Never** |
| Played-pair score PSI | 0.0290–0.0437 | 0.0677 | 2.0776 | 1 |
| Recall@10 | 0.5828–0.6267 | 0.5546 | 0.2116 | 1 |

- **PSI's rule of thumb fails here.** Item-level popularity PSI was above 0.25 in all 8 control weeks. With about 240 events spread over 80 items, sampling noise and the gap between training and weekly plays cross it with no authored drift at all. Only the protocol's control-referenced rule makes the alerts in this table usable.
- **Served-score PSI is blind to this drift.** It is identical in both scenarios: a frozen score table and the same active users give the same top-10 scores, whatever those users now want. This is a limit of the frozen-table setup, and it means this label-free monitor would not have caught the drift.
- **Recall@10 fell from 0.5546 to 0.2116 under drift**, and the control stayed between 0.5828 and 0.6267 with no alerts. Week 8 across seeds ran 0.173–0.237 under drift and 0.518–0.659 in control. The week-1 recall alert is marginal: 0.5546 against a limit of 0.5552.
- **Limits.** The same agent wrote the drift and the monitors, so detection only shows the code responds to this drift; it does not validate these thresholds on real traffic. Alerts do not trigger retraining. There is no exposure log, so "served" scores are what the table would serve, not recorded impressions. Played-pair PSI and recall need outcomes, which arrive late in a real system. Week numbers have no calendar meaning.

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

On 6 November 2026, the first `make all` before new work passed 31 tests. After the M3 and M7 work, `make all` passed **89 tests with 1 skipped** (the Airflow import test, which passes in its own environment through `make dag-check`). The regenerated `runs/synthetic_baselines.json` and `runs/experiment_baseline.json` kept the same SHA-256 hashes as before the work, so the baseline and experiment numbers above did not change. `make bench` was not rerun; its table is from the earlier run. No real record was downloaded, and the licence decision stays with the project owner.
