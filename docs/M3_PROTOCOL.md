# M3 protocol: two-tower and LambdaRank on the synthetic corpus

Committed alone on 6 November 2026, before any new model was scored on the seed-7 validation or test split. All data is the **seed-7 synthetic fixture**: 120 fictional users, 80 fictional games, 3,000 interactions. No real records are used; the licence decision on real Steam data stays with the project owner.

## Who chose these settings and what they had seen

An AI coding agent (Claude) chose every setting here, with no independent reviewer. It had already read all six baseline test results in `RESULTS.md`, so **test is a known fixture, not an untouched holdout**. It also ran the code on a different corpus (seed 11, validation and test) to check runtime and bugs. That run saw overfitting past about 10 epochs, which set the epoch range below. Seed-7 validation and test had not been scored with any new model when this file was committed.

## Models

Fixed code is at the parent commit: `recstack/two_tower.py`, `recstack/reranker.py`, `recstack/deep_run.py`.

**Two-tower (PyTorch).** Item tower: ID embedding + linear projection of description TF-IDF + genre embedding, then an MLP and L2 normalisation. User tower: confidence-weighted mean of item-tower vectors from the user's earlier plays, a recency-weighted mean of the same vectors (decay 0.7 per play), an activity-bucket embedding (at most 3, at most 10, more plays) and a projection of genre shares, then an MLP and L2 normalisation. Examples are novel plays by a user with at least one earlier training play; history is the earlier rows only. Loss is in-batch sampled softmax, with duplicates of the target and the row user's owned games removed from the negatives. Items with at most five training events have their ID embedding masked, in training and scoring.

Fixed: temperature 0.2, batch size 64, Adam learning rate 0.005, weight decay 1e-5, ID dropout 0.1 on warm items, one CPU thread.

**Ablations at the selected two-tower settings (not separately selected).** `two_tower_id_only`: no text or genre, no cold mask. `two_tower_content_only`: no ID channel for any item.

**LambdaRank (LightGBM).** One query per validation user with a novel target; candidates are all unowned catalog items; label 1 for a novel validation game. Base models fit on training rows only, so scores and labels share no rows. `plan` features: ALS, two-tower, log popularity, genre match, recent genre match (last three plays). `plus` features add item kNN, BPR and text scores. Fixed: learning rate 0.05, `min_data_in_leaf` 20, truncation 10, deterministic, one thread.

## Declared deviations from the plan

| Plan text | Done here | Why |
|---|---|---|
| Sentence-transformers text embeddings | TF-IDF of the generated description | The copy is "A <genre> game with variant <n>". A learned encoder would embed a template, and the model and its dependencies would use disk for no information. |
| Description + tags | Description only, bare numbers removed for the tower | The corpus has no tags. A variant number occurs once, so a projection could use it as a hidden item ID. After removal, **content carries genre-level information only**: same-genre cold items score identically. |
| Top-200 candidates | All unowned items, at most 80 | The catalog has 80 items, so there is no retrieval stage to test. |
| Price match | Dropped | The corpus has no price field. |
| Item recency | User's recent-genre match; recency-weighted history | The corpus has no release date. The generator draws each play independently of order, so recency has **no signal by construction** and is expected to be inert. |
| User segment | Behavioural: activity bucket and genre shares from earlier plays | The corpus has no segment field. |
| FAISS, MLflow | Not used | Out of M3 scope here. |

The re-ranker sees validation labels at fit time and the six baselines do not. This is stated beside its results.

## Validation-selected search space

Selection reads validation only (`deep_run.select` never receives the test split). Every setting uses five neural seeds: **11, 23, 37, 41, 53**.

- Two-tower, 12 settings: `dim` {16, 32} x `epochs` {5, 10, 20} x `logq` {off, on}.
- Re-ranker, 8 settings: `features` {plan, plus} x `num_leaves` {7, 15} x `rounds` {50, 150}, with the selected two-tower supplying its feature, scored by 5-fold out-of-fold validation users.

**Selection rule.** Highest mean validation NDCG@10 over seeds; equal values keep the earlier grid entry. No other criterion. The validation winner across the eight methods (six baselines, two-tower, re-ranker) is the method this project would carry forward. A re-ranker validation score is out of fold; other validation scores involve no validation-label fit.

## Metrics and intervals

Same `recstack/eval.py` metrics as the baselines: Recall, NDCG, MAP, hit rate, novelty and catalog coverage at 10 and 20, on `all`, `cold_item` and `new_user`, plus the complements `warm_item` and `warm_user`. Baselines keep model seed 7. Neural results are the mean over the five seeds, with a 95% percentile bootstrap over users (400 resamples) after averaging each user over seeds, and the between-seed SD reported separately.

Paired test comparisons, 2,000 user resamples on NDCG@10:

1. two-tower minus text, overall (text is the earlier validation winner)
2. re-ranker minus text, overall
3. re-ranker minus two-tower, overall
4. two-tower minus item kNN, cold item
5. content-only minus ID-only, cold item
6. hybrid minus ID-only, cold item
7. ID-only minus content-only, warm item
8. item kNN minus genre content, warm user

## How outcomes will be read

- A method is "better" on this fixture only if the paired 95% interval excludes zero; otherwise the report says no difference was shown. Eight intervals are read together with no multiplicity correction, so one false exclusion would not be surprising.
- "Content helps cold items" needs comparison 5 or 6 to exclude zero in favour of content. "CF wins warm" needs comparison 7 or 8 to exclude zero in favour of CF. Neither is promised.
- The cold-item slice had **21 users** in the baseline report. It can show a large gap on this fixture and little else. All content here is genre-level by construction, and the generator rewards genre by construction, so no result here transfers to real games.
- **A loss by the deep models is a valid outcome and will be reported as it stands.** After the first compare run on seed 7, no setting, seed or grid changes and no reselection from test. Any fix needed for a bug will be recorded with both results.

## Diagnostic declared in advance

`recommend` breaks exact score ties by lowest item ID, and the generator makes low IDs popular, so ties may act as a hidden popularity prior. The compare phase recomputes test NDCG@10 for text, genre content and the content-only two-tower with exact ties broken at random (20 shuffles) and records the top-10 share of single-digit IDs. This is a diagnostic only and does not change the main table.

## Reproduce

`make deep` runs `recstack.deep_run --phase select`, writing `evidence/m3_selection.json`, then `--phase compare`, writing `evidence/m3_test_comparison.json`. For seed 7 the driver refuses to run if this file is uncommitted or edited, and compare refuses a selection made under a different protocol hash.
