# Tie-break protocol TB-1: fair ranking of exact score ties, and a validation-only reselection

Committed alone on 6 November 2026, before any model was scored under this rule. It governs one
rerun of M3 model selection on the **seed-7 synthetic fixture**: 120 fictional users, 80 fictional
games, 3,000 interactions, split 1,808 / 596 / 596 by the per-user temporal protocol. **All data is
synthetic and seeded. No real record was downloaded, imported, trained on or measured**; the licence
decision on real Steam data stays with the project owner (`data/SOURCES.md`).

This protocol does not replace [`M3_PROTOCOL.md`](M3_PROTOCOL.md) and does not change any number it
produced. The 6 November M3 selection keeps its own lowest-ID rule, its own table and its own date.

## Who chose these settings, and what they had already seen

An AI coding agent (Claude) chose every setting in this file. There is no independent reviewer and
no second analyst. Before writing it, the agent had read:

- the full 6 November M3 selection and comparison evidence, including every validation and test
  number for all eight methods, so **test is a known, already-inspected fixture, not a holdout**;
- the post-hoc tie-break diagnostic in `evidence/m3_test_comparison.json`, which shows that the
  selected **text** model scored 0.3875 on validation under lowest-ID ties but a mean of **0.3159**
  under random ties, while genre **content** scored 0.3703 under either rule because it has no ties.

**So this reselection is designed and run by an author who already knows the likely direction of its
result: under any popularity-neutral tie rule, genre content is expected to finish above text.** That
is the strongest limitation of what follows. A rule chosen after seeing which rule changes the
ranking is not a blind test, and this one cannot be: the finding that prompted it is what identified
the defect. The disclosure is not a formality — it means the reselection confirms an expected
direction rather than discovering one, and it should be read as a correction of a known artefact, not
as independent evidence that genre content is the better model. The mitigations are only these: the
rule is fixed in this file before any scored run, the rule is popularity-neutral by construction
rather than tuned, it is applied identically to every method by one shared helper, the decision rule
and the handling of every possible winner are written down below, and the first scored run is the one
reported.

The agent also ran, before committing this file, a non-scoring count of tied adjacent score pairs per
baseline on the seed-7 training-fitted score tables, to decide how "equal" should be defined. That
count is a property of the fixture's score arithmetic, not a model result: exact float equality and a
1e-12 tolerance gave **identical** tie counts for all six baselines (text 8,108, item kNN 144, and 0
for popularity, ALS, BPR and genre content, out of 9,480 adjacent pairs). Exact equality is therefore
used below, with no tolerance parameter to tune.

## The declared tie rule (TB-1)

For a scorer's score vector over the catalog for one user:

1. Candidates are ordered by **descending score**. Items already in the user's training history are
   removed after ordering, exactly as `Recommenders.recommend` does today.
2. Items whose scores are **exactly equal as floats** are ordered by an item key that is independent
   of item ID order, popularity and every model:
   `key(item, p) = blake2b(b"p04-gamerec/tiebreak/v1|" + str(p) + "|" + item, digest_size=8)`
   read as a big-endian unsigned integer, where `p` is a tie-break permutation seed. The salt string
   is fixed by this protocol and must not be changed.
3. A model's reported value for a metric is the **expectation of that metric over a declared
   permutation set**: `p` takes the 20 values `0, 1, ..., 19`. Each user's metric is averaged over
   the 20 permutations first; the point estimate is then the mean over users.

TB-1 is deterministic (the same inputs give the same number on any machine, with no RNG draw order
to preserve), popularity-neutral (the key cannot see popularity, and over 20 permutations no item is
systematically favoured), model-independent (the same key orders ties for every scorer), and
low-variance (an average over 20 permutations rather than one arbitrary permutation).

### Alternatives considered and rejected

| Rule | Why not |
|---|---|
| **Lowest item ID** (the current rule) | The generator makes low IDs popular, so it injects a hidden popularity prior into any scorer with many ties. This is the defect being corrected. |
| **Highest item ID** | Equally arbitrary, and simply inverts the same artefact. |
| **A single hash permutation** (one `p`) | Deterministic and popularity-neutral, but with 8,108 of 9,480 adjacent text pairs tied, one permutation is a single draw from a wide distribution: the diagnostic's 20 random draws spanned 0.2991–0.3296 on validation. One draw would make the headline number depend on an arbitrary salt. |
| **One random shuffle per evaluation, unseeded** | Not reproducible, so no saved evidence could be rechecked. |
| **Random ties with a fixed seed set** (what the diagnostic did, by adding small uniform noise) | Statistically the same estimand as TB-1 and acceptable, but it reaches it by perturbing scores, so it needs a noise scale derived from the smallest positive score gap. That scale is a fragile implementation detail that can silently reorder genuinely distinct scores. TB-1 orders ties by an exact key and cannot reorder distinct scores at all. |
| **Breaking ties by ascending popularity** | Popularity-neutral in neither direction; it would flatter novelty metrics instead. |
| **Averaging over all permutations analytically** (expected NDCG under uniform random tie order) | Exactly correct and deterministic, but it needs per-metric closed forms for NDCG, recall, MAP, hit rate, novelty and coverage, including partially-tied blocks that straddle the cut at K. The 20-permutation mean is the same estimand with one shared code path for every metric. |
| **Removing the ties instead** (e.g. adding a popularity term to the text scorer, as genre content has) | That changes a model, not a ranking convention, and it would make the comparison worse rather than fairer. Scorers are left exactly as they are. |

### What TB-1 does not fix

TB-1 removes the popularity prior that entered through *tie-breaking*. It does not make the
comparison popularity-neutral overall. In particular the genre **content** scorer is
`genre_count + 0.001 * popularity`, so it carries an explicit popularity term **inside its score**
and has zero exact ties. TB-1 therefore cannot change its number at all, while it removes an
accidental popularity advantage from text. If genre content finishes above text under TB-1, part of
the reason is a popularity prior that genre content declares in its scoring function and text does
not have. That must be stated wherever the outcome is reported.

## Implementation: one shared helper, no per-model code

The rule lives in **`recstack/tiebreak.py`** and is the only ranking path used by this reselection:

- `order_key(item_ids, permutation_seed)` builds the key array in step 2.
- `TieBrokenModels(base, tables, order_key)` subclasses `recstack.score_table.ScoredModels` and
  overrides `recommend` with steps 1 and 2. Every method is ranked by this one override.
- Baselines are wrapped with **no** score table, so their scores still come from
  `Recommenders.scores` and only the tie ordering changes. Neural methods are wrapped with their
  precomputed user-by-item score table. There is no per-model tie code and no per-model branch.
- `recstack/models.py::Recommenders.recommend` is **not modified**, so every earlier result,
  including the committed split manifests, `runs/synthetic_baselines.json` and both `evidence/m3_*`
  files, stays reproducible under the lowest-ID rule it was computed with.

The runner is **`recstack/tiebreak_run.py`**, writing **`evidence/m3_tiebreak_reselection.json`**,
with `make tiebreak`. It will be committed before it is run scored. Like the M3 driver, it refuses a
seed-7 run if this protocol file is uncommitted or edited, and it records this file's SHA-256 and
first commit in its output.

## Metric

**Expected validation NDCG@10 under TB-1**, as defined above: mean over users of each user's mean
over the 20 permutations and, for the neural methods, over the five model seeds as well. Eligible
users are unchanged from M3 (`recstack/eval.py`): a user is scored if it has at least one validation
play that is not already in its training history. That was 100 users on validation.

Intervals are 95% percentile bootstrap intervals over **users** (400 resamples, seed 17) of those
per-user expectations. For the neural methods, the spread between model seeds and the spread between
tie-break permutations are reported separately, so the permutation average is not mistaken for
narrower uncertainty than it is. Recall, MAP, hit rate, novelty and catalog coverage at 10 and 20 are
recorded in the evidence file as well, but the selection metric is NDCG@10 only.

## The exact models in the comparison

The same eight methods as the 6 November M3 selection, fitted by the same unchanged code:

1. `popularity`, 2. `itemknn`, 3. `als`, 4. `bpr`, 5. `content` (genre), 6. `text` (description
TF-IDF) — all from `recstack/models.py` at model seed 7, with no tuning, exactly as before;
7. `two_tower` — the PyTorch model in `recstack/two_tower.py`, reselected over the **same 12-setting
grid** as M3 (`dim` {16, 32} x `epochs` {5, 10, 20} x `logq` {off, on}), because a fair tie rule can
move which grid entry wins;
8. `reranker` — the LightGBM LambdaRank model in `recstack/reranker.py`, reselected over the **same
8-setting grid** (`features` {plan, plus} x `num_leaves` {7, 15} x `rounds` {50, 150}), scored by the
same 5-fold out-of-fold construction over validation users, with the TB-1-selected two-tower
supplying its feature.

The re-ranker fits on validation labels and the other seven methods do not; its validation row is out
of fold, and that asymmetry is restated wherever its number appears. The `two_tower_id_only` and
`two_tower_content_only` ablations are not part of the selection and are not reselected.

## Seeds

| Role | Value |
|---|---|
| Corpus | 7 |
| Baseline model seed (ALS, BPR) | 7 |
| Neural model seeds | 11, 23, 37, 41, 53 |
| Tie-break permutations | 0, 1, ..., 19 |
| Bootstrap | 17, 400 resamples (2,000 for the paired comparison) |

## Selection reads validation only

`tiebreak_run.select` receives the train and validation splits only; the test split is not passed to
it. The evidence file records `test_split_read_by_selection: false` and the validation split hash.
Every hyperparameter choice and the final winner come from validation alone.

## Decision rule, and what happens for each possible winner

**The winner is the method with the highest expected validation NDCG@10 under TB-1.** Exact ties
inside a grid keep the earlier grid entry; an exact tie across methods is broken alphabetically.
There is no other criterion, no threshold and no tolerance band.

Declared in advance, for every outcome:

- **If `content` (genre) wins**, that is the expected direction given what the author had already
  seen (disclosed above). It is reported as the TB-1 outcome, with the disclosure and with the note
  that genre content's own score contains a popularity term that TB-1 does not touch.
- **If `text` wins anyway**, that is reported too, and it means the lowest-ID artefact did not
  decide the 6 November selection after all.
- **If any other method wins** — including a two-tower or re-ranker setting that the lowest-ID rule
  had placed below text — that is the finding and it is reported as it stands.
- In every case the **first scored run is the reported run**. No setting, grid, seed, permutation
  count, salt or metric changes after seeing it, and there is no second scored selection. If a run
  fails on a bug, the bug, the discarded output and the corrected result are all recorded.
- A change of winner **does not** mean the 6 November selection was done wrongly. That rule was
  fixed in advance of its own scored run, which is the property that made this defect findable. The
  6 November table stays in `RESULTS.md`, dated, with its lowest-ID rule named, and the TB-1 result
  is added beside it.
- The TB-1 winner is the model this project would carry forward from here. Nothing is promoted into
  the M7 registry under this protocol; the registry champion is out of scope.

## Test is a known fixture

The 97 eligible synthetic test users were scored and read on 6 November, before this protocol
existed. **Any test number under TB-1 is a reading on a known, already-inspected fixture, not a
held-out estimate**, and must be labelled that way every time it appears. Test is reported for all
eight methods for completeness only; it has no role in selection, no acceptance rule attached, and no
claim of generalisation. Nothing in this protocol licenses a conclusion about real players: the
corpus is synthetic, the generator rewards repeated genre taste by construction, and the item copy is
a template.

## One descriptive paired comparison

After the winner is fixed by the rule above, the evidence file records a paired user-level bootstrap
(2,000 resamples, seed 17) of **TB-1 winner minus `text`** on validation NDCG@10, and the same
difference under the old lowest-ID rule. This is **descriptive only**: it is computed on the split
that selected the winner, so it is optimistic by construction and is not a hypothesis test. It is
recorded to show whether the reselection moved the ranking by more than user-level noise, not to
claim significance.

## Reproduce

`make tiebreak` runs `recstack.tiebreak_run --phase select`, which writes the reselection to
`evidence/m3_tiebreak_reselection.json`, and then `--phase fixture`, which writes the labelled
known-fixture test reading to `evidence/m3_tiebreak_test_fixture.json`. Only the `select` phase
decides anything; the `fixture` phase refuses to reuse a selection made under a different protocol
hash. `make all` and `make deep` are unchanged and keep producing the earlier lowest-ID numbers.

**Amendment, 6 November 2026, before any scored TB-1 run.** This Reproduce section originally named
only `--phase select`. It was amended, in its own commit and still before anything was scored under
TB-1, to name the second phase and its output file, which the "Test is a known fixture" section above
had already required. The rule, the metric, the models, the seeds, the validation-only constraint and
the decision rule are untouched by that amendment. No scored TB-1 number existed when it was made.
