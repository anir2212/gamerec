# M7 protocol: local pipeline gate and simulated monitoring

Committed alone on 6 November 2026, before any gate decision or monitoring number was produced. An AI coding agent (Claude) chose these settings with no independent reviewer, after seeing the M3 validation and test results. Everything runs on the **seed-7 synthetic corpus and authored simulated weeks**. There is no deployed service, exposure log or live experiment.

## Promotion gate

- **Metric:** validation NDCG@10 from `recstack/eval.py`. The gate never reads test.
- **Rule:** promote only if candidate NDCG@10 >= champion NDCG@10 - **0.005**.
- **No champion yet:** promote only if the candidate is >= the popularity baseline's validation NDCG@10.
- **Fail closed:** block a missing, non-finite or out-of-range metric, an evaluation step that raised, zero scored users, a different metric name or split, or an evaluation row hash that differs from the champion's (scores on different rows are not comparable).
- **Candidates:** ALS and two-tower (M3-selected settings: `dim` 32, `epochs` 5, `logq` on). The better on validation is the candidate.
- **Acceptance:** tests show a failing candidate leaves `champion.json` and the champion's files byte-identical and the API serving the same version. Corrupt artifacts are refused at reload and the old version keeps serving.

Tolerance 0.005 is a judgment call, smaller than the between-seed SD seen in the M3 grid (about 0.009 to 0.018). So a retrain of the same settings can be blocked by seed noise alone. That is reported, not tuned away.

## Simulated weeks

A frozen champion score table from the pipeline is monitored over **8 simulated weeks**. Simulation seeds: **101, 102, 103, 104, 105**. Each week every user is active with probability 0.8 and draws 1 to 4 plays, using the fixture's generator: own genre with probability 0.74, item weight `1/(1+index)^1.15`.

| Scenario | Definition at week w (1 to 8) |
|---|---|
| `no_drift` (control) | The generator above, unchanged |
| `drift` | Taste: a share `min(1, 0.1 w)` of users, chosen by a seeded shuffle, move to the next genre. Popularity: each draw uses reversed item weights `1/(1+(N-1-index))^1.15` with probability `0.08 w` |

## Monitors

- **Item popularity drift:** PSI (share smoothing 1e-4) and Jensen-Shannon divergence in bits between training event shares and the week's event shares across all catalog items.
- **Score drift, served:** PSI of the top-10 scores for that week's active users against the top-10 scores for all users, over 10 reference-quantile bins.
- **Score drift, played:** PSI of champion scores on the week's (user, game) plays against validation-row plays, over 10 reference-quantile bins. This one needs outcomes.
- **Recall decay:** Recall@10 of the frozen table against each user's novel games that week, users with at least one novel game, with a user bootstrap interval (400 resamples) per scenario and seed.

## Alert rules

- PSI 0.1 and 0.25 are industry rules of thumb, not calibrated limits, and are reported as such. With about 300 weekly events across 80 items, sampling noise alone may exceed them, which the control measures.
- **Drift alert:** a monitor alerts in a `drift` week only if its across-seed mean exceeds the highest across-seed weekly mean in `no_drift`.
- **Recall alert:** across-seed mean Recall@10 is below the `no_drift` all-week mean by more than **0.05**.
- First alert week is reported per monitor. Alerts do not trigger retraining in this code.

## How outcomes will be read

The drift scenario was authored by the same agent that wrote the monitors, so the answer is known in advance. A detection shows that the code responds to this drift. It does not validate these thresholds on real traffic. If a monitor stays silent or alarms in the control, that is reported and the settings are not changed after the first run.
