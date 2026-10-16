# Offline proxy decision memo — 11 October 2026

**Decision:** No product launch decision. The measured assignment compares ALS against a genre-content scorer using a held-out game hit as a proxy for a click. It has no exposure, purchase, retention, or live guardrail data.

The synthetic proxy assigned 54 users to ALS and 43 to genre content. Hit rates were 50.0% and 81.4%, respectively; the observed difference was +31.4 percentage points (bootstrap 95% CI +13.35 to +48.06 points, two-sided pooled z-test p = 0.00138). The allocation SRM check gave p = 0.264. These calculations are reproducible with `python -m recstack.experiment_run`, but the small sample and generated preference process make the effect unsuitable for a real product claim.

A real test would pre-register the primary metric, sample size, guardrails, and fixed analysis time; verify exposure logging and SRM; and avoid repeated uncorrected daily significance checks. The separate null simulation showed 22.6% of 500 runs crossing p < 0.05 at least once over ten daily looks versus 7.4% at the one final look. The exact fractions depend on the simulation seed and size.
