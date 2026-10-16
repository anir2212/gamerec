# Iteration log

| Change | Hypothesis | Measured result | Decision |
|---|---|---|---|
| Keep a long-tail synthetic item distribution (`1 / rank^1.15`) | A uniform-ish catalog leaves too few low-count items to inspect cold-start behavior. | The earlier `1 / sqrt(rank)` run had 1 item at ≤5 train interactions and 5 cold-item test events. The revised seeded run had 22 such items and 27 test events. | Keep the revised fixture for diagnostics; it remains synthetic. |
| Add item cosine kNN | Neighbour structure should beat generic popularity for known users. | Test NDCG@10 0.3854 versus popularity 0.2510 on 97 scored users. | Keep as a strong baseline. |
| Add fixed-setting ALS and BPR | Compare factorization to item neighbours. | ALS NDCG@10 0.1741; BPR 0.3598. | Keep for comparison; tune on validation before any model-selection claim. |
| Add genre-content baseline | A user preference prior may help low-count items. | Validation selected it; test overall NDCG@10 0.4026. Cold-item NDCG@10 0.2169 versus kNN 0.2170 on 21 users. | Keep as a baseline; no robust cold-item NDCG gain is shown. |
| Add declared-genre unknown-user fallback | A user with no history should avoid an unpersonalized empty response. | FastAPI integration test returned five unique items with the declared genre first. | Keep and later compare with a real cold-user cohort. |

No LLM-suggested feature, MLflow run, or automatic promotion was used in this checkpoint.
