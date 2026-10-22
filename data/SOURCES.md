# Data provenance and rights

## Seeded synthetic data used in all current measurements

`recstack/data.py` generates 120 fictional users and 80 fictional games with a five-genre preference process and a long item tail. Seed 7 gives 3,000 interactions before splitting. No individual or game is real. The generator is an evaluation fixture; it cannot establish generalization to real players.

Its contract is `SYNTHETIC_CONTRACT` in `recstack/contract.py`. The timestamp meaning is `synthetic_ordinal`: `event_index * 1000 + user_index`, a generated order rather than a clock time. The split manifests for this fixture are saved in [`data/manifests/`](manifests/). `make all` recomputes them and fails if they differ.

## Dataset contract for any future source

Before any real record is imported, a source needs a `DatasetContract` that states:

- the four required columns: `user` (nonempty string), `item` (nonempty string, present in the catalog), `time` (nonnegative integer), and `minutes` (nonnegative integer weight);
- what `time` means (`time_semantics`) and its unit. Only `interaction_event`, `review_publication` and `synthetic_ordinal` can support a temporal claim. `snapshot` (crawl or export time) and `none` cannot;
- what the weight column means, and the licence status.

`validate_records(..., temporal_claim=True)` rejects a contract that cannot order events. It also rejects any row with a missing or non-integer time, wrong types, an unknown catalog item, or a repeated `(user, item, time)` key. Without a temporal claim, rows with no time are skipped, and no temporal result may be reported from them. `recstack/manifest.py` then cleans and splits the rows and saves the manifest. The manifest records the contract hash, input hash, cut-offs, per-split counts and row hashes, and leakage checks.

## Real-source audit, 20 October 2026 (nothing downloaded)

### Chosen candidate: Kaggle "Game Recommendations on Steam"

Pages accessed on 20 October 2026: the [dataset page](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam), plus Kaggle's public metadata endpoints for [view](https://www.kaggle.com/api/v1/datasets/view/antonkozyriev/game-recommendations-on-steam), [metadata](https://www.kaggle.com/api/v1/datasets/metadata/antonkozyriev/game-recommendations-on-steam) and [file list](https://www.kaggle.com/api/v1/datasets/list/antonkozyriev/game-recommendations-on-steam). The HTML page returned only its title to the fetch tool, so the facts below come from the metadata endpoints.

| Question | Finding |
|---|---|
| Licence | Metadata lists `licenseName` "CC0: Public Domain" and licence "CC0-1.0". Owner `antonkozyriev`, version 28, last updated 2024-08-14. |
| Upstream rights | The description credits the "Steam Official Store" as the source and says Valve holds the rights to the thumbnail image. The CC0 dedication comes from the uploader. This audit did not establish that the uploader could waive rights in Steam store data or in user-written reviews, and it did not review Valve's terms. |
| Permitted use | CC0 places no limits on research, commercial use or redistribution. Because of the upstream question, this project treats it as **private research use only**: raw rows stay out of Git and out of any public snapshot. Downloading needs a signed-in Kaggle account or API token, which this run did not use. |
| Interactions | `recommendations.csv`, described as "a table of user reviews: whether the user recommends a product", is a user–game relation. The description says it holds over 41 million cleaned recommendations. |
| Event time | Kaggle returned **no per-column definitions**. Third-party projects use a `date` column, but its meaning and precision are not documented by the source. If a sample confirms it is the review posting date, the contract is `review_publication`. In that case a temporal result means "earlier reviews predict later reviews". It does not mean "earlier play predicts later play". Day-level dates would create ties. |
| Weight | Third-party projects describe an `hours` playtime column. It is not documented whether it was measured at review time or at crawl time. Treat it as a cumulative weight, never as an event time. |
| Item metadata | `games.csv`: ratings, USD pricing, release date. `games_metadata.json`: "descriptions and tags". This is the descriptive catalog copy the text baseline lacks today. |
| Personal data and deletion | The description says user IDs were anonymised and there is no personal information. But review ID, game, date and hours could still be matched against public Steam review pages, so treat rows as **pseudonymous**. The source states no deletion or takedown terms. Project rule: store only under the Git-ignored `data/raw/`, never commit or publish rows, and delete local copies when the experiment ends or if the upstream version is withdrawn. |
| Size | `totalBytes` 2,238,703,480 (about 2.24 GB): `recommendations.csv` 2,023,223,067 B, `users.csv` 193,068,018 B, `games_metadata.json` 17,548,605 B, `games.csv` 4,863,790 B. |

**Verdict: conditionally usable for a private portfolio experiment, not yet approved.** It is the best candidate inspected because it has an explicit licence label, item descriptions and tags, and a dated interaction table. It was **not downloaded**: it is over 100 MB, it needs Kaggle credentials, and the host had about 7 GiB free disk.

**Planned mapping (not implemented, no data loaded):** `user_id → user`, `app_id → item`, `date → time` (days since epoch, `review_publication`, only after a sample confirms it), and `hours → minutes` (cumulative weight). A run must use the `global` cut-off protocol or a predeclared tie rule, because the `per_user` split rejects equal timestamps across a boundary.

**Still to decide (project owner):** (1) whether to accept the uploader's CC0 label despite the Steam origin; (2) whether to create a Kaggle token and download about 2.2 GB, or only `games.csv`, `games_metadata.json` and a subset; (3) whether negative recommendations (`is_recommended = false`) count as interactions or as negative labels; (4) how to confirm the `date` meaning from a sample before any temporal claim.

### Also checked: UCSD McAuley Lab Steam data

Pages accessed on 20 October 2026: the [McAuley datasets page](https://cseweb.ucsd.edu/~jmcauley/datasets.html), the [lab public-datasets page](https://mcauleylab.ucsd.edu/public_datasets/data/datasets.html) and the [Steam download index](https://mcauleylab.ucsd.edu/public_datasets/data/steam/).

- **Licence:** no licence, terms of use, or statement on commercial use or redistribution appears on either page. The pages only ask users to cite the listed papers ("Please cite the appropriate reference if you use any of the datasets below").
- **Files and size:** `australian_user_reviews.json.gz` 6.6 MB, `australian_users_items.json.gz` 71 MB, `steam_reviews.json.gz` 1.2 GB, `steam_games.json.gz` 2.5 MB (item metadata), `bundle_data.json.gz` 90 KB, and larger crawl archives. The page reports 7,793,069 reviews, 2,567,538 users and 15,474 items.
- **Fields:** the pages do not document review or user–item fields. A [third-party tutorial](https://docs.shaped.ai/docs/v2/tutorials/steam_review) parses review `posted` strings such as "Posted November 5, 2011.", which would be a day-precision review date. No inspected source documents a per-interaction time in the user–item playtime file. That file must be treated as `none` until shown otherwise.
- **Verdict: not chosen.** There is no stated reuse licence. Even the 6.6 MB review file was not downloaded, because the terms do not clearly permit this use. Using it would need the project owner to accept citation-only academic terms, or to get permission from the lab.

### Cookie Cats A/B data (unchanged from 11 October 2026)

The [Kaggle Mobile Games: A/B Testing data card](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing) identifies the Cookie Cats CSV as deriving from a DataCamp project. It listed the licence as **Unknown** when inspected on 11 October 2026. No CSV is bundled or analysed here. The statistical library can accept a separately obtained local file once its use and provenance are settled.
