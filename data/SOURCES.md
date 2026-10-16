# Data provenance and rights

## Seeded synthetic data used in all current measurements

`recstack/data.py` generates 120 fictional users and 80 fictional games with a five-genre preference process and a long item tail. Seed 7 gives 3,000 interactions before splitting. No individual or game is real. The generator is an evaluation fixture; it cannot establish generalization to real players.

## Planned public sources, not downloaded in this run

- The [UCSD McAuley Lab Steam dataset page](https://mcauleylab.ucsd.edu/public_datasets/data/datasets.html) describes Steam reviews, plays and item metadata, and its [download index](https://mcauleylab.ucsd.edu/public_datasets/data/steam/) lists the files. The site asks for a citation but does not state a reuse licence on the pages inspected on 11 October 2026. The user-item playtime file may not provide per-interaction timestamps; a temporal-split experiment must confirm dates from a suitable interaction file before use. The large files were not downloaded or republished.
- The [Kaggle Mobile Games: A/B Testing data card](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing) identifies the Cookie Cats CSV as deriving from a DataCamp project and lists its licence as **Unknown** on the page inspected on 11 October 2026. No CSV is bundled or analysed here. The statistical library can accept a separately obtained local file once its use and provenance are settled.
