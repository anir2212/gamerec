"""Tie rule TB-1 from ``docs/TIEBREAK_PROTOCOL.md``: one shared ranking helper.

``Recommenders.recommend`` orders exact score ties by lowest item ID, and the synthetic
generator makes low IDs popular, so that convention injects a popularity prior into any
scorer with many ties. TB-1 replaces the convention without touching any scorer:

1. order by descending score, exactly as before;
2. order exact float ties by ``order_key``, a per-permutation hash of the item ID that
   cannot see popularity, item ID order or the model;
3. report the mean of a metric over a declared set of 20 permutations.

Every method in the reselection is ranked by :class:`TieBrokenModels`, so the rule is
written once and cannot differ between models. ``recstack/models.py`` is deliberately
left alone, so every earlier lowest-ID result stays reproducible.
"""

from __future__ import annotations

import hashlib

from recstack.models import Recommenders
from recstack.score_table import ScoredModels

# Fixed by the protocol. Changing it would change every tie order.
SALT = b"p04-gamerec/tiebreak/v1|"
PERMUTATIONS = tuple(range(20))


def order_key(item_ids: list[str], permutation_seed: int) -> list[int]:
    """Popularity-neutral deterministic tie key, one integer per catalog item."""
    return [int.from_bytes(hashlib.blake2b(SALT + f"{permutation_seed}|{item}".encode(),
                                           digest_size=8).digest(), "big")
            for item in item_ids]


class TieBrokenModels(ScoredModels):
    """A fitted model, optional score tables, and one declared tie-break permutation."""

    def __init__(self, base: Recommenders, tables: dict, keys: list[int]):  # noqa: D107
        super().__init__(base, tables)
        if len(keys) != len(self.item_ids):
            raise ValueError(f"tie key has {len(keys)} entries, expected {len(self.item_ids)}")
        self.tie_keys = list(keys)

    def recommend(self, user: str, method: str, k: int = 10) -> list[str]:
        """Rank by descending score, breaking exact ties by the declared key."""
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.scores(user, method)
        owned = self.history.get(user, set())
        order = sorted(range(len(self.item_ids)),
                       key=lambda i: (-float(scores[i]), self.tie_keys[i]))
        return [self.item_ids[i] for i in order if self.item_ids[i] not in owned][:k]


def wrap(base: Recommenders, tables: dict | None = None,
         permutations: tuple[int, ...] = PERMUTATIONS) -> list[TieBrokenModels]:
    """One wrapped model per declared permutation.

    ``tables`` is empty for the baselines, so their scores still come from
    ``Recommenders.scores`` and only the tie ordering changes.
    """
    return [TieBrokenModels(base, dict(tables or {}), order_key(base.item_ids, seed))
            for seed in permutations]


def wrap_tables(base: Recommenders, name: str, tables: list,
                permutations: tuple[int, ...] = PERMUTATIONS) -> list[TieBrokenModels]:
    """Model-seed by permutation cross product, in seed-major order."""
    return [model for table in tables for model in wrap(base, {name: table}, permutations)]
