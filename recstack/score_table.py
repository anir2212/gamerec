"""Serve or evaluate precomputed user-by-item scores through the Recommenders interface.

This module does not import PyTorch or LightGBM, so the API can load a registered
score table without any training dependency.
"""

from __future__ import annotations

import numpy as np

from recstack.models import Recommenders


class ScoredModels(Recommenders):
    """A fitted ``Recommenders`` plus named score tables shaped (users, items)."""

    def __init__(self, base: Recommenders, tables: dict[str, np.ndarray]):  # noqa: D107
        self.__dict__.update(base.__dict__)
        expected = (len(base.user_ids), len(base.item_ids))
        for name, table in tables.items():
            if table.shape != expected:
                raise ValueError(f"score table {name} has shape {table.shape}, expected {expected}")
            if not np.isfinite(table).all():
                raise ValueError(f"score table {name} contains non-finite values")
        self.tables = tables
        self._base_scores = base.scores

    def scores(self, user: str, method: str) -> np.ndarray:
        if method in self.tables and user in self.user_index:
            return self.tables[method][self.user_index[user]].copy()
        return self._base_scores(user, method)


def base_table(base: Recommenders, method: str) -> np.ndarray:
    """Materialize a baseline's scores for every training user."""
    return np.vstack([base.scores(user, method) for user in base.user_ids])
