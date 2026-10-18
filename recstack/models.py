"""Popularity, item cosine, implicit ALS/BPR, and content baselines."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from recstack.data import Interaction, Item


class Recommenders:
    def __init__(self, train: list[Interaction], catalog: list[Item], seed: int = 7):
        self.catalog = catalog
        self.item_ids = [item.item for item in catalog]
        self.item_index = {item: i for i, item in enumerate(self.item_ids)}
        self.user_ids = sorted({row.user for row in train})
        self.user_index = {user: i for i, user in enumerate(self.user_ids)}
        self.genre = {item.item: item.genre for item in catalog}
        self.history: dict[str, set[str]] = defaultdict(set)
        self.item_interactions = Counter(row.item for row in train)
        self.user_interactions = Counter(row.user for row in train)
        popularity = np.zeros(len(catalog), dtype=np.float32)
        row_indices, column_indices, values = [], [], []
        for row in train:
            if row.item not in self.item_index:
                raise ValueError(f"unknown catalog item {row.item}")
            user_id, item_id = self.user_index[row.user], self.item_index[row.item]
            confidence = math.log1p(row.minutes)
            row_indices.append(user_id)
            column_indices.append(item_id)
            values.append(confidence)
            popularity[item_id] += confidence
            self.history[row.user].add(row.item)
        self.matrix = csr_matrix((values, (row_indices, column_indices)),
                                 shape=(len(self.user_ids), len(self.item_ids)),
                                 dtype=np.float32)
        self.popularity = popularity
        # Item copy is available without interaction labels. Profiles use only
        # the user's training-history confidence; validation/test plays are unseen.
        self.text_features = TfidfVectorizer(stop_words="english").fit_transform(
            item.description for item in catalog
        ).astype(np.float32)
        gram = (self.matrix.T @ self.matrix).toarray()
        lengths = np.sqrt(np.diag(gram))
        denominator = np.outer(lengths, lengths)
        self.item_cosine = np.divide(gram, denominator, out=np.zeros_like(gram),
                                     where=denominator > 0)
        np.fill_diagonal(self.item_cosine, 0)
        self.als = AlternatingLeastSquares(factors=24, regularization=0.05,
                                           iterations=15, random_state=seed,
                                           num_threads=1)
        self.als.fit(self.matrix, show_progress=False)
        self.bpr = BayesianPersonalizedRanking(factors=24, learning_rate=0.05,
                                                regularization=0.01, iterations=40,
                                                random_state=seed, num_threads=1)
        self.bpr.fit(self.matrix, show_progress=False)

    def scores(self, user: str, method: str) -> np.ndarray:
        if method == "popularity" or user not in self.user_index:
            return self.popularity.copy()
        user_id = self.user_index[user]
        if method == "itemknn":
            # Sum neighbours of each interacted item, weighted by implicit confidence.
            return np.asarray(self.matrix[user_id] @ self.item_cosine).ravel()
        if method == "als":
            return np.asarray(self.als.item_factors @ self.als.user_factors[user_id]).ravel()
        if method == "bpr":
            return np.asarray(self.bpr.item_factors @ self.bpr.user_factors[user_id]).ravel()
        if method == "content":
            favourite = Counter(self.genre[item] for item in self.history[user])
            return np.array([favourite[self.genre[item]] + 0.001 * self.popularity[i]
                             for i, item in enumerate(self.item_ids)])
        if method == "text":
            profile = self.matrix[user_id] @ self.text_features
            return np.asarray((self.text_features @ profile.T).toarray()).ravel()
        raise ValueError(f"unknown model {method}")

    def recommend(self, user: str, method: str, k: int = 10) -> list[str]:
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.scores(user, method)
        owned = self.history.get(user, set())
        order = sorted(range(len(self.item_ids)),
                       key=lambda i: (-float(scores[i]), self.item_ids[i]))
        return [self.item_ids[i] for i in order if self.item_ids[i] not in owned][:k]

    def train_counts(self) -> Counter:
        return self.item_interactions.copy()
