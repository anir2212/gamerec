"""Two-tower retrieval model trained with in-batch sampled softmax (synthetic corpus only).

Item tower: ID embedding + TF-IDF description projection + genre embedding.
User tower: confidence-weighted mean of the item-tower vectors in the user's
earlier plays, a recency-weighted mean of the same vectors, and a behavioural
segment (activity bucket plus genre shares of those earlier plays).

Training examples use each user's own earlier training rows as history, so a
target item never sits in its own input. Items with at most ``COLD_ITEM_MAX``
training events have their ID embedding masked: their vector comes from content
only, both in training and in scoring.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from recstack.data import GENRES, Interaction, Item

COLD_ITEM_MAX = 5
RECENCY_DECAY = 0.7
VARIANTS = ("hybrid", "id_only", "content_only")


@dataclass(frozen=True)
class TowerConfig:
    dim: int = 16
    temperature: float = 0.2
    logq: bool = False
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.005
    weight_decay: float = 1e-5
    id_dropout: float = 0.1
    variant: str = "hybrid"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown two-tower variant {self.variant}")
        if self.temperature <= 0 or self.dim <= 0 or self.epochs <= 0 or self.batch_size < 2:
            raise ValueError("invalid two-tower configuration")

    def to_dict(self) -> dict:
        return asdict(self)


def content_vectors(catalog: list[Item]) -> np.ndarray:
    """TF-IDF of descriptions with bare numbers removed.

    The fixture's copy is "A <genre> game with variant <n>". A number occurs in one
    description only, so a learned projection could use it as a disguised item ID
    and "content only" would stop being content only.
    """
    def without_numbers(text: str) -> str:
        return re.sub(r"\b\d+\b", " ", text)

    vectors = TfidfVectorizer(stop_words="english", preprocessor=lambda t: without_numbers(t.lower()))
    return vectors.fit_transform(item.description for item in catalog).toarray().astype(np.float32)


class Features:
    """Catalog and training-history arrays. Only training rows may enter this object."""

    def __init__(self, train: list[Interaction], catalog: list[Item], text: np.ndarray | None = None):
        self.item_ids = [item.item for item in catalog]
        self.item_index = {item: index for index, item in enumerate(self.item_ids)}
        self.n_items = len(catalog)
        self.text = content_vectors(catalog) if text is None else text.astype(np.float32)
        self.genre_ids = np.array([GENRES.index(item.genre) for item in catalog], dtype=np.int64)
        counts = np.zeros(self.n_items, dtype=np.float32)
        for row in train:
            counts[self.item_index[row.item]] += 1
        self.counts = counts
        self.cold = counts <= COLD_ITEM_MAX
        # Add-one smoothing keeps log q finite for an item with no training row.
        self.log_q = np.log((counts + 1) / (counts.sum() + self.n_items)).astype(np.float32)
        timelines: dict[str, list[Interaction]] = {}
        for row in sorted(train, key=lambda r: (r.user, r.time, r.item)):
            timelines.setdefault(row.user, []).append(row)
        self.timelines = timelines
        self.user_ids = sorted(timelines)
        self.owned = {user: {self.item_index[row.item] for row in rows} for user, rows in timelines.items()}

    def user_vector(self, rows: list[Interaction]) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Return confidence history weights, recency weights, activity bucket and genre shares."""
        weights = np.zeros(self.n_items, dtype=np.float32)
        recent = np.zeros(self.n_items, dtype=np.float32)
        genres = np.zeros(len(GENRES), dtype=np.float32)
        for age, row in enumerate(reversed(rows)):
            index = self.item_index[row.item]
            weights[index] += math.log1p(row.minutes)
            recent[index] += RECENCY_DECAY ** age
            genres[self.genre_ids[index]] += 1
        for values in (weights, recent, genres):
            total = values.sum()
            if total > 0:
                values /= total
        bucket = 0 if len(rows) <= 3 else 1 if len(rows) <= 10 else 2
        return weights, recent, bucket, genres

    def training_examples(self) -> dict[str, np.ndarray]:
        """One example per novel play that has at least one earlier play by that user."""
        history, recency, buckets, genres, targets, users = [], [], [], [], [], []
        for user_position, user in enumerate(self.user_ids):
            rows = self.timelines[user]
            for position in range(1, len(rows)):
                target = self.item_index[rows[position].item]
                prefix = rows[:position]
                if target in {self.item_index[row.item] for row in prefix}:
                    continue  # Serving cannot recommend an owned game, so do not train on repeats.
                weights, recent, bucket, shares = self.user_vector(prefix)
                history.append(weights)
                recency.append(recent)
                buckets.append(bucket)
                genres.append(shares)
                targets.append(target)
                users.append(user_position)
        if not targets:
            raise ValueError("no two-tower training examples: every user needs a novel later play")
        return {"history": np.stack(history), "recency": np.stack(recency),
                "bucket": np.array(buckets, dtype=np.int64), "genres": np.stack(genres),
                "target": np.array(targets, dtype=np.int64), "user": np.array(users, dtype=np.int64)}

    def scoring_inputs(self) -> dict[str, np.ndarray]:
        rows = [self.user_vector(self.timelines[user]) for user in self.user_ids]
        return {"history": np.stack([row[0] for row in rows]), "recency": np.stack([row[1] for row in rows]),
                "bucket": np.array([row[2] for row in rows], dtype=np.int64),
                "genres": np.stack([row[3] for row in rows])}


class TwoTower(nn.Module):
    def __init__(self, features: Features, config: TowerConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.register_buffer("text", torch.as_tensor(features.text))
        self.register_buffer("genre_ids", torch.as_tensor(features.genre_ids))
        id_allowed = np.ones(features.n_items, dtype=bool) if config.variant == "id_only" else ~features.cold
        self.register_buffer("id_allowed", torch.as_tensor(id_allowed))
        # Items with no ID channel and the same content must score exactly alike. Without this,
        # floating-point rounding, not the model, would order them.
        canonical, first_seen = np.arange(features.n_items), {}
        masked = np.zeros(features.n_items, dtype=bool) if config.variant == "id_only" else (
            np.ones(features.n_items, dtype=bool) if config.variant == "content_only" else ~id_allowed)
        for index in np.flatnonzero(masked):
            key = (features.text[index].tobytes(), int(features.genre_ids[index]))
            canonical[index] = first_seen.setdefault(key, int(index))
        self.register_buffer("canonical", torch.as_tensor(canonical))
        self.id_embedding = nn.Embedding(features.n_items, dim)
        self.text_projection = nn.Linear(features.text.shape[1], dim, bias=False)
        self.genre_embedding = nn.Embedding(len(GENRES), dim)
        self.item_mlp = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.bucket_embedding = nn.Embedding(3, dim)
        self.genre_share = nn.Linear(len(GENRES), dim, bias=False)
        self.user_mlp = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def item_vectors(self, drop_ids: bool = False) -> torch.Tensor:
        """Vectors for the whole catalog; a masked ID contributes zeros, which keeps cold items content-only."""
        variant = self.config.variant
        indices = torch.arange(self.text.shape[0], device=self.text.device)
        ids = self.id_embedding(indices)
        keep = self.id_allowed
        if variant == "content_only":
            keep = torch.zeros_like(keep)
        elif drop_ids and self.config.id_dropout > 0 and variant == "hybrid":
            keep = keep & (torch.rand(keep.shape, device=keep.device) >= self.config.id_dropout)
        ids = ids * keep.unsqueeze(1)
        text = self.text_projection(self.text)
        genre = self.genre_embedding(self.genre_ids)
        if variant == "id_only":
            text, genre = torch.zeros_like(text), torch.zeros_like(genre)
        vectors = nn.functional.normalize(self.item_mlp(torch.cat([ids, text, genre], dim=1)), dim=1)
        return vectors if self.training else vectors[self.canonical]

    def user_vectors(self, items: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [batch["history"] @ items, batch["recency"] @ items,
                 self.bucket_embedding(batch["bucket"]), self.genre_share(batch["genres"])]
        return nn.functional.normalize(self.user_mlp(torch.cat(parts, dim=1)), dim=1)


def in_batch_loss(users: torch.Tensor, positives: torch.Tensor, targets: torch.Tensor, owned_mask: torch.Tensor,
                  temperature: float, log_q: torch.Tensor | None = None) -> torch.Tensor:
    """Sampled softmax where the other rows' positives are the negatives.

    ``owned_mask[i, j]`` marks a batch item that row i's user played in training. Those and
    duplicates of the row's own target are not negatives, so they are removed from the softmax.
    ``log_q`` applies the log-frequency correction for how often an item is drawn as a negative.
    """
    logits = users @ positives.T / temperature
    if log_q is not None:
        logits = logits - log_q[targets]
    same_target = targets.unsqueeze(1) == targets.unsqueeze(0)
    remove = (same_target | owned_mask) & ~torch.eye(len(targets), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(remove, -1e9)
    return nn.functional.cross_entropy(logits, torch.arange(len(targets), device=logits.device))


def fit(features: Features, config: TowerConfig, seed: int) -> TwoTower:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    examples = {key: torch.as_tensor(value) for key, value in features.training_examples().items()}
    owned = torch.zeros((len(features.user_ids), features.n_items), dtype=torch.bool)
    for position, user in enumerate(features.user_ids):
        owned[position, sorted(features.owned[user])] = True
    log_q = torch.as_tensor(features.log_q) if config.logq else None
    model = TwoTower(features, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    size = len(examples["target"])
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(size, generator=generator)
        for start in range(0, size, config.batch_size):
            index = order[start:start + config.batch_size]
            if len(index) < 2:
                continue
            batch = {key: examples[key][index] for key in ("history", "recency", "bucket", "genres")}
            targets = examples["target"][index]
            items = model.item_vectors(drop_ids=True)
            users = model.user_vectors(items, batch)
            owned_mask = owned[examples["user"][index]][:, targets]
            loss = in_batch_loss(users, items[targets], targets, owned_mask, config.temperature, log_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def score_table(model: TwoTower, features: Features, user_order: list[str] | None = None) -> np.ndarray:
    """Cosine user/item scores from training history only; logQ is a training correction and is not applied here."""
    inputs = {key: torch.as_tensor(value) for key, value in features.scoring_inputs().items()}
    items = model.item_vectors()
    table = (model.user_vectors(items, inputs) @ items.T).numpy().astype(np.float64)[:, model.canonical.numpy()]
    if user_order is not None and user_order != features.user_ids:
        raise ValueError("score table user order differs from the feature user order")
    return table
