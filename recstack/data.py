"""Deterministic synthetic interactions and leak-free per-user temporal splits."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Interaction:
    user: str
    item: str
    time: int
    minutes: int


@dataclass(frozen=True)
class Item:
    item: str
    genre: str
    description: str


GENRES = ("strategy", "arcade", "puzzle", "simulation", "racing")


def synthetic(seed: int = 7, users: int = 120, items: int = 80):
    """Generate preference-correlated activity with an explicit long item tail."""
    rng = random.Random(seed)
    catalog = [Item(f"g{i:03d}", GENRES[i % len(GENRES)],
                    f"A {GENRES[i % len(GENRES)]} game with variant {i}")
               for i in range(items)]
    rows = []
    for user_index in range(users):
        preferred = GENRES[user_index % len(GENRES)]
        # Ten short-history users make the new-user evaluation denominator nonempty.
        count = 5 if user_index < 10 else rng.randint(20, 34)
        for event_index in range(count):
            own_genre = rng.random() < 0.74
            pool = [item for item in catalog if item.genre == preferred] if own_genre else catalog
            weights = [1 / (1 + int(item.item[1:])) ** 1.15 for item in pool]
            game = rng.choices(pool, weights=weights)[0]
            rows.append(Interaction(f"u{user_index:03d}", game.item,
                                    event_index * 1000 + user_index,
                                    rng.randint(5, 240)))
    return rows, catalog


def clean(rows: list[Interaction], cap_minutes: int = 1000) -> list[Interaction]:
    """Drop nonpositive play and cap extreme durations before confidence weights."""
    if cap_minutes <= 0:
        raise ValueError("cap_minutes must be positive")
    return [Interaction(r.user, r.item, r.time, min(r.minutes, cap_minutes))
            for r in rows if r.minutes > 0]


def temporal_split(rows: list[Interaction], test_fraction: float = 0.2,
                   validation_fraction: float = 0.2):
    """Sort each user's timeline; ties preserve item order for deterministic folds."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 or test_fraction + validation_fraction >= 1:
        raise ValueError("invalid split fractions")
    groups: dict[str, list[Interaction]] = defaultdict(list)
    for row in rows:
        groups[row.user].append(row)
    train, validation, test = [], [], []
    for user in sorted(groups):
        timeline = sorted(groups[user], key=lambda x: (x.time, x.item))
        if len(timeline) < 3:
            raise ValueError(f"user {user} needs at least three interactions")
        n_test = max(1, round(len(timeline) * test_fraction))
        n_val = max(1, round(len(timeline) * validation_fraction))
        if n_test + n_val >= len(timeline):
            raise ValueError(f"no training interactions for {user}")
        train.extend(timeline[:-(n_test + n_val)])
        validation.extend(timeline[-(n_test + n_val):-n_test])
        test.extend(timeline[-n_test:])
    assert_no_temporal_leakage(train, validation, test)
    return train, validation, test


def assert_no_temporal_leakage(train, validation, test):
    """Fail if any user has an earlier validation/test event than their training tail."""
    by_split = [defaultdict(list) for _ in range(3)]
    for group, rows in zip(by_split, (train, validation, test), strict=True):
        for row in rows:
            group[row.user].append(row.time)
    for user in set().union(*(set(group) for group in by_split)):
        a, b, c = (group[user] for group in by_split)
        if not a or not b or not c or max(a) >= min(b) or max(b) >= min(c):
            raise ValueError(f"temporal leakage for {user}")


def split_stats(train, validation, test, catalog):
    counts = Counter(row.item for row in train)
    by_user = Counter(row.user for row in train)
    return {
        "train": len(train), "validation": len(validation), "test": len(test),
        "users": len(by_user), "catalog_items": len(catalog),
        "new_users_at_most_3_train": sum(n <= 3 for n in by_user.values()),
        "cold_items_at_most_5_train": sum(counts[item.item] <= 5 for item in catalog),
        "test_events_on_cold_items": sum(counts[row.item] <= 5 for row in test),
    }


def to_records(rows: list[Interaction]) -> list[dict]:
    return [asdict(row) for row in rows]
