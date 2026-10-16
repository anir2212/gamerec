"""Local-only FastAPI recommender backed by a seeded trained synthetic model."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from recstack.data import GENRES, clean, synthetic, temporal_split
from recstack.models import Recommenders

app = FastAPI(title="GameRec local baseline", version="0.1.0")


@lru_cache(maxsize=1)
def trained_models() -> Recommenders:
    rows, catalog = synthetic(seed=7)
    train, _, _ = temporal_split(clean(rows))
    return Recommenders(train, catalog, seed=7)


@app.get("/health")
def health():
    return {"status": "ok", "data": "seeded synthetic", "model": "local-baselines"}


@app.get("/recommend")
def recommend(user_id: str, k: int = Query(10, ge=1, le=20), method: str = "itemknn",
              genres: str = ""):
    if method not in {"popularity", "itemknn", "als", "bpr", "content"}:
        raise HTTPException(400, "unknown recommender method")
    preferred = {genre.strip().lower() for genre in genres.split(",") if genre.strip()}
    if preferred - set(GENRES):
        raise HTTPException(400, "unknown genre")
    models = trained_models()
    known = user_id in models.user_index
    scores = models.scores(user_id, method if known else "popularity")
    if not known and preferred:
        # Explicit cold-user fallback: declared genres outrank generic popularity.
        for index, item in enumerate(models.item_ids):
            if models.genre[item] in preferred:
                scores[index] += 1_000_000
    owned = models.history.get(user_id, set())
    order = sorted(range(len(models.item_ids)),
                   key=lambda index: (-float(scores[index]), models.item_ids[index]))
    genre_counts = Counter()
    selected = []
    for index in order:
        item = models.item_ids[index]
        genre = models.genre[item]
        if item in owned or genre_counts[genre] >= 4:
            continue
        genre_counts[genre] += 1
        selected.append({"item_id": item, "genre": genre,
                         "score": float(scores[index])})
        if len(selected) >= k:
            break
    return {"user_id": user_id, "known_user": known,
            "method": method if known else "genre-popularity-fallback",
            "items": selected, "data": "seeded synthetic"}
