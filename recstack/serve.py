"""Local-only FastAPI recommender backed by a seeded trained synthetic model."""

from __future__ import annotations

import os
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request

from recstack.data import GENRES, clean, synthetic, temporal_split
from recstack.models import Recommenders
from recstack.registry import Registry, RegistryError

app = FastAPI(title="GameRec local baseline", version="0.1.0")


@lru_cache(maxsize=1)
def trained_models() -> Recommenders:
    rows, catalog = synthetic(seed=7)
    train, _, _ = temporal_split(clean(rows))
    return Recommenders(train, catalog, seed=7)


# The one registered score table in this process. It changes only through a successful reload.
_champion: dict = {"version": None, "table": None}
_champion_lock = threading.Lock()
LOCAL_CLIENTS = {"127.0.0.1", "::1", "testclient"}


def registry_path() -> Path:
    return Path(os.environ.get("GAMEREC_REGISTRY", "runs/registry"))


@app.get("/health")
def health():
    return {"status": "ok", "data": "seeded synthetic", "model": "local-baselines",
            "champion_version": _champion["version"]}


@app.post("/admin/reload")
def reload_champion(request: Request):
    """Load the registry's champion. A failed load keeps whatever this process already serves.

    The route only accepts loopback clients. That is a local guard, not production authentication.
    """
    if request.client is None or request.client.host not in LOCAL_CLIENTS:
        raise HTTPException(403, "reload is limited to local clients")
    registry = Registry(registry_path())
    pointer = registry.champion()
    if pointer is None:
        raise HTTPException(409, "no champion is registered")
    models = trained_models()
    try:
        table, user_ids, item_ids, _ = registry.load(pointer["version"])
    except RegistryError as error:
        raise HTTPException(409, f"champion refused: {error}") from error
    if user_ids != models.user_ids or item_ids != models.item_ids:
        raise HTTPException(409, "champion refused: its users or items differ from this service's corpus")
    with _champion_lock:
        _champion.update(version=pointer["version"], table=table)
    return {"champion_version": pointer["version"], "data": "seeded synthetic"}


@app.get("/recommend")
def recommend(user_id: str, k: int = Query(10, ge=1, le=20), method: str = "itemknn",
              genres: str = ""):
    if method not in {"popularity", "itemknn", "als", "bpr", "content", "text", "champion"}:
        raise HTTPException(400, "unknown recommender method")
    preferred = {genre.strip().lower() for genre in genres.split(",") if genre.strip()}
    if preferred - set(GENRES):
        raise HTTPException(400, "unknown genre")
    models = trained_models()
    known = user_id in models.user_index
    with _champion_lock:
        champion_version, champion_table = _champion["version"], _champion["table"]
    if method == "champion" and champion_table is None:
        raise HTTPException(409, "no champion is loaded; call /admin/reload first")
    if method == "champion" and known:
        scores = np.array(champion_table[models.user_index[user_id]], dtype=np.float64)
    else:
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
            "champion_version": champion_version if method == "champion" else None,
            "items": selected, "data": "seeded synthetic"}
