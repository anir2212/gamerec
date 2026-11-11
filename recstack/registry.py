"""File-based model registry: immutable versions plus one atomically replaced champion pointer.

This local stand-in for an MLflow registry stores score tables on disk. It is not a service.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


class RegistryError(RuntimeError):
    """A version is missing, altered or inconsistent. Callers keep what they already serve."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Registry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.versions = self.root / "versions"
        self.pointer = self.root / "champion.json"
        self.log = self.root / "decisions.jsonl"

    def champion(self) -> dict | None:
        """Return the champion pointer, which holds the version and its evaluation record."""
        if not self.pointer.exists():
            return None
        return json.loads(self.pointer.read_text())

    def _next_version(self) -> str:
        existing = [int(path.name[1:]) for path in self.versions.glob("v*") if re.fullmatch(r"v\d{4}", path.name)]
        return f"v{max(existing, default=0) + 1:04d}"

    def add_version(self, table: np.ndarray, user_ids: list[str], item_ids: list[str], metadata: dict) -> str:
        """Write a complete version directory under a temporary name, then rename it into place."""
        if table.shape != (len(user_ids), len(item_ids)):
            raise RegistryError("score table shape does not match its IDs")
        if not np.isfinite(table).all():
            raise RegistryError("score table contains non-finite values")
        self.versions.mkdir(parents=True, exist_ok=True)
        version = self._next_version()
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".staging-"))
        try:
            scores = staging / "scores.npz"
            np.savez(scores, table=table, user_ids=np.array(user_ids), item_ids=np.array(item_ids))
            record = {**metadata, "version": version, "scores_sha256": sha256_file(scores)}
            (staging / "metadata.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            os.rename(staging, self.versions / version)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return version

    def metadata(self, version: str) -> dict:
        path = self.versions / version / "metadata.json"
        if not path.exists():
            raise RegistryError(f"unknown version {version}")
        return json.loads(path.read_text())

    def load(self, version: str) -> tuple[np.ndarray, list[str], list[str], dict]:
        """Load a version after checking its file hash, shape and finiteness."""
        metadata = self.metadata(version)
        path = self.versions / version / "scores.npz"
        if not path.exists() or sha256_file(path) != metadata["scores_sha256"]:
            raise RegistryError(f"artifact hash mismatch for {version}")
        with np.load(path, allow_pickle=False) as data:
            table, user_ids, item_ids = data["table"], data["user_ids"].tolist(), data["item_ids"].tolist()
        if table.shape != (len(user_ids), len(item_ids)) or not np.isfinite(table).all():
            raise RegistryError(f"invalid score table in {version}")
        return table, user_ids, item_ids, metadata

    def promote(self, version: str) -> dict:
        """Atomically point the champion at a verified version."""
        _, _, _, metadata = self.load(version)
        record = {"version": version, "scores_sha256": metadata["scores_sha256"],
                  "evaluation": metadata["evaluation"]}
        temporary = self.pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.pointer)
        return record

    def record_decision(self, decision: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as handle:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")

    def decisions(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]
