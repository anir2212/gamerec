"""Deterministic temporal split manifest with leakage checks.

The manifest records the contract, input digest, cut-offs, per-split counts and
row hashes, so a later result can name exactly which rows trained and scored it.
It contains no wall-clock time, so the same input always yields the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from recstack.contract import SYNTHETIC_CONTRACT, DatasetContract, rows_digest, sha256_json, validate_records
from recstack.data import (Interaction, Item, assert_global_cutoffs, assert_no_temporal_leakage, clean,
                           global_cutoffs, global_temporal_split, synthetic, temporal_split, to_records)

PROTOCOLS = ("per_user", "global")
TIE_POLICY = {
    "per_user": "sort by (time, item); equal times across a user's split boundary raise ValueError",
    "global": "rows with time >= cut-off go to the later split, so equal times never straddle a boundary",
}


def _summary(rows: list[Interaction]) -> dict:
    return {"rows": len(rows), "users": len({r.user for r in rows}), "items": len({r.item for r in rows}),
            "time_min": min(r.time for r in rows), "time_max": max(r.time for r in rows),
            "sha256": rows_digest(rows)}


def leakage_checks(rows, train, validation, test, protocol: str, cutoffs: dict | None = None) -> dict:
    """Raise on any protocol violation; return the disclosed (allowed) overlaps as counts."""
    if protocol == "per_user":
        assert_no_temporal_leakage(train, validation, test)
    elif protocol == "global":
        assert_global_cutoffs(train, validation, test, cutoffs["validation_cutoff"], cutoffs["test_cutoff"])
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    keys = [{(r.user, r.item, r.time) for r in part} for part in (train, validation, test)]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise ValueError("leakage: the same (user, item, time) row appears in two splits")
    if len(train) + len(validation) + len(test) != len(rows) or \
            rows_digest(train + validation + test) != rows_digest(rows):
        raise ValueError("split is not an exact partition of the validated input")
    train_users = {r.user for r in train}
    train_items = {r.item for r in train}
    owned = {(r.user, r.item) for r in train}
    checks = {
        "temporal_order": "pass",
        "no_row_in_two_splits": "pass",
        "exact_partition": "pass",
        "validation_users_without_train": len({r.user for r in validation} - train_users),
        "test_users_without_train": len({r.user for r in test} - train_users),
        "test_items_unseen_in_train": len({r.item for r in test} - train_items),
        "test_rows_already_owned_in_train": sum((r.user, r.item) in owned for r in test),
    }
    if protocol == "per_user":
        # The per-user protocol orders each timeline separately, so another
        # user's training event may be later than this test event. Disclose it.
        latest_train = max(r.time for r in train)
        checks["test_rows_before_latest_train_time_any_user"] = sum(r.time < latest_train for r in test)
        if checks["validation_users_without_train"] or checks["test_users_without_train"]:
            raise ValueError("per-user protocol requires training history for every evaluated user")
    return checks


def build_manifest(records: list[dict], catalog: list[Item], contract: DatasetContract,
                   protocol: str = "per_user", test_fraction: float = 0.2,
                   validation_fraction: float = 0.2, cap_minutes: int = 1000, command: str = ""):
    """Validate, clean and split records; return ``(manifest, (train, validation, test))``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    validated = validate_records(records, contract, temporal_claim=True, catalog=catalog)
    rows = clean(validated, cap_minutes)
    if protocol == "per_user":
        train, validation, test = temporal_split(rows, test_fraction, validation_fraction)
        table = []
        by_user = {}
        for name, part in (("train", train), ("validation", validation), ("test", test)):
            for r in part:
                by_user.setdefault(r.user, {}).setdefault(name, []).append(r.time)
        for user in sorted(by_user):
            times = by_user[user]
            table.append([user, max(times["train"]), min(times["validation"]),
                          max(times["validation"]), min(times["test"])])
        cutoffs = {"kind": "per-user boundaries",
                   "per_user_table_columns": ["user", "train_max", "validation_min", "validation_max", "test_min"],
                   "per_user_table_sha256": sha256_json(table), "users": len(table)}
    else:
        validation_cutoff, test_cutoff = global_cutoffs(rows, test_fraction, validation_fraction)
        train, validation, test = global_temporal_split(rows, validation_cutoff, test_cutoff)
        cutoffs = {"kind": "global", "validation_cutoff": validation_cutoff, "test_cutoff": test_cutoff}
    checks = leakage_checks(rows, train, validation, test, protocol, cutoffs)
    item_counts = Counter(r.item for r in train)
    manifest = {
        "manifest_version": 1,
        "command": command,
        "contract": contract.to_dict(),
        "contract_sha256": contract.digest(),
        "catalog": {"items": len(catalog),
                    "sha256": sha256_json(sorted([i.item, i.genre, i.description] for i in catalog))},
        "input": {"rows_validated": len(validated), "rows_after_clean": len(rows),
                  "rows_dropped_nonpositive_minutes": len(validated) - len(rows),
                  "cap_minutes": cap_minutes, "sha256_after_clean": rows_digest(rows)},
        "protocol": {"name": protocol, "test_fraction": test_fraction,
                     "validation_fraction": validation_fraction, "tie_policy": TIE_POLICY[protocol]},
        "cutoffs": cutoffs,
        "splits": {"train": _summary(train), "validation": _summary(validation), "test": _summary(test)},
        "cold_counts": {"catalog_items_at_most_5_train": sum(item_counts[i.item] <= 5 for i in catalog)},
        "leakage_checks": checks,
    }
    return manifest, (train, validation, test)


def synthetic_manifest(seed: int = 7, protocol: str = "per_user") -> dict:
    rows, catalog = synthetic(seed)
    command = f"python -m recstack.manifest --seed {seed} --protocol {protocol}"
    manifest, _ = build_manifest(to_records(rows), catalog, SYNTHETIC_CONTRACT, protocol, command=command)
    manifest["dataset"] = f"seeded synthetic fixture, seed {seed}"
    return manifest


def default_path(seed: int, protocol: str) -> Path:
    return Path("data/manifests") / f"synthetic_seed{seed}_{protocol}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="per_user")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="recompute and fail if the saved manifest differs")
    args = parser.parse_args(argv)
    path = args.output or default_path(args.seed, args.protocol)
    text = json.dumps(synthetic_manifest(args.seed, args.protocol), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not path.exists() or path.read_text() != text:
            print(f"manifest mismatch: {path} differs from a fresh computation", file=sys.stderr)
            return 1
        print(f"manifest matches: {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
