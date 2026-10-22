"""Source-agnostic interaction schema contract and row validation.

A contract states what each column means before any record is accepted. In
particular, ``time_semantics`` says what the timestamp measures. Only some
meanings can support a temporal (past-predicts-future) evaluation claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from recstack.data import Interaction, Item

REQUIRED_COLUMNS = {"user": str, "item": str, "time": int, "minutes": int}

# What a timestamp can mean, and whether that meaning can order a temporal split.
TIME_SEMANTICS = {
    "synthetic_ordinal": True,     # generator's per-user event order (fixture only)
    "interaction_event": True,     # time the play/purchase/rating itself happened
    "review_publication": True,    # time a review was posted; orders reviews, not play
    "snapshot": False,             # crawl or export time shared by many rows
    "none": False,                 # source has no per-interaction time
}


class ContractError(ValueError):
    """Raised when records or a requested claim break the dataset contract."""


@dataclass(frozen=True)
class DatasetContract:
    name: str
    source: str
    time_semantics: str
    time_unit: str
    weight_meaning: str
    licence_status: str

    def __post_init__(self):
        if self.time_semantics not in TIME_SEMANTICS:
            raise ContractError(f"unknown time_semantics {self.time_semantics!r}")

    @property
    def supports_temporal_claim(self) -> bool:
        return TIME_SEMANTICS[self.time_semantics]

    def to_dict(self) -> dict:
        return {**asdict(self), "required_columns": {k: v.__name__ for k, v in REQUIRED_COLUMNS.items()},
                "supports_temporal_claim": self.supports_temporal_claim}

    def digest(self) -> str:
        return sha256_json(self.to_dict())


SYNTHETIC_CONTRACT = DatasetContract(
    name="gamerec-synthetic-v1",
    source="recstack.data.synthetic (fictional users and games)",
    time_semantics="synthetic_ordinal",
    time_unit="ordinal: event_index * 1000 + user_index",
    weight_meaning="generated minutes for that one event",
    licence_status="generated in this repository; no third-party rights",
)


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows_digest(rows: list[Interaction]) -> str:
    """Order-independent SHA-256 over canonical (user, item, time, minutes) rows."""
    return sha256_json(sorted([row.user, row.item, row.time, row.minutes] for row in rows))


def validate_records(records: list[dict], contract: DatasetContract, temporal_claim: bool = True,
                     catalog: list[Item] | None = None) -> list[Interaction]:
    """Return typed interactions or raise ``ContractError`` naming the first bad rows.

    With ``temporal_claim=True`` the contract's timestamp meaning must support
    ordering and every row must carry an integer event time. Without a temporal
    claim, rows with no time are skipped because they cannot be ordered; compare
    input and output lengths to count them, and report no temporal result.
    """
    if temporal_claim and not contract.supports_temporal_claim:
        raise ContractError(f"{contract.name}: time_semantics={contract.time_semantics!r} "
                            "cannot support a temporal evaluation claim")
    known_items = None if catalog is None else {item.item for item in catalog}
    problems: list[str] = []
    rows: list[Interaction] = []
    seen: set[tuple[str, str, int]] = set()
    for index, record in enumerate(records):
        missing = [column for column in REQUIRED_COLUMNS if column not in record]
        missing_time = record.get("time") is None
        if missing_time and "time" in missing:
            missing.remove("time")
        if missing:
            problems.append(f"row {index}: missing columns {missing}")
            continue
        if missing_time:
            if temporal_claim:
                problems.append(f"row {index}: missing event time under a temporal claim")
            continue  # Non-temporal use: no time means the row cannot be ordered; skip it.
        user, item, time, minutes = (record[column] for column in REQUIRED_COLUMNS)
        if not isinstance(user, str) or not user.strip() or not isinstance(item, str) or not item.strip():
            problems.append(f"row {index}: user and item must be nonempty strings")
            continue
        if isinstance(time, bool) or not isinstance(time, int) or time < 0:
            problems.append(f"row {index}: time must be a nonnegative integer")
            continue
        if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0:
            problems.append(f"row {index}: minutes must be a nonnegative integer")
            continue
        if known_items is not None and item not in known_items:
            problems.append(f"row {index}: item {item!r} not in catalog")
            continue
        key = (user, item, time)
        if key in seen:
            problems.append(f"row {index}: duplicate (user, item, time) {key}")
            continue
        seen.add(key)
        rows.append(Interaction(user, item, time, minutes))
    if problems:
        shown = "; ".join(problems[:5])
        raise ContractError(f"{contract.name}: {len(problems)} invalid rows: {shown}")
    return rows
