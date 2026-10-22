from dataclasses import replace
from pathlib import Path

import pytest

from recstack.contract import SYNTHETIC_CONTRACT, ContractError, DatasetContract, rows_digest, validate_records
from recstack.data import (Interaction, Item, clean, global_cutoffs, global_temporal_split, synthetic,
                           temporal_split, to_records)
from recstack.manifest import build_manifest, default_path, leakage_checks, main, synthetic_manifest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = [Item("a", "puzzle", "a"), Item("b", "puzzle", "b"), Item("c", "racing", "c")]


def record(user="u", item="a", time=1, minutes=10):
    return {"user": user, "item": item, "time": time, "minutes": minutes}


def test_synthetic_records_pass_the_contract_unchanged():
    rows, catalog = synthetic(seed=7)
    validated = validate_records(to_records(rows), SYNTHETIC_CONTRACT, catalog=catalog)
    assert validated == rows
    assert rows_digest(validated) == rows_digest(list(reversed(rows)))  # order-independent


def test_missing_event_time_is_rejected_only_under_a_temporal_claim():
    records = [record(time=1), record(item="b", time=None), {"user": "u", "item": "c", "minutes": 5}]
    with pytest.raises(ContractError, match="missing event time"):
        validate_records(records, SYNTHETIC_CONTRACT, temporal_claim=True)
    kept = validate_records(records, SYNTHETIC_CONTRACT, temporal_claim=False)
    assert kept == [Interaction("u", "a", 1, 10)]


@pytest.mark.parametrize("semantics", ["none", "snapshot"])
def test_untimed_sources_cannot_back_a_temporal_claim(semantics):
    contract = replace(SYNTHETIC_CONTRACT, name="playtime-summary", time_semantics=semantics)
    assert not contract.supports_temporal_claim
    with pytest.raises(ContractError, match="cannot support a temporal evaluation claim"):
        validate_records([record()], contract, temporal_claim=True)
    assert validate_records([record()], contract, temporal_claim=False) == [Interaction("u", "a", 1, 10)]
    with pytest.raises(ContractError, match="cannot support"):
        build_manifest([record()], CATALOG, contract)


def test_unknown_time_semantics_is_rejected_at_contract_creation():
    with pytest.raises(ContractError, match="unknown time_semantics"):
        DatasetContract("x", "y", "play_time_maybe", "s", "w", "l")


@pytest.mark.parametrize("bad, message", [
    (record(time="2020-01-01"), "time must be a nonnegative integer"),
    (record(time=1.5), "time must be a nonnegative integer"),
    (record(time=True), "time must be a nonnegative integer"),
    (record(time=-1), "time must be a nonnegative integer"),
    (record(minutes=-3), "minutes must be a nonnegative integer"),
    (record(user="  "), "nonempty strings"),
    (record(item=7), "nonempty strings"),
    (record(item="zzz"), "not in catalog"),
    ({"user": "u", "time": 1, "minutes": 5}, "missing columns"),
])
def test_malformed_rows_are_rejected(bad, message):
    with pytest.raises(ContractError, match=message):
        validate_records([record(item="b", time=0), bad], SYNTHETIC_CONTRACT, catalog=CATALOG)


def test_duplicate_user_item_time_is_rejected():
    with pytest.raises(ContractError, match="duplicate"):
        validate_records([record(), record(minutes=99)], SYNTHETIC_CONTRACT)


def test_manifest_is_deterministic_and_matches_the_evaluated_split():
    first = synthetic_manifest(7, "per_user")
    assert first == synthetic_manifest(7, "per_user")
    rows, _ = synthetic(seed=7)
    train, validation, test = temporal_split(clean(rows))
    assert first["splits"]["train"]["sha256"] == rows_digest(train)
    assert first["splits"]["test"]["sha256"] == rows_digest(test)
    assert (first["splits"]["train"]["rows"], first["splits"]["validation"]["rows"],
            first["splits"]["test"]["rows"]) == (1808, 596, 596)
    assert first["leakage_checks"]["temporal_order"] == "pass"


@pytest.mark.parametrize("protocol", ["per_user", "global"])
def test_committed_manifest_matches_fresh_computation(protocol, tmp_path):
    saved = ROOT / default_path(7, protocol)
    assert main(["--protocol", protocol, "--output", str(saved), "--check"]) == 0
    altered = tmp_path / "manifest.json"
    altered.write_text(saved.read_text().replace('"rows": 1', '"rows": 2', 1))
    assert main(["--protocol", protocol, "--output", str(altered), "--check"]) == 1
    assert main(["--protocol", protocol, "--output", str(tmp_path / "missing.json"), "--check"]) == 1


def test_global_cutoffs_hold_and_ties_move_to_the_later_split():
    rows = [Interaction("u", "a", t, 5) for t in (1, 2, 3, 4)] + [Interaction("v", "b", 4, 5),
                                                                   Interaction("v", "c", 5, 5)]
    validation_cutoff, test_cutoff = global_cutoffs(rows, 0.34, 0.33)
    train, validation, test = global_temporal_split(rows, validation_cutoff, test_cutoff)
    assert max(r.time for r in train) < validation_cutoff <= min(r.time for r in validation)
    assert max(r.time for r in validation) < test_cutoff <= min(r.time for r in test)
    # Both rows at time 4 land together on one side of the boundary.
    sides = {name for name, part in (("train", train), ("validation", validation), ("test", test))
             if any(r.time == 4 for r in part)}
    assert len(sides) == 1
    with pytest.raises(ValueError, match="too coarse"):
        global_cutoffs([Interaction("u", i, 7, 5) for i in "abcdef"])
    with pytest.raises(ValueError, match="precede"):
        global_temporal_split(rows, 5, 5)


def test_per_user_ties_across_a_boundary_raise():
    rows = [Interaction("u", item, time, 5) for item, time in (("a", 1), ("b", 2), ("c", 2))]
    with pytest.raises(ValueError, match="temporal leakage"):
        temporal_split(rows, 1 / 3, 1 / 3)


def test_leakage_checks_fail_on_corrupted_splits():
    rows, _ = synthetic(seed=7, users=20, items=15)
    rows = clean(rows)
    train, validation, test = temporal_split(rows)
    future = test[-1]
    with pytest.raises(ValueError, match="temporal leakage"):
        leakage_checks(rows, train + [future], validation, test[:-1], "per_user")
    # Dropping one of a user's several test rows keeps the time order valid but
    # silently loses a row; the partition check must notice.
    user = next(u for u in sorted({r.user for r in test}) if sum(r.user == u for r in test) > 1)
    dropped = next(r for r in test if r.user == user)
    with pytest.raises(ValueError, match="exact partition"):
        leakage_checks(rows, train, validation, [r for r in test if r != dropped], "per_user")
    v_cut, t_cut = global_cutoffs(rows)
    g_train, g_val, g_test = global_temporal_split(rows, v_cut, t_cut)
    with pytest.raises(ValueError, match="training row at or after"):
        leakage_checks(rows, g_train + g_test[:1], g_val, g_test[1:], "global",
                       {"validation_cutoff": v_cut, "test_cutoff": t_cut})
    with pytest.raises(ValueError, match="unknown protocol"):
        leakage_checks(rows, train, validation, test, "random")


def test_global_protocol_discloses_users_without_training_history():
    rows = [Interaction("old", "a", 1, 5), Interaction("old", "b", 2, 5), Interaction("old", "c", 3, 5),
            Interaction("new", "a", 9, 5)]
    v_cut, t_cut = 2, 3
    train, validation, test = global_temporal_split(rows, v_cut, t_cut)
    checks = leakage_checks(rows, train, validation, test, "global",
                            {"validation_cutoff": v_cut, "test_cutoff": t_cut})
    assert checks["test_users_without_train"] == 1
    # The per-user protocol forbids this, so the same split fails its rules.
    with pytest.raises(ValueError):
        leakage_checks(rows, train, validation, test, "per_user")
