"""Hash-chain integrity (doc 10 §2, §9): the verifier detects any single-row
tamper, delete, or reorder in a fixture chain."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dynamo.learning.hash_chain import (
    GENESIS,
    ChainFault,
    canonical_json,
    chain_body,
    row_hash,
    verify_chain,
)


def build_chain(n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev = GENESIS
    for i in range(n):
        body = {"id": f"dc-{i}", "symbol": "AAPL", "payload": {"conviction": i / 10}}
        digest = row_hash(prev, body)
        rows.append({**body, "prev_hash": prev, "row_hash": digest})
        prev = digest
    return rows


def test_valid_chain_verifies() -> None:
    assert verify_chain(build_chain(6)).ok
    assert verify_chain([]).ok  # empty chain trivially valid


@pytest.mark.parametrize("victim", [0, 2, 5])
def test_tampered_body_detected_at_offending_row(victim: int) -> None:
    rows = build_chain(6)
    rows[victim]["payload"] = {"conviction": 0.99}  # rewrite history
    result = verify_chain(rows)
    assert not result.ok
    assert result.fault == ChainFault.HASH_MISMATCH
    assert result.index == victim


@pytest.mark.parametrize("victim", [0, 2, 4])
def test_deleted_row_detected(victim: int) -> None:
    rows = build_chain(6)
    del rows[victim]
    result = verify_chain(rows)
    assert not result.ok
    assert result.fault in (ChainFault.BAD_GENESIS, ChainFault.BROKEN_LINK)
    assert result.index == victim  # break surfaces where the gap is


def test_reordered_rows_detected() -> None:
    rows = build_chain(6)
    rows[2], rows[3] = rows[3], rows[2]
    result = verify_chain(rows)
    assert not result.ok
    assert result.fault == ChainFault.BROKEN_LINK
    assert result.index == 2


def test_forged_hash_pair_detected() -> None:
    # Attacker recomputes the row hash but cannot relink the successor.
    rows = build_chain(4)
    rows[1]["payload"] = {"conviction": 1.0}
    rows[1]["row_hash"] = row_hash(rows[1]["prev_hash"], chain_body(rows[1]))
    result = verify_chain(rows)
    assert not result.ok
    assert result.fault == ChainFault.BROKEN_LINK
    assert result.index == 2


def test_wrong_genesis_detected() -> None:
    rows = build_chain(3)
    rows[0]["prev_hash"] = "f" * 64
    result = verify_chain(rows)
    assert not result.ok
    assert result.fault == ChainFault.BAD_GENESIS


def test_canonical_json_is_key_order_invariant() -> None:
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert row_hash(GENESIS, a) == row_hash(GENESIS, b)


@given(
    st.lists(
        st.dictionaries(
            st.sampled_from(["symbol", "note", "value"]),
            st.one_of(st.text(max_size=8), st.floats(allow_nan=False, allow_infinity=False)),
            max_size=3,
        ),
        min_size=2,
        max_size=8,
    ),
    st.data(),
)
def test_any_single_body_mutation_breaks_the_chain(
    bodies: list[dict[str, Any]], data: st.DataObject
) -> None:
    prev = GENESIS
    rows: list[dict[str, Any]] = []
    for i, body in enumerate(bodies):
        full = {"id": f"dc-{i}", **body}
        digest = row_hash(prev, full)
        rows.append({**full, "prev_hash": prev, "row_hash": digest})
        prev = digest
    assert verify_chain(rows).ok
    victim = data.draw(st.integers(min_value=0, max_value=len(rows) - 1))
    rows[victim]["id"] = "dc-tampered"
    assert not verify_chain(rows).ok
