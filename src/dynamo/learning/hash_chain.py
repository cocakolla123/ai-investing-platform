"""Hash-chained append-only integrity for ``decision_context`` (doc 10 §2).

``row_hash = sha256(prev_hash ‖ canonical_json(body))`` with ``prev_hash``
pointing at the previous row's ``row_hash`` (genesis rows use a fixed
constant). Tampering with, deleting, or reordering any historical row breaks
every subsequent hash — evident, not preventable, which is the right property
for a single-user system auditing itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

GENESIS = "0" * 64

_HASH_FIELDS = ("prev_hash", "row_hash")


def _default(obj: Any) -> str:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    return str(obj)


def canonical_json(body: Mapping[str, Any]) -> str:
    """Deterministic serialization: sorted keys, tight separators, ISO dates."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=_default)


def row_hash(prev_hash: str, body: Mapping[str, Any]) -> str:
    payload = prev_hash.encode("utf-8") + canonical_json(body).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chain_body(row: Mapping[str, Any]) -> dict[str, Any]:
    """The hashed portion of a row: everything except the hash fields."""
    return {k: v for k, v in row.items() if k not in _HASH_FIELDS}


class ChainFault(StrEnum):
    BAD_GENESIS = "bad_genesis"
    BROKEN_LINK = "broken_link"  # prev_hash != previous row_hash (delete/reorder)
    HASH_MISMATCH = "hash_mismatch"  # body tampered


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    fault: ChainFault | None = None
    index: int | None = None  # first offending row
    detail: str = ""


def verify_chain(rows: Sequence[Mapping[str, Any]]) -> ChainVerification:
    """Verify a full chain in storage order.

    Detects tampering (body changed), deletion, and reordering (both break the
    ``prev_hash`` linkage). An empty chain is trivially valid.
    """
    expected_prev = GENESIS
    for i, row in enumerate(rows):
        prev = str(row.get("prev_hash", ""))
        if prev != expected_prev:
            fault = ChainFault.BAD_GENESIS if i == 0 else ChainFault.BROKEN_LINK
            return ChainVerification(
                False, fault, i,
                f"row {i}: prev_hash {prev[:12]}… != expected {expected_prev[:12]}…",
            )
        recomputed = row_hash(prev, chain_body(row))
        stored = str(row.get("row_hash", ""))
        if recomputed != stored:
            return ChainVerification(
                False, ChainFault.HASH_MISMATCH, i,
                f"row {i}: stored row_hash does not match recomputed body hash",
            )
        expected_prev = stored
    return ChainVerification(True)
