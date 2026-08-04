"""Deterministic client order ids (doc 09 §4).

The same intent always hashes to the same id, so a crash/retry resubmission
cannot duplicate: the broker rejects the duplicate id and the order manager
treats that as "already submitted — go fetch its state".
"""

from __future__ import annotations

import hashlib

CLIENT_ORDER_ID_LEN = 24


def client_order_id(target_portfolio_id: str, symbol: str, side: str, ts_bucket: str) -> str:
    """``sha256(target_portfolio_id|symbol|side|ts_bucket)[:24]`` (doc 09 §4).

    ``ts_bucket`` is the scheduling slot: the nightly run date, or the 15-minute
    intraday slot for stop maintenance.
    """
    raw = f"{target_portfolio_id}|{symbol}|{side}|{ts_bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:CLIENT_ORDER_ID_LEN]
