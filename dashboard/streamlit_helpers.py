"""Shared helpers for the Streamlit dashboard (doc 11 §2, M1-M5 phase).

Streamlit reads DIRECTLY from the database via repositories/SQL (same trust
boundary as the API). Admin actions go through the API over httpx so the
confirm-header + audit path is exercised. Every read is guarded so the app
renders cleanly on an empty (or unreachable) database.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from dynamo.config.settings import get_settings
from dynamo.persistence.db import make_engine

API_URL = os.environ.get("DYNAMO_API_URL", "http://127.0.0.1:8000")


@st.cache_resource
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


def safe_df(stmt: Any) -> pd.DataFrame:
    """Run a SQLAlchemy select into a DataFrame; empty frame on any failure."""
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(stmt, conn)
    except SQLAlchemyError:
        st.warning("Database unavailable — showing empty data.")
        return pd.DataFrame()


def empty_note(frame: pd.DataFrame, message: str) -> bool:
    """True (and renders an info box) when there is nothing to show yet."""
    if frame.empty:
        st.info(message)
        return True
    return False


def _api_auth() -> tuple[str, str] | None:
    password = os.environ.get("DYNAMO_DASHBOARD_PASSWORD")
    return ("dynamo", password) if password else None


def api_post(path: str, confirm: str, body: dict[str, Any] | None = None) -> None:
    """POST an admin action to the API with its confirm header; toast result."""
    try:
        resp = httpx.post(
            f"{API_URL}{path}",
            headers={"X-Dynamo-Confirm": confirm},
            json=body,
            auth=_api_auth(),
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")
        return
    if resp.status_code >= 400:
        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
    else:
        st.success(f"{path} → {resp.json()}")


def latest_closes(symbols: list[str]) -> dict[str, float]:
    """Most recent canonical close per symbol (empty dict when no bars)."""
    if not symbols:
        return {}
    from sqlalchemy import select

    from dynamo.persistence import schema

    t = schema.bars_1d
    frame = safe_df(select(t.c.symbol, t.c.ts, t.c.close).where(t.c.symbol.in_(symbols)))
    if frame.empty:
        return {}
    frame = frame.sort_values("ts").groupby("symbol").tail(1)
    return dict(zip(frame["symbol"], frame["close"], strict=False))
