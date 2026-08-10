"""Performance & attribution screen (doc 11 §3.5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import func, select
from streamlit_helpers import empty_note, safe_df

from dynamo.config.settings import get_settings
from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Performance", layout="wide")
st.title("Performance & attribution")

# --- Attribution from trades joined to orders (sleeve tags) ---

st.subheader("Attribution")
t, o = schema.trades, schema.orders
trades = safe_df(
    select(
        o.c.sleeve,
        t.c.symbol,
        t.c.side,
        func.sum(t.c.qty * t.c.fill_price).label("notional"),
        func.sum(t.c.fees).label("fees"),
        func.count().label("fills"),
    )
    .join_from(t, o, t.c.client_order_id == o.c.client_order_id, isouter=True)
    .group_by(o.c.sleeve, t.c.symbol, t.c.side)
)
if not empty_note(trades, "No fills yet — attribution appears after the first trades."):
    pivot = trades.pivot_table(
        index=["sleeve", "symbol"], columns="side", values="notional", aggfunc="sum"
    ).fillna(0.0)
    sold = pivot["sell"] if "sell" in pivot.columns else 0.0
    bought = pivot["buy"] if "buy" in pivot.columns else 0.0
    pivot["net_cash_flow"] = sold - bought
    st.dataframe(pivot.reset_index(), width="stretch", hide_index=True)
    by_sleeve = pivot.groupby("sleeve")["net_cash_flow"].sum()
    st.bar_chart(by_sleeve)

# --- Realized slippage vs model ---

st.subheader("Realized slippage")
slippage = safe_df(
    select(t.c.symbol, t.c.slippage_bps, t.c.filled_at).where(t.c.slippage_bps.is_not(None))
)
if not empty_note(slippage, "No slippage measurements yet."):
    st.metric("Mean slippage (bps)", f"{slippage['slippage_bps'].mean():+.1f}")
    st.line_chart(slippage.set_index("filled_at")["slippage_bps"])

# --- Calibration: the primary "is the AI trustworthy" view (doc 10) ---

st.subheader("Calibration")
cal = safe_df(select(schema.calibration_points).order_by(schema.calibration_points.c.as_of))
if not empty_note(cal, "No calibration points yet — needs resolved predictions."):
    scopes = sorted(set(zip(cal["scope_type"], cal["scope_id"], strict=False)))
    scope = st.selectbox("Scope", scopes, format_func=lambda s: f"{s[0]}:{s[1]}")
    subset = cal[(cal["scope_type"] == scope[0]) & (cal["scope_id"] == scope[1])]

    st.markdown("**Brier trend** (lower is better)")
    st.line_chart(subset.set_index("as_of")["brier"])

    st.markdown("**Reliability diagram** (latest)")
    buckets = subset.iloc[-1]["bucket_stats"]
    if isinstance(buckets, dict) and buckets:
        rows = [
            {
                "bucket": name,
                "predicted": stats.get("predicted"),
                "observed": stats.get("observed"),
                "n": stats.get("n"),
            }
            for name, stats in buckets.items()
            if isinstance(stats, dict)
        ]
        if rows:
            frame = pd.DataFrame(rows).set_index("bucket")
            st.line_chart(frame[["predicted", "observed"]])
            st.dataframe(frame, width="stretch")
    else:
        st.info("No bucket stats on the latest calibration point.")

# --- Backtest tearsheets (quantstats HTML written next to each run report) ---

st.subheader("Backtest tearsheets")
_backtests_dir = Path(get_settings().artifact_dir) / "backtests"
_tearsheets = (
    sorted(_backtests_dir.glob("*/*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if _backtests_dir.exists()
    else []
)
if not _tearsheets:
    st.info("No tearsheets yet — they appear after an event-driven backtest run.")
else:
    for sheet in _tearsheets:
        run_name = sheet.parent.name
        col_name, col_dl = st.columns([5, 1])
        col_name.markdown(f"**{run_name}** — `{sheet.name}`")
        try:
            col_dl.download_button(
                "Download",
                data=sheet.read_bytes(),
                file_name=f"{run_name}-{sheet.name}",
                mime="text/html",
                key=f"tearsheet-{run_name}-{sheet.name}",
            )
        except OSError:
            col_dl.caption("unreadable")
