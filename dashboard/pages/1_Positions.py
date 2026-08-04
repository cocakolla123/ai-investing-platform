"""Positions screen (doc 11 §3.2)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, latest_closes, safe_df

from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Positions", layout="wide")
st.title("Positions")

positions = safe_df(select(schema.positions).order_by(schema.positions.c.symbol))
if empty_note(positions, "No open positions — nothing has been bought yet."):
    st.stop()

closes = latest_closes(list(positions["symbol"]))
positions["last_price"] = positions["symbol"].map(closes.get)
positions["unrealized_pnl"] = (
    (positions["last_price"] - positions["avg_entry_price"]) * positions["qty"]
).where(positions["last_price"].notna())

# Entry decision-context refs: the "why do I own this" click is one hop.
contexts = safe_df(
    select(
        schema.decision_context.c.symbol,
        schema.decision_context.c.id.label("decision_context_id"),
        schema.decision_context.c.verdict_id,
        schema.decision_context.c.created_at,
    ).order_by(schema.decision_context.c.created_at)
)
if not contexts.empty:
    latest_ctx = contexts.groupby("symbol").tail(1)
    positions = positions.merge(
        latest_ctx[["symbol", "decision_context_id", "verdict_id"]], on="symbol", how="left"
    )
else:
    positions["decision_context_id"] = None
    positions["verdict_id"] = None

show_cols = [
    "symbol",
    "qty",
    "avg_entry_price",
    "last_price",
    "unrealized_pnl",
    "current_stop",
    "stop_kind",
    "decision_context_id",
    "verdict_id",
    "updated_at",
]
st.dataframe(positions[show_cols], width="stretch", hide_index=True)

missing_stops = positions[positions["current_stop"].isna()]
if not missing_stops.empty:
    st.warning(
        "Positions without a stop (pending exit review): "
        + ", ".join(missing_stops["symbol"])
    )

st.subheader("Entry context")
selected = st.selectbox("Symbol", positions["symbol"])
ctx = safe_df(
    select(schema.decision_context)
    .where(schema.decision_context.c.symbol == selected)
    .order_by(schema.decision_context.c.created_at.desc())
    .limit(1)
)
if not empty_note(ctx, "No decision context recorded for this position."):
    row = ctx.iloc[0]
    st.caption(
        f"{row['id']} · sleeve {row['sleeve']} · verdict {row['verdict_id']} · "
        f"expected {row['expected_upside_pct']:+.1%} / {row['expected_downside_pct']:+.1%} "
        f"over {row['horizon_days']}d"
    )
    st.json(row["payload"] if isinstance(row["payload"], dict) else {}, expanded=False)

st.subheader("Lots")
lots = positions.loc[positions["symbol"] == selected, "lots"].iloc[0]
if lots:
    st.dataframe(pd.DataFrame(lots), width="stretch", hide_index=True)
else:
    st.info("No lot detail stored for this position.")
