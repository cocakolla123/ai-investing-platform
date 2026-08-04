"""Prediction markets screen (doc 11 §3.7). Research only — no execution."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, safe_df

from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Prediction Markets", layout="wide")
st.title("Prediction markets")

# Mandatory banner, verbatim (doc 00 §7, doc 03 §6).
st.warning("Research only. No execution hooks exist.")

views = safe_df(
    select(schema.prediction_market_views).order_by(schema.prediction_market_views.c.as_of.desc())
)
if empty_note(views, "No prediction-market views yet."):
    st.stop()

# Latest view per market; uncertainty columns are mandatory UI copy.
views = views.drop_duplicates(subset=["venue", "market_id"], keep="first")
rows = []
for _, v in views.iterrows():
    raw = v["payload"]
    p: dict = raw if isinstance(raw, dict) else {}
    rows.append(
        {
            "question": p.get("question", v["market_id"]),
            "venue": v["venue"],
            "as_of": v["as_of"],
            "market_implied": p.get("market_implied_prob"),
            "estimated": p.get("estimated_prob"),
            "edge": p.get("edge"),
            "confidence": p.get("confidence"),
            "uncertainties": "; ".join(p.get("uncertainties") or []),
            "why_wrong": "; ".join(p.get("why_wrong") or []),
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# Thin-market trust flags from the latest stored prices.
prices = safe_df(
    select(schema.prediction_market_prices).order_by(schema.prediction_market_prices.c.ts.desc())
)
if not prices.empty:
    latest_prices = prices.drop_duplicates(subset=["venue", "market_id"], keep="first")
    untrusted = latest_prices[~latest_prices["trusted"]]
    if not untrusted.empty:
        st.error(
            "Untrusted (thin-market) quotes: "
            + ", ".join(f"{r['venue']}/{r['market_id']}" for _, r in untrusted.iterrows())
        )

# Calibration over resolved markets.
st.subheader("Calibration (resolved markets)")
markets = safe_df(
    select(schema.prediction_markets).where(schema.prediction_markets.c.resolved_outcome.is_not(None))
)
if empty_note(markets, "No resolved markets yet."):
    st.stop()
resolved = markets.merge(views, on=["venue", "market_id"], how="inner")
if resolved.empty:
    st.info("No views on resolved markets yet.")
else:
    resolved["estimated"] = resolved["payload"].map(
        lambda p: p.get("estimated_prob") if isinstance(p, dict) else None
    )
    resolved["outcome"] = resolved["resolved_outcome"].astype(float)
    st.dataframe(
        resolved[["venue", "market_id", "question", "estimated", "outcome"]],
        width="stretch",
        hide_index=True,
    )
