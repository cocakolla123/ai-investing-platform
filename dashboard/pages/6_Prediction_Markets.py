"""Prediction markets screen (doc 11 §3.7; ADR-0009 Kalshi trading sleeve)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, safe_df

from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Prediction Markets", layout="wide")
st.title("Prediction markets")

# Mandatory banner, verbatim (ADR-0009).
st.warning(
    "Kalshi: demo/live trading enabled behind the gate. "
    "Polymarket: research only — no execution hooks."
)

# --- Kalshi sleeve: positions / orders / P&L (ADR-0009) ---

st.subheader("Kalshi sleeve")
positions = safe_df(select(schema.pm_positions))
orders = safe_df(
    select(schema.pm_orders).order_by(schema.pm_orders.c.created_at.desc()).limit(50)
)
trades = safe_df(
    select(schema.pm_trades).order_by(schema.pm_trades.c.filled_at.desc()).limit(200)
)

if positions.empty and orders.empty and trades.empty:
    st.info("No Kalshi sleeve activity yet.")
else:
    realized_usd = 0.0
    if not trades.empty:
        settlements = trades[trades["kind"] == "settlement"]
        if not settlements.empty:
            realized_usd = float(settlements["realized_pnl_cents"].fillna(0.0).sum()) / 100.0
    exposure_usd = 0.0
    if not positions.empty:
        exposure_usd = float(
            (positions["contracts"] * positions["avg_price_cents"]).sum()
        ) / 100.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Open markets", 0 if positions.empty else len(positions))
    c2.metric("Exposure (cost basis)", f"${exposure_usd:,.2f}")
    c3.metric("Realized P&L", f"${realized_usd:,.2f}")

    if not positions.empty:
        st.caption("Open positions")
        st.dataframe(
            positions[
                ["market_id", "side", "contracts", "avg_price_cents", "question", "updated_at"]
            ],
            width="stretch",
            hide_index=True,
        )
    if not orders.empty:
        st.caption("Orders (latest 50)")
        st.dataframe(
            orders[
                [
                    "client_order_id",
                    "market_id",
                    "side",
                    "contracts",
                    "limit_price_cents",
                    "state",
                    "created_at",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    if not trades.empty:
        st.caption("Fills & settlements (latest 200)")
        st.dataframe(
            trades[
                [
                    "market_id",
                    "side",
                    "contracts",
                    "price_cents",
                    "kind",
                    "realized_pnl_cents",
                    "filled_at",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

# --- Cross-venue spreads (OddsPipe enrichment; ADR-0009 addendum) ---

st.subheader("Cross-venue spreads (OddsPipe)")
cross_venue = safe_df(
    select(schema.pm_cross_venue_spreads).order_by(schema.pm_cross_venue_spreads.c.ts.desc())
)
if cross_venue.empty:
    st.info("No cross-venue spreads yet (OddsPipe key unset or no matched pairs).")
else:
    latest_cv = cross_venue.drop_duplicates(subset=["kalshi_market_id"], keep="first")
    top_cv = latest_cv.reindex(
        latest_cv["yes_diff"].abs().sort_values(ascending=False).index
    ).head(10)
    st.caption("Top 10 matched Kalshi/Polymarket pairs by |YES spread|")
    st.dataframe(
        top_cv[
            [
                "question",
                "kalshi_market_id",
                "kalshi_yes",
                "polymarket_id",
                "polymarket_yes",
                "yes_diff",
                "ts",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

# --- Research views (doc 03 §6) ---

st.subheader("Committee views")
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
