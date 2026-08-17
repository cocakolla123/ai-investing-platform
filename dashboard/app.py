"""Overview screen (doc 11 §3.1) — entry point of the Streamlit dashboard.

Run: ``uv run streamlit run dashboard/app.py``
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import api_post, empty_note, latest_closes, safe_df

from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Overview", layout="wide")
st.title("Overview")

# --- Equity curve vs SPY (paper account) ---

st.subheader("Equity vs SPY")
equity = safe_df(select(schema.equity_curve).order_by(schema.equity_curve.c.as_of))
if not empty_note(equity, "No equity history yet — the nightly loop has not run."):
    window = st.selectbox("Window (days)", [30, 90, 180, 365], index=1)
    eq = equity.tail(window).set_index("as_of")
    chart = pd.DataFrame({"equity": eq["equity"] / eq["equity"].iloc[0] * 100.0})
    spy = safe_df(
        select(schema.bars_1d.c.ts, schema.bars_1d.c.close).where(
            schema.bars_1d.c.symbol == "SPY"
        )
    )
    if not spy.empty:
        spy = spy.sort_values("ts").tail(window)
        spy_series = spy.set_index(spy["ts"].dt.date)["close"]
        chart["SPY"] = (spy_series / spy_series.iloc[0] * 100.0).reindex(chart.index)
    st.line_chart(chart)

    latest_row = equity.iloc[-1]
    cash_pct = latest_row["cash"] / latest_row["equity"] * 100 if latest_row["equity"] else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${latest_row['equity']:,.0f}")
    c2.metric("Cash %", f"{cash_pct:.1f}%")
    c3.metric("Gross exposure", f"{latest_row['gross_exposure']:.2f}")

# --- Regime badge + ladder state ---

col_regime, col_ladder = st.columns(2)
with col_regime:
    st.subheader("Regime")
    regime = safe_df(
        select(schema.regime_states).order_by(schema.regime_states.c.as_of.desc()).limit(1)
    )
    if not empty_note(regime, "No regime state yet."):
        row = regime.iloc[0]
        st.metric("Regime", str(row["regime"]), help=f"composite inputs: {row['inputs']}")
        st.caption(f"score {row['composite_score']:+.2f}, {row['sessions_in_state']} sessions")

with col_ladder:
    st.subheader("Drawdown ladder")
    ladder = safe_df(
        select(schema.ladder_states).order_by(schema.ladder_states.c.as_of.desc()).limit(1)
    )
    if not empty_note(ladder, "No ladder state yet."):
        row = ladder.iloc[0]
        st.metric("Rung", str(row["rung"]))
        st.caption(f"drawdown {row['drawdown']:+.2%} from HWM ${row['hwm']:,.0f}")

# --- Allocation ---

st.subheader("Allocation")
positions = safe_df(select(schema.positions))
if not empty_note(positions, "No open positions."):
    from dynamo.domain.symbols import is_crypto_symbol

    closes = latest_closes(list(positions["symbol"]))
    positions["price"] = (
        positions["symbol"].map(closes.get).fillna(positions["avg_entry_price"])
    )
    positions["value"] = positions["qty"] * positions["price"]
    # Sleeve split (ADR-0013): crypto pairs ("BTC/USD") are their own sleeve.
    positions["sleeve"] = positions["symbol"].map(
        lambda s: "crypto" if is_crypto_symbol(str(s)) else "equities"
    )
    col_sym, col_sleeve = st.columns([3, 1])
    with col_sym:
        st.bar_chart(positions.set_index("symbol")["value"])
    with col_sleeve:
        st.bar_chart(positions.groupby("sleeve")["value"].sum())

# --- Abstention rate (rolling 30d), a health indicator (doc 00 §4) ---

st.subheader("Abstention rate (30d)")
cutoff = (datetime.now(UTC) - timedelta(days=30)).date()
verdicts = safe_df(
    select(schema.committee_verdicts.c.action).where(schema.committee_verdicts.c.as_of >= cutoff)
)
if not empty_note(verdicts, "No committee verdicts in the last 30 days."):
    rate = (verdicts["action"] == "no_action").mean()
    st.metric("Abstention", f"{rate:.0%}", help="Healthy committees say 'no action' often.")

# --- Alert strip ---

st.subheader("Unacked alerts")
alerts = safe_df(
    select(schema.alerts)
    .where(schema.alerts.c.acked_at.is_(None))
    .order_by(schema.alerts.c.created_at.desc())
)
if not empty_note(alerts, "No unacked alerts."):
    for _, alert in alerts.iterrows():
        icon = {"page": ":red_circle:", "warn": ":large_orange_circle:"}.get(
            str(alert["severity"]), ":large_blue_circle:"
        )
        c_msg, c_btn = st.columns([5, 1])
        c_msg.markdown(f"{icon} **{alert['name']}** — {alert['message']}")
        if bool(alert["ackable"]):
            if c_btn.button("Ack", key=f"ack-{alert['id']}"):
                api_post(f"/alerts/{alert['id']}/ack", "alert-ack")
                st.rerun()
        else:
            c_btn.caption("clears with condition")
