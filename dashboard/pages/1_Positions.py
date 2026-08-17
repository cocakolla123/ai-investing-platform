"""Positions screen (doc 11 §3.2)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, latest_closes, safe_df

from dynamo.domain.symbols import is_crypto_symbol
from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Positions", layout="wide")
st.title("Positions")


def _crypto_section(crypto: pd.DataFrame) -> None:
    """Crypto sleeve (ADR-0013): positions + exposure vs the sleeve cap."""
    from dynamo.config.settings import get_settings

    st.subheader("Crypto (Alpaca crypto sleeve)")
    if empty_note(crypto, "No open crypto positions."):
        return
    crypto = crypto.copy()
    closes = latest_closes(list(crypto["symbol"]))
    crypto["last_price"] = crypto["symbol"].map(closes.get)
    crypto["value"] = (crypto["qty"] * crypto["last_price"]).where(
        crypto["last_price"].notna()
    )
    crypto["unrealized_pnl"] = (
        (crypto["last_price"] - crypto["avg_entry_price"]) * crypto["qty"]
    ).where(crypto["last_price"].notna())
    settings = get_settings()
    eq = schema.equity_curve
    equity_df = safe_df(select(eq).order_by(eq.c.as_of.desc()).limit(1))
    exposure = float(crypto["value"].fillna(0.0).sum())
    if not equity_df.empty:
        equity = float(equity_df.iloc[0]["equity"])
        cap = settings.crypto_sleeve_cap * equity
        col1, col2, col3 = st.columns(3)
        col1.metric("Crypto exposure", f"${exposure:,.0f}")
        col2.metric(
            "Sleeve cap",
            f"${cap:,.0f}",
            f"{exposure / cap:.0%} used" if cap > 0 else None,
        )
        col3.metric("Open pairs", int((crypto["qty"] != 0).sum()))
    else:
        st.metric("Crypto exposure", f"${exposure:,.0f} (no equity curve yet)")
    st.dataframe(
        crypto[
            [
                "symbol",
                "qty",
                "avg_entry_price",
                "last_price",
                "value",
                "unrealized_pnl",
                "current_stop",
                "stop_kind",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Crypto trades 24/7 — stops are maintained on the weekday 15-minute "
        "loop plus an every-2h weekend schedule."
    )


def _options_section() -> None:
    """Options sleeve (ADR-0010): positions, premium at risk vs cap, expiries."""
    from dynamo.config.settings import get_settings
    from dynamo.domain.models.options import CONTRACT_MULTIPLIER, OptionStructure

    st.subheader("Options (defined-risk sleeve)")
    opts = safe_df(
        select(schema.options_positions).order_by(
            schema.options_positions.c.expiry, schema.options_positions.c.occ_symbol
        )
    )
    if empty_note(opts, "No open option positions."):
        return

    def _risk(row: pd.Series) -> float:
        structure = str(row["structure"])
        if structure in (OptionStructure.LONG_CALL.value, OptionStructure.LONG_PUT.value):
            return float(row["contracts"]) * float(row["avg_price"]) * CONTRACT_MULTIPLIER
        if structure == OptionStructure.CASH_SECURED_PUT.value:
            return float(row["contracts"]) * float(row["strike"]) * CONTRACT_MULTIPLIER
        return 0.0  # covered call — share-covered

    opts["premium_at_risk"] = opts.apply(_risk, axis=1)
    opts["days_to_expiry"] = opts["expiry"].map(
        lambda d: (pd.Timestamp(d).date() - pd.Timestamp.now().date()).days
    )
    settings = get_settings()
    eq = schema.equity_curve
    equity_df = safe_df(select(eq).order_by(eq.c.as_of.desc()).limit(1))
    total_risk = float(opts["premium_at_risk"].sum())
    if not equity_df.empty:
        equity = float(equity_df.iloc[0]["equity"])
        cap = settings.options_premium_at_risk_cap * equity
        col1, col2, col3 = st.columns(3)
        col1.metric("Premium at risk", f"${total_risk:,.0f}")
        col2.metric(
            "Sleeve cap",
            f"${cap:,.0f}",
            f"{total_risk / cap:.0%} used" if cap > 0 else None,
        )
        col3.metric("Open contracts", int(opts["contracts"].sum()))
    else:
        st.metric("Premium at risk", f"${total_risk:,.0f} (no equity curve yet)")
    st.dataframe(
        opts[
            [
                "occ_symbol",
                "underlying",
                "structure",
                "side",
                "contracts",
                "strike",
                "avg_price",
                "premium_at_risk",
                "expiry",
                "days_to_expiry",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    soon = opts[opts["days_to_expiry"] <= 7]
    if not soon.empty:
        st.warning("Expiring within 7 days: " + ", ".join(soon["occ_symbol"]))


all_positions = safe_df(select(schema.positions).order_by(schema.positions.c.symbol))
if all_positions.empty:
    st.info("No open positions — nothing has been bought yet.")
    _crypto_section(all_positions)
    _options_section()
    st.stop()

# Crypto pairs ("BTC/USD") share the positions table but render in their own
# sleeve section below (ADR-0013); the main table is stocks & ETFs only.
_is_crypto = all_positions["symbol"].map(is_crypto_symbol)
crypto_positions = all_positions[_is_crypto].reset_index(drop=True)
positions = all_positions[~_is_crypto].reset_index(drop=True)
if positions.empty:
    st.info("No open stock/ETF positions.")
    _crypto_section(crypto_positions)
    _options_section()
    st.stop()

st.subheader("Stocks & ETFs")
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

missing_stops = all_positions[all_positions["current_stop"].isna()]
if not missing_stops.empty:
    st.warning(
        "Positions without a stop (pending exit review): "
        + ", ".join(missing_stops["symbol"])
    )

st.subheader("Entry context")
selected = st.selectbox("Symbol", all_positions["symbol"])
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
lots = all_positions.loc[all_positions["symbol"] == selected, "lots"].iloc[0]
if lots:
    st.dataframe(pd.DataFrame(lots), width="stretch", hide_index=True)
else:
    st.info("No lot detail stored for this position.")

_crypto_section(crypto_positions)
_options_section()
