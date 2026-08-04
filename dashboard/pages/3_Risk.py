"""Risk screen (doc 11 §3.4): limit utilization, VaR, stress, clusters."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, safe_df

from dynamo.domain.models.risk import RiskConfig
from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Risk", layout="wide")
st.title("Risk")

verdicts = safe_df(
    select(schema.risk_verdicts).order_by(schema.risk_verdicts.c.as_of.desc()).limit(1)
)
if empty_note(verdicts, "No risk verdicts yet — the risk engine has not run."):
    st.stop()

row = verdicts.iloc[0]
payload = row["payload"] if isinstance(row["payload"], dict) else {}
metrics = payload.get("metrics") or {}
config = RiskConfig()

st.caption(f"as of {row['as_of']} · ladder {payload.get('ladder', '?')}")
if payload.get("degraded"):
    st.error("Risk verdict is DEGRADED — inputs incomplete, no new risk allowed.")
if not payload.get("new_risk_allowed", True):
    st.warning("New risk is currently not allowed.")

# --- Limit utilization vs Medium-preset caps (doc 06 §2) ---

st.subheader("Limit utilization")


def _bar(label: str, value: float | None, cap: float | None) -> None:
    if value is None or cap in (None, 0):
        st.caption(f"{label}: no data")
        return
    utilization = min(max(value / cap, 0.0), 1.0)
    st.progress(utilization, text=f"{label}: {value:.3f} of cap {cap:.3f}")


_bar("Gross exposure", metrics.get("gross_exposure"), payload.get("gross_exposure_cap"))
_bar("VaR 95 (1d)", metrics.get("var95_1d"), config.var95_cap)
cash_fraction = metrics.get("cash_fraction")
if cash_fraction is not None:
    floor = payload.get("cash_floor", config.cash_floor)
    st.progress(
        min(max(cash_fraction, 0.0), 1.0),
        text=f"Cash: {cash_fraction:.1%} (floor {floor:.1%})",
    )

# --- VaR / CVaR ---

st.subheader("VaR / CVaR")
c1, c2, c3, c4 = st.columns(4)
c1.metric("VaR95 1d", f"{metrics.get('var95_1d', 0) or 0:.3%}")
c2.metric("VaR99 1d", f"{metrics.get('var99_1d', 0) or 0:.3%}")
c3.metric("CVaR95 1d", f"{metrics.get('cvar95_1d', 0) or 0:.3%}")
exceptions = int(metrics.get("var_exceptions_250d", 0) or 0)
c4.metric(
    "VaR exceptions (250d)",
    exceptions,
    help="Kupiec traffic light: green <10, investigate above.",
)

# --- Stress results ---

st.subheader("Deterministic stress")
stress = payload.get("stress") or payload.get("stress_results") or []
if stress:
    st.dataframe(pd.DataFrame(stress), width="stretch", hide_index=True)
else:
    st.info("No stress results in the latest risk verdict.")

# --- Correlation clusters vs cluster cap ---

st.subheader("Correlation clusters")
clusters = payload.get("clusters") or {}
if clusters:
    frame = pd.DataFrame(
        [{"symbol": sym, "cluster": cid} for sym, cid in clusters.items()]
    ).sort_values(["cluster", "symbol"])
    caps: dict[str, float] = dict(payload.get("cluster_caps") or {})
    frame["cluster_cap"] = frame["cluster"].map(caps.get).fillna(config.cluster_cap)
    st.dataframe(frame, width="stretch", hide_index=True)
else:
    st.info("No cluster assignments in the latest risk verdict.")

# --- Vetoes ---

vetoes = payload.get("vetoes") or []
if vetoes:
    st.subheader("Vetoes")
    st.dataframe(pd.DataFrame(vetoes), width="stretch", hide_index=True)
