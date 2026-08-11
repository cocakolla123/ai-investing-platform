"""Calibration-by-subgroup page (plan Phase 6).

Browser over the hierarchical calibration scopes (plan Phase 2): every
``hier``-scope ``calibration_points`` row is a node on the path
``global → sleeve → strategy → regime → sector → quant_alignment →
event_type`` (path-string scope ids like ``sleeve=core|regime=risk_off``).
Filter by scope-path substring, inspect Brier + reliability per node — with
``n`` per scope front and center, because thin nodes shrink toward their
parent and a 5-observation Brier is noise, not calibration.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, safe_df

from dynamo.persistence import schema

MIN_MEANINGFUL_N = 30  # matches MIN_CONFORMAL_N: below this a node has no own fit

st.set_page_config(page_title="Dynamo — Calibration Subgroups", layout="wide")
st.title("Calibration by subgroup")

st.info(
    "Hierarchical scopes are ADVISORY (plan Phase 2): calibrated "
    "probabilities are logged onto experiences and reported — nothing "
    "consumes them for sizing. Nodes with n below "
    f"{MIN_MEANINGFUL_N} have no conformal fit of their own; they shrink "
    "toward their parent scope."
)

cp = schema.calibration_points
points = safe_df(
    select(cp.c.scope_id, cp.c.as_of, cp.c.brier, cp.c.n, cp.c.bucket_stats)
    .where(cp.c.scope_type == "hier")
    .order_by(cp.c.as_of.desc(), cp.c.scope_id)
)
if empty_note(
    points,
    "No hierarchical calibration points yet — the weekly calibration_update "
    "asset writes hier-scope rows once resolved live experiences exist "
    "(hierarchical_calibration_enabled).",
):
    st.stop()

latest_day = points["as_of"].max()
latest = points[points["as_of"] == latest_day].copy()
st.caption(f"Latest calibration snapshot: {latest_day} — {len(latest)} scope nodes")

query = st.text_input(
    "Filter scope path (substring, e.g. 'sleeve=core' or 'regime=risk_off')", ""
)
filtered = (
    latest[latest["scope_id"].str.contains(query, case=False, regex=False)]
    if query
    else latest
)
if empty_note(filtered, "No scope paths match that filter."):
    st.stop()

table = filtered[["scope_id", "brier", "n"]].sort_values(["scope_id"]).reset_index(drop=True)
table["own_fit"] = table["n"] >= MIN_MEANINGFUL_N
st.dataframe(table, width="stretch", hide_index=True)
thin = int((~table["own_fit"]).sum())
if thin:
    st.caption(
        f"{thin} of {len(table)} nodes are below n={MIN_MEANINGFUL_N} — their "
        "calibration comes from parent scopes, shown here for coverage only."
    )

# --- reliability for one scope ----------------------------------------------

st.subheader("Reliability by conviction bucket")
scope = st.selectbox("Scope path", table["scope_id"].tolist())
row = filtered[filtered["scope_id"] == scope].iloc[0]
st.metric("Observations (n)", int(row["n"]))
stats = row["bucket_stats"] or {}
buckets = sorted({key.rsplit("_", 1)[0] for key in stats if "_" in key})
records = [
    {
        "bucket": bucket,
        "n": stats.get(f"{bucket}_n", 0.0),
        "forecast": stats.get(f"{bucket}_forecast"),
        "observed": stats.get(f"{bucket}_observed"),
    }
    for bucket in buckets
]
reliability = pd.DataFrame.from_records(records)
if empty_note(reliability, "No populated conviction buckets for this scope yet."):
    st.stop()
st.dataframe(reliability, width="stretch", hide_index=True)
st.caption(
    "Perfect calibration puts forecast == observed in every bucket; the gap, "
    "weighted by n, is the expected calibration error."
)
st.line_chart(reliability.set_index("bucket")[["forecast", "observed"]])
