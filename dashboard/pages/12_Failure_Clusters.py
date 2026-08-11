"""Failure-clusters page (plan Phase 6).

The latest failure-clustering run: failed decisions grouped by failure class
and semantic similarity of their review texts, with the ImprovementProposals
those clusters opened. Proposals are OPEN for human review — the iron law
holds: nothing a cluster suggests is ever applied without approval through
the promotion pipeline.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, safe_df

from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Failure Clusters", layout="wide")
st.title("Failure clusters")

st.info(
    "Clusters of at least 5 failed decisions open an ImprovementProposal "
    "with their suggested kind (claude_reasoning → prompt_revision, "
    "quant_model → retrain, sizing/risk → threshold_change, data_quality → "
    "new_data_source, strategy/regime → strategy_retirement). Smaller "
    "clusters are listed for context and open NOTHING. Approving a proposal "
    "is a human decision in the promotion pipeline — never automatic."
)

fc = schema.failure_clusters
clusters = safe_df(
    select(
        fc.c.run_as_of,
        fc.c.label,
        fc.c.failure_class,
        fc.c.size,
        fc.c.suggested_kind,
        fc.c.example_refs,
        fc.c.proposal_id,
    ).order_by(fc.c.run_as_of.desc(), fc.c.size.desc(), fc.c.label)
)
if empty_note(
    clusters,
    "No clustering runs yet — the failure_clustering sensor triggers a run "
    "every ~150 new resolved failures (manual materialization works too).",
):
    st.stop()

latest_day = clusters["run_as_of"].max()
latest = clusters[clusters["run_as_of"] == latest_day].copy()
st.caption(
    f"Latest clustering run: {latest_day} — {len(latest)} clusters over "
    f"{int(latest['size'].sum())} failed decisions"
)
st.dataframe(latest, width="stretch", hide_index=True)

st.subheader("Dominant failure classes")
by_class = (
    latest.groupby("failure_class")["size"].sum().sort_values(ascending=False).to_frame()
)
st.bar_chart(by_class)

# --- linked open proposals ---------------------------------------------------

st.subheader("Open improvement proposals from clustering")
ip = schema.improvement_proposals
proposal_ids = [p for p in latest["proposal_id"].dropna().tolist() if p]
if not proposal_ids:
    st.info(
        "The latest run opened no proposals — no cluster reached the minimum "
        "size. Thin evidence deliberately opens nothing."
    )
    st.stop()
proposals = safe_df(
    select(
        ip.c.id,
        ip.c.created_at,
        ip.c.kind,
        ip.c.target,
        ip.c.expected_effect,
        ip.c.evidence,
        ip.c.status,
    )
    .where(ip.c.id.in_(proposal_ids), ip.c.status == "open")
    .order_by(ip.c.created_at.desc())
)
if empty_note(
    proposals,
    "No linked proposals are still open — they have been reviewed "
    "(accepted/rejected) already.",
):
    st.stop()
for record in proposals.to_dict(orient="records"):
    with st.expander(f"{record['kind']} on `{record['target']}` — {record['status']}"):
        st.write(record["expected_effect"] or "no expected effect stated")
        evidence = pd.DataFrame.from_records(list(record["evidence"] or []))
        if not evidence.empty:
            st.dataframe(evidence, width="stretch", hide_index=True)
