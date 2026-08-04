"""Ops-lite screen (doc 11 §3.8): runs, spend, quarantine, approval queues.

Reads come straight from the database; approve/reject/killswitch go through
the API so the confirm-header + audit trail is always exercised.
"""

from __future__ import annotations

import streamlit as st
from sqlalchemy import func, select
from streamlit_helpers import api_post, empty_note, safe_df

from dynamo.config.settings import get_settings
from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Ops", layout="wide")
st.title("Ops")

# --- Pipeline runs (batch jobs) ---

st.subheader("Pipeline runs")
jobs = safe_df(
    select(schema.batch_jobs).order_by(schema.batch_jobs.c.submitted_at.desc()).limit(20)
)
if not empty_note(jobs, "No batch jobs recorded yet."):
    st.dataframe(jobs, width="stretch", hide_index=True)

# --- Claude spend vs budget ---

st.subheader("Claude spend")
spend = safe_df(
    select(func.coalesce(func.sum(schema.cost_ledger.c.cost_usd), 0.0).label("mtd")).where(
        func.date_trunc("month", schema.cost_ledger.c.ts)
        == func.date_trunc("month", func.now())
    )
)
mtd = float(spend["mtd"].iloc[0]) if not spend.empty else 0.0
budget = get_settings().claude_monthly_budget_usd
utilization = min(mtd / budget, 1.0) if budget > 0 else 1.0
st.progress(utilization, text=f"${mtd:.2f} of ${budget:.2f} month-to-date")
if utilization >= 0.7:
    st.warning("Spend at or above 70% of the monthly budget.")

# --- Quarantine ---

st.subheader("Data-quality quarantine")
quarantine = safe_df(
    select(schema.quarantine)
    .where(schema.quarantine.c.resolved_at.is_(None))
    .order_by(schema.quarantine.c.created_at.desc())
)
if not empty_note(quarantine, "Quarantine is empty."):
    st.dataframe(
        quarantine[["scope", "key", "reason", "created_at"]], width="stretch", hide_index=True
    )

# --- Universe-change queue ---

st.subheader("Universe-change queue")
universe = safe_df(
    select(schema.alerts)
    .where(
        schema.alerts.c.name == "universe_change_recommendation",
        schema.alerts.c.acked_at.is_(None),
    )
    .order_by(schema.alerts.c.created_at)
)
if not empty_note(universe, "No pending universe-change recommendations."):
    for _, rec in universe.iterrows():
        c_msg, c_approve, c_reject = st.columns([4, 1, 1])
        c_msg.markdown(f"**#{rec['id']}** {rec['message']}")
        if c_approve.button("Approve", key=f"uni-approve-{rec['id']}"):
            api_post(f"/universe/queue/{rec['id']}/approve", "universe-approve")
            st.rerun()
        if c_reject.button("Reject", key=f"uni-reject-{rec['id']}"):
            api_post(f"/universe/queue/{rec['id']}/reject", "universe-reject")
            st.rerun()

# --- Improvement-proposal queue ---

st.subheader("Improvement proposals")
proposals = safe_df(
    select(schema.improvement_proposals)
    .where(schema.improvement_proposals.c.status == "open")
    .order_by(schema.improvement_proposals.c.created_at)
)
if not empty_note(proposals, "No open improvement proposals."):
    for _, prop in proposals.iterrows():
        with st.container(border=True):
            st.markdown(f"**{prop['id']}** · {prop['kind']} · target `{prop['target']}`")
            st.markdown(prop["expected_effect"] or "_no expected effect stated_")
            c_approve, c_reject, _sp = st.columns([1, 1, 4])
            if c_approve.button("Approve", key=f"imp-approve-{prop['id']}"):
                api_post(f"/improvements/{prop['id']}/approve", "improvement-approve")
                st.rerun()
            if c_reject.button("Reject", key=f"imp-reject-{prop['id']}"):
                api_post(f"/improvements/{prop['id']}/reject", "improvement-reject")
                st.rerun()

# --- Rebalance proposals awaiting approval ---

st.subheader("Rebalance proposals")
targets = safe_df(
    select(
        schema.target_portfolios.c.id,
        schema.target_portfolios.c.as_of,
        schema.target_portfolios.c.approved_at,
        schema.target_portfolios.c.payload,
    )
    .where(schema.target_portfolios.c.approved_at.is_(None))
    .order_by(schema.target_portfolios.c.as_of.desc())
    .limit(10)
)
if targets.empty:
    pending = targets
else:
    not_rejected = targets["payload"].map(
        lambda p: not (isinstance(p, dict) and bool(p.get("rejected")))
    )
    pending = targets.loc[not_rejected]
if not empty_note(pending, "No rebalance proposals awaiting approval."):
    for _, tp in pending.iterrows():
        c_msg, c_approve, c_reject = st.columns([4, 1, 1])
        c_msg.markdown(f"**{tp['id']}** as of {tp['as_of']}")
        if c_approve.button("Approve", key=f"tp-approve-{tp['id']}"):
            api_post(f"/rebalance/{tp['id']}/approve", "rebalance-approve")
            st.rerun()
        if c_reject.button("Reject", key=f"tp-reject-{tp['id']}"):
            api_post(f"/rebalance/{tp['id']}/reject", "rebalance-reject")
            st.rerun()

# --- Kill-switch ---

st.subheader("Kill-switch")
note = st.text_input("Note (required to disengage)")
c_on, c_off = st.columns(2)
if c_on.button("Engage kill-switch", type="primary"):
    api_post("/killswitch", "killswitch", {"engage": True, "note": note})
if c_off.button("Disengage kill-switch"):
    if not note.strip():
        st.error("Disengaging requires a non-empty note (explicit acknowledgment).")
    else:
        api_post("/killswitch", "killswitch", {"engage": False, "note": note})
