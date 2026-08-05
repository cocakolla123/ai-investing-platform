"""Activity Log (doc 11 §3 pattern): the daily report, readable in one place.

Stored reports come from the ``daily_reports`` table (written 02:15 ET by the
daily_report asset). For today — before that job has run — the page builds the
same report live from the database. Renders cleanly on an empty database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, get_engine, safe_df

from dynamo.config.settings import get_settings
from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Activity Log", layout="wide")
st.title("Activity Log")

today = datetime.now(UTC).date()
t = schema.daily_reports
dates_df = safe_df(select(t.c.as_of).order_by(t.c.as_of.desc()))

# --- Stored daily reports (picker defaults to the latest report date) ---

if not empty_note(
    dates_df,
    "No stored daily reports yet — the 02:15 ET report job writes the first one "
    "after a full day of activity.",
):
    picked = st.selectbox("Report date", list(dates_df["as_of"]), index=0)
    stored = safe_df(select(t.c.markdown).where(t.c.as_of == picked))
    if stored.empty:
        st.info(f"Report for {picked} is missing — try another date.")
    else:
        st.markdown(str(stored["markdown"].iloc[0]))

# --- Today so far (live, until the stored report lands) ---

has_today = (not dates_df.empty) and any(d == today for d in dates_df["as_of"])
if not has_today:
    st.divider()
    st.subheader("Today so far")
    st.caption(
        "Built live from the database right now; the stored copy is written at 02:15 ET."
    )
    try:
        from dynamo.reporting.daily import render_markdown
        from dynamo.reporting.sql_source import build_daily_report_for_engine

        report = build_daily_report_for_engine(
            get_engine(), today, get_settings().claude_monthly_budget_usd
        )
        st.markdown(render_markdown(report))
    except Exception:  # pragma: no cover - defensive: keep the page rendering
        st.info("Database unavailable — nothing to show for today yet.")
