"""Research screen (doc 11 §3.3): watchlist, verdict detail, dossiers.

Dissents and self-critique are rendered ALWAYS — never collapsed (doc 03 §2).
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st
from sqlalchemy import func, select
from streamlit_helpers import empty_note, get_engine, safe_df

from dynamo.persistence import schema
from dynamo.persistence.repositories.market import MarketDataRepository

st.set_page_config(page_title="Dynamo — Research", layout="wide")
st.title("Research")

# --- Ranked watchlist ---

st.subheader("Watchlist")
latest = safe_df(select(func.max(schema.watchlist.c.as_of).label("as_of")))
as_of = latest["as_of"].iloc[0] if not latest.empty else None
watch = (
    safe_df(
        select(schema.watchlist)
        .where(schema.watchlist.c.as_of == as_of)
        .order_by(schema.watchlist.c.sleeve, schema.watchlist.c.rank)
    )
    if as_of is not None
    else latest.iloc[0:0]
)
if not empty_note(watch, "No watchlist yet — the nightly loop has not produced one."):
    st.caption(f"as of {as_of}")
    st.dataframe(
        watch[
            [
                "sleeve",
                "rank",
                "symbol",
                "blend_score",
                "committee_component",
                "quant_component",
                "quality_component",
                "badges",
                "verdict_ref",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

# --- Committee report view per symbol ---

st.subheader("Committee verdict")
symbols = safe_df(select(schema.committee_verdicts.c.symbol).distinct())
if empty_note(symbols, "No committee verdicts yet."):
    st.stop()

symbol = st.selectbox("Symbol", sorted(symbols["symbol"]))
verdicts = safe_df(
    select(schema.committee_verdicts)
    .where(schema.committee_verdicts.c.symbol == symbol)
    .order_by(schema.committee_verdicts.c.as_of.desc())
)
if not verdicts.empty:
    v = verdicts.iloc[0]
    payload = v["payload"] if isinstance(v["payload"], dict) else {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Action", str(v["action"]))
    c2.metric("Conviction", f"{v['conviction']:.2f}")
    c3.metric("Disagreement", f"{v['disagreement_score']:.2f}")

    st.markdown(f"**Thesis** — {payload.get('thesis', '—')}")
    st.markdown(f"**Bull case** — {payload.get('bull_case', '—')}")
    st.markdown(f"**Bear case** — {payload.get('bear_case', '—')}")

    # Always rendered, never behind a click (doc 03 §2).
    st.markdown("**Dissents**")
    dissents = payload.get("dissents") or []
    if dissents:
        for d in dissents:
            st.markdown(f"- *{d.get('agent')}* ({d.get('stance')}): {d.get('reason')}")
    else:
        st.markdown("- none recorded")
    st.markdown(f"**Self-critique** — {payload.get('self_critique') or '—'}")

    if payload.get("exit_conditions"):
        st.markdown("**Exit conditions**")
        for cond in payload["exit_conditions"]:
            st.markdown(f"- {cond.get('description')}")
    if payload.get("quant_alignment"):
        st.caption(f"quant alignment: {payload['quant_alignment']}")

    with st.expander("Full verdict payload (evidence refs resolvable to source records)"):
        st.json(payload)

# --- Deep-dive request (queues for the next nightly set) ---

if st.button(
    f"Request deep-dive on {symbol}",
    help="Queues this symbol for the next nightly deep-dive set — "
    "it counts against the nightly cap (doc 03 §1).",
):
    MarketDataRepository(get_engine()).add_alert(
        severity="info",
        name="deep_dive_request",
        message=f"Deep-dive requested for {symbol} from the dashboard "
        "(counts against the nightly deep-dive cap)",
        ackable=True,
        created_at=datetime.now(UTC),
    )
    st.success(f"Deep-dive queued for {symbol} — counts against the nightly cap.")

# --- Dossier browser ---

st.subheader("Dossier")
dossiers = safe_df(
    select(schema.dossiers)
    .where(schema.dossiers.c.symbol == symbol)
    .order_by(schema.dossiers.c.as_of.desc())
)
if not empty_note(dossiers, "No dossier for this symbol yet."):
    d = dossiers.iloc[0]
    payload = d["payload"] if isinstance(d["payload"], dict) else {}
    st.caption(f"as of {d['as_of']} · prompt {d['prompt_version']} · model {d['model_id']}")
    st.markdown(payload.get("business_summary", "—"))
    gaps = payload.get("data_gaps") or []
    if gaps:
        st.warning("Data gaps: " + "; ".join(gaps))
    with st.expander("Full dossier (fact base, filing deltas, technicals)"):
        st.json(payload)
