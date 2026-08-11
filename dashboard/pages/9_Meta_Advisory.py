"""Meta-model advisory page (plan Phase 3): scores vs raw conviction.

ADVISORY-FIRST surface: the meta-model's calibrated win probability next to
the committee's raw conviction, joined to realized outcomes once experiences
resolve. Nothing here is a sizing input unless ``meta_sizing_enabled`` is
flipped after full promotion + human approval — the banner reflects the
ACTUAL setting.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select
from streamlit_helpers import empty_note, safe_df

from dynamo.config.settings import get_settings
from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Meta Advisory", layout="wide")
st.title("Meta-model advisory")

settings = get_settings()
if settings.meta_sizing_enabled:
    st.warning(
        "meta_sizing_enabled=True — sizing tilts CONSUME the meta-model's "
        "effective probability. This is legitimate only after full promotion "
        "with signed human approval."
    )
else:
    st.info(
        "**ADVISORY — not used for sizing (meta_sizing_enabled=False).** "
        "Scores below are logged and reported only; position sizing still "
        "uses raw committee conviction."
    )

# --- production model status ------------------------------------------------

mv = schema.model_versions
models = safe_df(
    select(mv.c.id, mv.c.algo, mv.c.status, mv.c.created_at, mv.c.metrics)
    .where(mv.c.model_type == "meta")
    .order_by(mv.c.created_at.desc())
)
st.subheader("Registry")
if empty_note(models, "No meta-models trained yet — run dynamo.cli.meta_train once "
                      "150 resolved live experiences exist."):
    pass
else:
    st.dataframe(models.drop(columns=["metrics"]), width="stretch")

# --- latest scores vs raw conviction ---------------------------------------

t, v = schema.meta_decision_scores, schema.committee_verdicts
scores = safe_df(
    select(
        t.c.as_of,
        t.c.symbol,
        v.c.conviction.label("raw_conviction"),
        t.c.effective_probability,
        t.c.model_version,
    )
    .join_from(t, v, t.c.verdict_id == v.c.id, isouter=True)
    .where(t.c.source == "live")
    .order_by(t.c.as_of.desc(), t.c.symbol)
)
st.subheader("Latest scores vs raw conviction")
if not empty_note(
    scores,
    "No meta scores yet — the nightly meta_scoring asset writes rows only "
    "once a meta-model reaches PRODUCTION in the registry.",
):
    latest_day = scores["as_of"].max()
    latest = scores[scores["as_of"] == latest_day].copy()
    latest["delta"] = latest["effective_probability"] - latest["raw_conviction"]
    st.caption(f"Scores for {latest_day}")
    st.dataframe(latest, width="stretch")

# --- realized outcomes + Brier comparison ----------------------------------

st.subheader("Meta vs raw conviction against realized outcomes")
exp = schema.decision_experience
resolved = safe_df(
    select(
        t.c.as_of,
        t.c.symbol,
        v.c.conviction.label("raw_conviction"),
        t.c.effective_probability,
        t.c.model_version,
        exp.c.pnl_pct,
        exp.c.label,
    )
    .join_from(t, v, t.c.verdict_id == v.c.id, isouter=True)
    .join_from(
        t,
        exp,
        (t.c.symbol == exp.c.symbol) & (t.c.as_of == exp.c.as_of),
    )
    .where(
        t.c.source == "live",
        exp.c.source == "live",
        exp.c.resolved.is_(True),
        exp.c.label.in_(["win", "loss"]),
    )
    .order_by(t.c.as_of.desc(), t.c.symbol)
)
if not empty_note(
    resolved,
    "No scored decisions have resolved into experiences yet — this table "
    "fills as trades close and the experience_resolution asset joins them.",
):
    frame = resolved.dropna(subset=["effective_probability"]).copy()
    frame["outcome"] = (frame["label"] == "win").astype(float)
    st.dataframe(frame, width="stretch")
    scored = frame.dropna(subset=["raw_conviction"])
    if len(scored):
        brier_meta = float(((scored["effective_probability"] - scored["outcome"]) ** 2).mean())
        brier_raw = float(((scored["raw_conviction"] - scored["outcome"]) ** 2).mean())
        c1, c2, c3 = st.columns(3)
        c1.metric("Meta Brier (lower is better)", f"{brier_meta:.4f}")
        c2.metric("Raw-conviction Brier", f"{brier_raw:.4f}")
        c3.metric("Resolved scored decisions", f"{len(scored)}")
        if len(scored) < 30:
            st.caption(
                f"Small sample ({len(scored)} decisions) — Brier comparison is "
                "not yet meaningful; sample sizes are shown deliberately."
            )
        trailing = scored.copy()
        trailing["as_of"] = pd.to_datetime(trailing["as_of"])
        monthly = (
            trailing.set_index("as_of")
            .sort_index()
            .resample("ME")[["effective_probability", "raw_conviction", "outcome"]]
            .apply(
                lambda g: pd.Series(
                    {
                        "meta_brier": ((g["effective_probability"] - g["outcome"]) ** 2).mean(),
                        "raw_brier": ((g["raw_conviction"] - g["outcome"]) ** 2).mean(),
                        "n": float(len(g)),
                    }
                )
            )
        )
        if len(monthly):
            st.line_chart(monthly[["meta_brier", "raw_brier"]])
