"""Learning-vs-control page (plan Phase 6): did learning earn its keep?

Generational training results (meta-model vs the frozen control — raw
conviction under identity calibration — on IDENTICAL test decisions), the
value-of-learning trend across generations, and replay-run summaries. Sample
sizes are shown prominently everywhere: on a personal platform the honest
answer is often "not enough data yet", and this page says so rather than
dressing thin samples up as signal.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import func, select
from streamlit_helpers import empty_note, safe_df

from dynamo.persistence import schema

st.set_page_config(page_title="Dynamo — Learning vs Control", layout="wide")
st.title("Learning vs frozen control")

st.info(
    "**value_of_learning = brier_control - brier_meta.** POSITIVE means the "
    "learned meta-model beats the frozen control (raw committee conviction, "
    "identity calibration) on the identical test decisions; negative means "
    "learning has not yet earned trust. Winners still go through the full "
    "promotion pipeline with human approval — nothing here changes sizing."
)

# --- generation results ------------------------------------------------------

gr = schema.generation_results
generations = safe_df(
    select(
        gr.c.generation,
        gr.c.train_start,
        gr.c.train_end,
        gr.c.test_start,
        gr.c.test_end,
        gr.c.model_version,
        gr.c.n_train,
        gr.c.n_test,
        gr.c.brier_meta,
        gr.c.brier_control,
        gr.c.ev_meta,
        gr.c.ev_control,
        gr.c.value_of_learning,
        gr.c.created_at,
    ).order_by(gr.c.created_at.desc(), gr.c.generation)
)
st.subheader("Generation results")
if not empty_note(
    generations,
    "No generation results yet — run dynamo.cli.generations once enough "
    "monthly experience history exists (generations below the 150-experience "
    "training gate are recorded as skipped, never trained).",
):
    latest_batch = generations[generations["created_at"] == generations["created_at"].max()]
    st.dataframe(
        latest_batch.drop(columns=["created_at"]).sort_values("generation"),
        width="stretch",
        hide_index=True,
    )
    trained = latest_batch.dropna(subset=["value_of_learning"]).sort_values("generation")
    n_total = int(latest_batch["n_test"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Generations (latest batch)", f"{len(latest_batch)}")
    c2.metric("Generations trained", f"{len(trained)}")
    c3.metric("Total test decisions", f"{n_total}")
    if n_total < 100:
        st.caption(
            f"Small sample — {n_total} test decisions across all generations. "
            "Value-of-learning numbers are not yet meaningful; sample sizes "
            "are shown deliberately."
        )
    st.subheader("Value-of-learning trend")
    if len(trained):
        st.line_chart(trained.set_index("generation")[["value_of_learning"]])
        st.line_chart(trained.set_index("generation")[["brier_meta", "brier_control"]])
        st.caption("Per-generation test sample sizes (n_test):")
        st.bar_chart(trained.set_index("generation")[["n_test"]])
    else:
        st.info("No generation trained yet — every fold was below the training gate.")

# --- replay runs -------------------------------------------------------------

st.subheader("Replay runs")
rr = schema.replay_runs
replay_runs = safe_df(
    select(
        rr.c.id,
        rr.c.mode,
        rr.c.status,
        rr.c.start_as_of,
        rr.c.end_as_of,
        rr.c.label,
        rr.c.budget_usd,
        rr.c.spent_usd,
        rr.c.created_at,
    ).order_by(rr.c.created_at.desc())
)
if not empty_note(
    replay_runs,
    "No replay runs yet — dynamo.cli.replay is manual only (committee mode "
    "requires --max-usd).",
):
    exp = schema.decision_experience
    per_run = safe_df(
        select(exp.c.replay_run_id, func.count().label("experiences"))
        .where(exp.c.source == "replay")
        .group_by(exp.c.replay_run_id)
    )
    if not per_run.empty:
        replay_runs = replay_runs.merge(
            per_run, left_on="id", right_on="replay_run_id", how="left"
        ).drop(columns=["replay_run_id"])
    st.caption(
        "Replay experiences are firewalled (source='replay') and enter "
        "training ONLY via dynamo.cli.generations --include-replay <run_id> — "
        "they never touch live calibration."
    )
    st.dataframe(replay_runs, width="stretch", hide_index=True)

# --- replay equity curves (backtest-campaign Phase 5) -----------------------
# Metrics are computed inline from the persisted curve (pandas/numpy only) —
# deliberately NOT via dynamo.replay.report, which drags the backtest stack
# into a page that only needs light math.

st.subheader("Replay equity curves")
req = schema.replay_equity
available = safe_df(select(req.c.run_id).distinct().order_by(req.c.run_id))
if not empty_note(
    available,
    "No persisted replay equity curves yet — quant/portfolio replays "
    "(dynamo.cli.replay run) persist one row per session into replay_equity.",
):
    run_ids = list(available["run_id"])
    selected = st.multiselect(
        "Runs to overlay (curves are normalized to 100 at start so different "
        "bankrolls compare directly)",
        run_ids,
        default=run_ids[:3],
    )
    if selected:
        curves = safe_df(
            select(req.c.run_id, req.c.as_of, req.c.equity)
            .where(req.c.run_id.in_(selected))
            .order_by(req.c.as_of)
        )
        if not curves.empty:
            wide = curves.pivot(index="as_of", columns="run_id", values="equity")
            wide = wide.sort_index()
            normalized = wide.apply(lambda s: 100.0 * s / s.dropna().iloc[0])
            st.line_chart(normalized)

            metric_rows = []
            for run_id in selected:
                if run_id not in wide.columns:
                    continue
                s = wide[run_id].dropna()
                if len(s) < 2:
                    continue
                r = s.pct_change().dropna()
                std = float(r.std(ddof=1))
                sharpe = float(r.mean()) / std * (252.0**0.5) if std > 0 else 0.0
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "sessions": len(s),
                        "start_equity": float(s.iloc[0]),
                        "final_equity": float(s.iloc[-1]),
                        "return_pct": (float(s.iloc[-1] / s.iloc[0]) - 1.0) * 100.0,
                        "sharpe": sharpe,
                        "max_drawdown_pct": float((s / s.cummax() - 1.0).min()) * 100.0,
                        "vol_annual_pct": std * (252.0**0.5) * 100.0,
                    }
                )
            if metric_rows:
                st.dataframe(
                    pd.DataFrame(metric_rows), width="stretch", hide_index=True
                )
            st.caption(
                "Light per-curve metrics computed from replay_equity; the full "
                "report (deflated Sharpe, benchmark, per-regime blocks) comes "
                "from `dynamo.cli.replay report --run-id <id>`."
            )
