"""Analyze the pre-registered trend-rule replays (docs/preregistration_trend_rule.md).

Validity checks first, then per-run report metrics (the same
``replay_report`` the CLI prints), the head-to-head paired daily t-stat, and
the decision. Writes evals/trend_rule/results.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from dynamo.cli.replay import _benchmark_closes
from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.replay import ReplayRepository
from dynamo.replay.report import replay_report
from dynamo.replay.report import total_costs as fills_total_costs

RUNS = {  # name -> (model, start, end)
    "trend-long": ("trend-rule-v1", "2025-01-02", "2026-08-14"),
    "trend-camp": ("trend-rule-v1", "2026-03-03", "2026-08-14"),
    "v8-camp": ("campaign-v8", "2026-03-03", "2026-08-14"),
}
OUT = Path("evals/trend_rule")


def main(code_version: str) -> None:
    engine = make_engine(Settings().database_url)
    repo = ReplayRepository(engine)
    artifact_dir = Path(Settings().artifact_dir)
    with engine.connect() as c:
        rows = c.execute(
            text("select id, config_json from replay_runs where config_json->>'code_version' = :v"),
            {"v": code_version},
        ).all()
    ids: dict[str, str] = {}
    for rid, cfg in rows:
        for name, (model, start, end) in RUNS.items():
            if (cfg.get("quant_model_version"), cfg.get("start"), cfg.get("end")) == (
                model,
                start,
                end,
            ):
                ids[name] = rid
    assert set(ids) == set(RUNS), f"runs found: {ids}"

    with engine.connect() as c:
        stored_scores = c.execute(
            text("select count(*) from quant_scores where as_of between :s and :e"),
            {"s": "2025-01-02", "e": "2026-08-14"},
        ).scalar()
        ledger_rows = c.execute(
            text("select count(*) from cost_ledger where request_id like any(:p)"),
            {"p": [f"{rid}:%" for rid in ids.values()]},
        ).scalar()
    validity = {
        "run_ids": ids,
        "stored_quant_scores_in_windows": int(stored_scores or 0),
        "model_calls": int(ledger_rows or 0),
    }

    per_run: dict[str, Any] = {}
    for name, rid in ids.items():
        run = repo.get_run(rid)
        assert run is not None
        closes = _benchmark_closes(engine, run)
        rep = replay_report(rid, repo, artifact_dir, benchmark_closes=closes)
        m, b = rep.metrics, rep.benchmark
        costs = fills_total_costs(repo.fills_for(rid))
        per_run[name] = {
            "sessions": m.n_sessions,
            "total_return": m.total_return,
            "sharpe": m.sharpe,
            "deflated_sharpe": rep.deflated_sharpe,
            "max_drawdown": m.max_drawdown,
            "turnover_one_way": rep.behavior.one_way_turnover_total,
            "costs_usd": costs,
            "benchmark": None
            if b is None
            else {
                "excess_total_return": b.excess_total_return,
                "beta": b.beta,
                "alpha_annual": b.alpha_annual,
                "tracking_error_annual": b.tracking_error_annual,
                "information_ratio": b.information_ratio,
                "alpha_t_stat": b.alpha_t_stat,
                "years_to_significance": b.years_to_significance,
            },
        }

    with engine.connect() as c:
        eq = pd.read_sql(
            text("select run_id, as_of, equity from replay_equity where run_id = any(:r)"),
            c,
            params={"r": [ids["trend-camp"], ids["v8-camp"]]},
        )
    wide = eq.pivot_table(index="as_of", columns="run_id", values="equity").sort_index()
    rets = wide.pct_change().dropna()
    diff = (rets[ids["trend-camp"]] - rets[ids["v8-camp"]]).to_numpy(float)
    t = (
        float(diff.mean() / diff.std(ddof=1) * math.sqrt(diff.size))
        if diff.size > 1 and diff.std(ddof=1) > 0
        else math.nan
    )
    head_to_head = {
        "sessions": int(diff.size),
        "mean_daily_diff": float(diff.mean()) if diff.size else math.nan,
        "annualized_diff": float(diff.mean() * 252) if diff.size else math.nan,
        "t": t,
        "trend_beats_v8": bool(t >= 2.0),
    }

    long = per_run["trend-long"]["benchmark"] or {}
    alpha, alpha_t = long.get("alpha_annual", math.nan), long.get("alpha_t_stat", math.nan)
    dd = per_run["trend-long"]["max_drawdown"]
    if alpha > 0 and alpha_t >= 1.96 and dd >= -0.15:
        decision = "GO"
    elif alpha > 0 and dd >= -0.15:
        decision = "INCONCLUSIVE"
    else:
        decision = "NO-GO"

    result = {
        "preregistration": "docs/preregistration_trend_rule.md",
        "code_version": code_version,
        "validity": validity,
        "runs": per_run,
        "head_to_head_trend_minus_v8": head_to_head,
        "decision": decision,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
