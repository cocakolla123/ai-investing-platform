"""Analyze the pre-registered bankroll-profile comparison.

docs/preregistration_bankroll_profile.md -> evals/bankroll/results.json.
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

BANKROLLS = (1_000.0, 10_000.0, 100_000.0)
OUT = Path("evals/bankroll")


def _summary(engine: Any, repo: ReplayRepository, artifact_dir: Path, rid: str) -> dict[str, Any]:
    run = repo.get_run(rid)
    assert run is not None
    rep = replay_report(rid, repo, artifact_dir, benchmark_closes=_benchmark_closes(engine, run))
    m, b = rep.metrics, rep.benchmark
    with engine.connect() as c:
        held = c.execute(
            text(
                "select avg(n) from (select as_of, count(*) n from replay_decisions "
                "where run_id = :r and target_weight > 0 group by as_of) t"
            ),
            {"r": rid},
        ).scalar()
    fills = repo.fills_for(rid)
    return {
        "run_id": rid,
        "total_return": m.total_return,
        "sharpe": m.sharpe,
        "max_drawdown": m.max_drawdown,
        "fills": len(fills),
        "symbols_traded": len({f["symbol"] for f in fills}),
        "costs_usd": fills_total_costs(fills),
        "avg_positions": float(held) if held is not None else math.nan,
        "excess_vs_spy": None if b is None else b.excess_total_return,
        "beta": None if b is None else b.beta,
        "alpha_annual": None if b is None else b.alpha_annual,
        "alpha_t": None if b is None else b.alpha_t_stat,
    }


def main(code_version: str) -> None:
    engine = make_engine(Settings().database_url)
    repo = ReplayRepository(engine)
    artifact_dir = Path(Settings().artifact_dir)
    with engine.connect() as c:
        rows = c.execute(
            text("select id, config_json from replay_runs where config_json->>'code_version' = :v"),
            {"v": code_version},
        ).all()
    runs: dict[tuple[float, str], str] = {}
    for rid, cfg in rows:
        arm = "profile" if cfg.get("bankroll_profile") else "flat"
        runs[(float(cfg["initial_equity"]), arm)] = rid
    result: dict[str, Any] = {
        "preregistration": "docs/preregistration_bankroll_profile.md",
        "code_version": code_version,
        "bankrolls": {},
    }
    for eq in BANKROLLS:
        flat, prof = runs[(eq, "flat")], runs[(eq, "profile")]
        with engine.connect() as c:
            curve = pd.read_sql(
                text("select run_id, as_of, equity from replay_equity where run_id = any(:r)"),
                c,
                params={"r": [flat, prof]},
            )
        wide = curve.pivot_table(index="as_of", columns="run_id", values="equity").sort_index()
        rets = wide.pct_change().dropna()
        diff = (rets[prof] - rets[flat]).to_numpy(float)
        sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
        t = float(diff.mean() / sd * math.sqrt(diff.size)) if sd > 0 else math.nan
        p_sum = _summary(engine, repo, artifact_dir, prof)
        ok_dd = p_sum["max_drawdown"] >= -0.15
        verdict = (
            "profile better"
            if t >= 2.0 and ok_dd
            else ("profile worse" if t <= -2.0 or not ok_dd else "no difference shown")
        )
        result["bankrolls"][f"{int(eq)}"] = {
            "flat": _summary(engine, repo, artifact_dir, flat),
            "profile": p_sum,
            "paired_daily_t": t,
            "annualized_diff": float(diff.mean() * 252) if diff.size else math.nan,
            "verdict": verdict,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
