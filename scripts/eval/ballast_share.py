"""How much of the committee campaign's book was SPY ballast, precisely.

For a portfolio replay run (default: the committee campaign at $100k on the
fixed code, replay-1bccf6b04259), over its sessions:

- mean daily SPY **target weight** (the constructor's target, share of equity);
- mean daily SPY **market value / equity** (SPY shares from fills, at each
  session's canonical close; the first session has no fills yet);
- the committee's own positions: mean and median number of non-SPY names with a
  target weight above zero per session.

Writes evals/rerun/ballast_share.json. Read-only.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import canonical_bars

OUT = Path("evals/rerun")


def main(run_id: str) -> None:
    engine = make_engine(Settings().database_url)
    with engine.connect() as c:
        eq = pd.read_sql(
            text("select as_of, equity from replay_equity where run_id = :r order by as_of"),
            c,
            params={"r": run_id},
        )
        spy_w = pd.read_sql(
            text(
                "select as_of, target_weight from replay_decisions "
                "where run_id = :r and symbol = 'SPY'"
            ),
            c,
            params={"r": run_id},
        )
        others = pd.read_sql(
            text(
                "select as_of, count(*) n from replay_decisions where run_id = :r "
                "and symbol <> 'SPY' and target_weight > 0 group by as_of"
            ),
            c,
            params={"r": run_id},
        )
        fills = pd.read_sql(
            text(
                "select session, side, qty from replay_fills where run_id = :r and symbol = 'SPY'"
            ),
            c,
            params={"r": run_id},
        )
        raw = pd.read_sql(
            text("select * from bars_1d where symbol = 'SPY' and ts >= :a and ts <= :b"),
            c,
            params={
                "a": eq["as_of"].min() - pd.Timedelta(days=7),
                "b": eq["as_of"].max() + pd.Timedelta(days=1),
            },
        )
    bars = canonical_bars(raw)
    bars["d"] = pd.to_datetime(bars["ts"], utc=True).dt.date
    close = bars.groupby("d")["close"].last()
    signed = fills.assign(q=fills["qty"].where(fills["side"] == "buy", -fills["qty"]))
    mv_share = []
    for as_of, equity in zip(eq["as_of"], eq["equity"], strict=True):
        shares = float(signed.loc[signed["session"] <= as_of, "q"].sum())
        mv_share.append(shares * float(close.loc[as_of]) / float(equity) if equity else 0.0)
    counts = others.set_index("as_of")["n"].reindex(eq["as_of"], fill_value=0).tolist()
    result = {
        "run_id": run_id,
        "sessions": len(eq),
        "window": [str(eq["as_of"].min()), str(eq["as_of"].max())],
        "spy_target_weight_mean": round(float(spy_w["target_weight"].mean()), 4),
        "spy_target_weight_median": round(float(spy_w["target_weight"].median()), 4),
        "spy_market_value_share_of_equity_mean": round(statistics.fmean(mv_share), 4),
        "spy_market_value_share_mean_excluding_first_session": round(
            statistics.fmean(mv_share[1:]), 4
        ),
        "committee_positions_per_session_mean": round(statistics.fmean(counts), 2),
        "committee_positions_per_session_median": statistics.median(counts),
        "committee_positions_per_session_max": max(counts),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ballast_share.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "replay-1bccf6b04259")
