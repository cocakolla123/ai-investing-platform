"""Look-ahead canary and constraint audit (doc 08 sections 6, 9).

The canary is a deliberately look-ahead-biased strategy: it peeks at the
NEXT session's open->close move (information the feature path cannot carry)
and only enters ahead of up-moves. Its edge does not flow through the
point-in-time visible window, so re-running with features shifted +1 session
leaves its performance intact — exactly the signature the audit must flag.
If this canary ever passes the audit, CI fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

from dynamo.backtest.audits import (
    constraint_violation_audit,
    look_ahead_audit,
    run_look_ahead_audit,
)
from dynamo.backtest.engine import BacktestEngine, SessionAdvice
from dynamo.backtest.parity import (
    parity_backtest_config,
    parity_construction_config,
    parity_risk_config,
)
from tests.unit.backtest.builders import (
    full_universe,
    sell_verdict,
    sessions,
    signal,
    toy_data,
)

SYM = "AAA"


class LeakyProvider:
    """Trades on tomorrow's open->close move, peeked from the FULL frames."""

    def __init__(self, opens: pd.Series, closes: pd.Series) -> None:
        self.opens = opens  # captured full history: the leak
        self.closes = closes

    def advise(
        self, as_of: date, visible_closes: pd.DataFrame, holdings: Mapping[str, float]
    ) -> SessionAdvice:
        del visible_closes  # the leak ignores the point-in-time window
        idx: list[date] = [d for d in self.opens.index if isinstance(d, date)]
        i = idx.index(as_of)
        if i + 1 < len(idx):
            nxt = idx[i + 1]
            if float(self.closes[nxt]) > float(self.opens[nxt]):
                return SessionAdvice(signals=(signal(SYM, as_of),))
        if holdings.get(SYM, 0.0) > 0.0:
            return SessionAdvice(verdicts=(sell_verdict(SYM, as_of),))
        return SessionAdvice()


def _random_walk(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, n))
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    return list(opens), list(closes)


def test_canary_is_flagged_by_the_audit():
    n = 120
    dates = sessions(n)
    opens, closes = _random_walk(n)
    data = toy_data({SYM: opens}, {SYM: closes}, dates, full_universe([SYM]))
    engine = BacktestEngine(
        parity_backtest_config(cost_bps=0.0, initial_equity=100_000.0),
        construction_config=parity_construction_config(),
        risk_config=parity_risk_config(),
    )
    provider = LeakyProvider(
        pd.Series(opens, index=pd.Index(dates)),
        pd.Series(closes, index=pd.Index(dates)),
    )
    baseline = engine.run(data, provider)
    audit = run_look_ahead_audit(engine, data, provider, baseline=baseline)

    # Perfect foresight: strongly positive baseline Sharpe...
    assert audit.baseline_sharpe > 1.0
    # ...that SURVIVES the +1-session feature shift (the leak bypasses the
    # visible window), so the audit must fail the run. CI fails if it passes.
    assert audit.sharpe_drop < audit.min_drop
    assert audit.passed is False


def test_audit_thresholds():
    assert look_ahead_audit(2.0, 0.5).passed  # dropped 75%: genuine timely edge
    assert not look_ahead_audit(2.0, 1.5).passed  # dropped 25%: leakage
    assert look_ahead_audit(-0.2, -0.2).passed  # no positive edge to defend


def test_constraint_violation_audit_zero_on_toy_run():
    n = 30
    dates = sessions(n)
    opens, closes = _random_walk(n, seed=11)
    data = toy_data({SYM: opens}, {SYM: closes}, dates, full_universe([SYM]))
    engine = BacktestEngine(
        parity_backtest_config(cost_bps=5.0, initial_equity=100_000.0),
        construction_config=parity_construction_config(),
        risk_config=parity_risk_config(),
    )
    provider = LeakyProvider(
        pd.Series(opens, index=pd.Index(dates)),
        pd.Series(closes, index=pd.Index(dates)),
    )
    result = engine.run(data, provider)
    passed, count = constraint_violation_audit(result)
    assert passed and count == 0
