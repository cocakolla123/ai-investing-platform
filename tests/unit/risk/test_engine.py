"""RiskEngine: fail-closed behavior, happy path, veto contract end-to-end."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
from risk_helpers import make_bars, make_panel, make_regime_history

from dynamo.domain.models.portfolio import Lot, PortfolioState, Position
from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import LadderRung, Regime, RiskConfig
from dynamo.risk.engine import RiskEngine, RiskEngineInputs
from dynamo.risk.ladder import LadderState
from dynamo.risk.sizing import SizingCandidate, SizingContext, size_positions
from dynamo.risk.types import ReturnsPanel

AS_OF = datetime(2024, 8, 1, 21, 0, tzinfo=UTC)
SYMS = ("AAA", "BBB", "SPY", "TLT")


def _portfolio(equity: float = 100_000.0) -> PortfolioState:
    lot = Lot(
        symbol="AAA",
        sleeve=Sleeve.TACTICAL,
        qty=100.0,
        entry_price=50.0,
        entry_date=date(2024, 7, 1),
    )
    return PortfolioState(
        as_of=AS_OF,
        equity=equity,
        cash=equity - 100.0 * 55.0,
        positions=(Position(symbol="AAA", qty=100.0, avg_entry_price=50.0, lots=(lot,)),),
        high_water_mark=equity * 1.02,
    )


def _inputs(**overrides: object) -> RiskEngineInputs:
    base: dict[str, object] = {
        "portfolio": _portfolio(),
        "returns": make_panel(1, SYMS, 550),
        "prices": {"AAA": 55.0},
        "sector_by_symbol": {"AAA": "Tech", "BBB": "Tech"},
        "adv_dollar_20d": {"AAA": 5e7, "BBB": 1e3, "SPY": 1e10, "TLT": 1e9},
        "regime_history": make_regime_history(sessions=320),
        "prev_ladder": LadderState(),
        "config": RiskConfig(),
        "as_of": AS_OF,
        "session_index": 100,
        "bars": {"AAA": make_bars(2)},
    }
    base.update(overrides)
    return RiskEngineInputs(**base)  # type: ignore[arg-type]


ENGINE = RiskEngine()


class TestFailClosed:
    def test_empty_returns_history(self) -> None:
        empty = ReturnsPanel((), SYMS, np.zeros((0, len(SYMS))))
        verdict = ENGINE.compute_verdict(_inputs(returns=empty))
        assert verdict.degraded
        assert not verdict.new_risk_allowed
        assert any(v.rule == "fail_closed" for v in verdict.vetoes)
        assert verdict.per_position_cap == {}

    def test_stale_returns_history(self) -> None:
        stale = make_panel(1, SYMS, 550, end=date(2024, 6, 1))
        verdict = ENGINE.compute_verdict(_inputs(returns=stale))
        assert verdict.degraded and not verdict.new_risk_allowed
        assert any("stale" in v.reason for v in verdict.vetoes)

    def test_missing_mark_price_for_held_position(self) -> None:
        verdict = ENGINE.compute_verdict(_inputs(prices={}))
        assert verdict.degraded and not verdict.new_risk_allowed
        assert any("mark price" in v.reason for v in verdict.vetoes)

    def test_held_symbol_missing_from_returns(self) -> None:
        no_aaa = make_panel(1, ("BBB", "SPY"), 550)
        verdict = ENGINE.compute_verdict(_inputs(returns=no_aaa))
        assert verdict.degraded
        assert any("AAA" in v.reason for v in verdict.vetoes)

    def test_short_regime_history(self) -> None:
        verdict = ENGINE.compute_verdict(_inputs(regime_history=make_regime_history(sessions=100)))
        assert verdict.degraded and not verdict.new_risk_allowed
        assert verdict.regime.regime is Regime.RISK_OFF  # conservative fallback

    def test_invalid_equity(self) -> None:
        verdict = ENGINE.compute_verdict(_inputs(portfolio=_portfolio(equity=float("nan"))))
        assert verdict.degraded and not verdict.new_risk_allowed

    def test_degraded_still_reports_ladder_and_cash_floor(self) -> None:
        empty = ReturnsPanel((), SYMS, np.zeros((0, len(SYMS))))
        verdict = ENGINE.compute_verdict(
            _inputs(returns=empty, prev_ladder=LadderState(rung=LadderRung.DD15))
        )
        assert verdict.ladder is LadderRung.DD15
        assert verdict.gross_exposure_cap <= 0.5  # DD15 defensive override holds
        assert verdict.cash_floor >= 0.5


class TestHappyPath:
    def test_full_verdict_shape(self) -> None:
        verdict = ENGINE.compute_verdict(_inputs())
        cfg = RiskConfig()
        assert not verdict.degraded
        assert verdict.new_risk_allowed
        assert verdict.ladder is LadderRung.NORMAL
        assert set(verdict.per_position_cap) == set(SYMS)
        for cap in verdict.per_position_cap.values():
            assert 0.0 <= cap <= cfg.per_position_hard
        assert set(verdict.clusters) == set(SYMS)
        assert verdict.gross_exposure_cap <= cfg.gross_cap
        assert verdict.cash_floor >= cfg.cash_floor
        assert verdict.net_exposure_cap == verdict.gross_exposure_cap
        assert verdict.metrics.var95_1d is not None
        assert verdict.metrics.var99_1d is not None
        assert verdict.metrics.var99_1d >= verdict.metrics.var95_1d
        assert verdict.metrics.portfolio_vol_annual is not None
        assert verdict.metrics.net_beta is not None
        assert verdict.metrics.gross_exposure > 0.0

    def test_stops_emitted_for_held_positions(self) -> None:
        verdict = ENGINE.compute_verdict(_inputs())
        assert any(s.symbol == "AAA" for s in verdict.stop_updates)

    def test_illiquid_symbol_vetoed(self) -> None:
        verdict = ENGINE.compute_verdict(_inputs())
        liquidity_vetoes = {v.symbol for v in verdict.vetoes if v.rule == "liquidity"}
        assert "BBB" in liquidity_vetoes  # $1k ADV cannot support any position
        assert "SPY" not in liquidity_vetoes

    def test_ladder_drawdown_gates_new_risk(self) -> None:
        deep = _portfolio()
        deep = deep.model_copy(update={"high_water_mark": deep.equity / 0.88})
        verdict = ENGINE.compute_verdict(_inputs(portfolio=deep))
        assert verdict.ladder is LadderRung.DD10
        assert not verdict.new_risk_allowed


class TestVetoContract:
    def test_quarantined_symbol_capped_to_zero_and_sized_to_zero(self) -> None:
        """Committee buy at conviction 0.9 on a vetoed symbol -> weight 0."""
        verdict = ENGINE.compute_verdict(_inputs(quarantined=frozenset({"BBB"})))
        assert any(v.symbol == "BBB" and v.rule == "quarantine" for v in verdict.vetoes)
        assert verdict.per_position_cap["BBB"] == 0.0

        vetoed = frozenset(v.symbol for v in verdict.vetoes if v.symbol != "*")
        candidate = SizingCandidate(
            symbol="BBB",
            sleeve=Sleeve.LONG_TERM,
            conviction=0.9,  # maximum-conviction committee buy
            sigma_annual=0.20,
            adv_dollar_20d=1e9,
        )
        ctx = SizingContext(equity=100_000.0, vetoed=vetoed, sleeve_budget=0.10)
        (sized,) = size_positions([candidate], RiskConfig(), ctx)
        assert sized.weight == 0.0
        assert sized.clamps_applied == ("veto",)


class TestRegimeAndKupiecEffects:
    def test_kupiec_red_tightens_caps_by_quarter(self) -> None:
        cfg = RiskConfig()
        green = ENGINE.compute_verdict(_inputs(var_exceptions_250d=3))
        red = ENGINE.compute_verdict(_inputs(var_exceptions_250d=40))
        assert green.sector_caps["Tech"] == cfg.sector_cap
        assert red.sector_caps["Tech"] == cfg.sector_cap * 0.75
        for symbol in SYMS:
            assert red.per_position_cap[symbol] <= green.per_position_cap[symbol] + 1e-12

    def test_crisis_regime_blocks_new_risk(self) -> None:
        crash = make_regime_history(
            sessions=320, drift=-0.02, vol=0.03, oas=9.0, slope=-0.6, breadth=0.05
        )
        verdict = ENGINE.compute_verdict(_inputs(regime_history=crash))
        assert verdict.regime.regime in (Regime.RISK_OFF, Regime.CRISIS)
        if verdict.regime.regime is Regime.CRISIS:
            assert not verdict.new_risk_allowed
        else:
            assert verdict.gross_exposure_cap <= RiskConfig().gross_cap * 0.8


def test_verdict_is_deterministic() -> None:
    first = ENGINE.compute_verdict(_inputs())
    second = ENGINE.compute_verdict(_inputs())
    assert first == second
