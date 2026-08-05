"""Same-day tactical entries + PDT guard (ADR-0010): daily cap, trigger rules,
determinism, and every PDT scenario."""

from __future__ import annotations

from datetime import date

from dynamo.domain.models.research import Action, Signal, Sleeve
from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.intraday import (
    PDT_EQUITY_FLOOR,
    PDT_MAX_DAY_TRADES,
    SKIP_ALREADY_ENTERED_TODAY,
    SKIP_CAP_REACHED,
    SKIP_NO_QUOTE,
    SKIP_NO_SIZING_INPUTS,
    SKIP_NO_VERDICT,
    SKIP_NOT_TRIGGERED,
    TRIGGER_BREAKOUT,
    TRIGGER_PROMOTION,
    IntradayEntryParams,
    count_day_trades,
    pdt_exit_allowed,
    select_intraday_entries,
)
from dynamo.risk.sizing import SizingCandidate, SizingContext
from tests.unit.options.helpers import AS_OF, make_verdict

CONFIG = RiskConfig()


def candidate(symbol: str, conviction: float = 0.8) -> SizingCandidate:
    return SizingCandidate(
        symbol=symbol,
        sleeve=Sleeve.TACTICAL,
        conviction=conviction,
        sigma_annual=0.25,
        adv_dollar_20d=50_000_000.0,
        sector="tech",
    )


def breakout(symbol: str, entry_ref: float | None = 100.0, strength: float = 0.6) -> Signal:
    return Signal(
        strategy_id="breakout-v1",
        symbol=symbol,
        as_of=AS_OF,
        entry_ref_price=entry_ref,
        strength=strength,
    )


def run(
    *,
    promoted=(),
    signals=(),
    quotes=None,
    verdicts=None,
    candidates=None,
    entries_today=0,
    entered_today=frozenset(),
    cap=3,
    equity=100_000.0,
):
    symbols = set(promoted) | {s.symbol for s in signals}
    quotes = quotes if quotes is not None else dict.fromkeys(symbols, 101.0)
    verdicts = (
        verdicts
        if verdicts is not None
        else {s: make_verdict(s, action=Action.BUY) for s in promoted}
    )
    candidates = (
        candidates if candidates is not None else {s: candidate(s) for s in symbols}
    )
    return select_intraday_entries(
        promoted_material=list(promoted),
        breakout_signals=list(signals),
        live_quotes=quotes,
        verdicts=verdicts,
        sizing_candidates=candidates,
        config=CONFIG,
        context=SizingContext(equity=equity),
        entries_today=entries_today,
        entered_today=entered_today,
        params=IntradayEntryParams(cap_per_day=cap),
    )


class TestTriggers:
    def test_promotion_needs_a_buy_or_add_verdict(self) -> None:
        result = run(
            promoted=["NEW1", "NEW2"],
            verdicts={
                "NEW1": make_verdict("NEW1", action=Action.BUY),
                "NEW2": make_verdict("NEW2", action=Action.HOLD),
            },
        )
        assert [e.symbol for e in result.entries] == ["NEW1"]
        assert result.entries[0].trigger == TRIGGER_PROMOTION
        assert ("NEW2", SKIP_NO_VERDICT) in result.skipped

    def test_breakout_requires_quote_at_or_above_entry_ref(self) -> None:
        result = run(
            signals=[breakout("BRK", entry_ref=105.0)],
            quotes={"BRK": 101.0},  # below the breakout level — not triggered
        )
        assert result.entries == ()
        assert ("BRK", SKIP_NOT_TRIGGERED) in result.skipped

        triggered = run(signals=[breakout("BRK", entry_ref=100.5)], quotes={"BRK": 101.0})
        assert [e.symbol for e in triggered.entries] == ["BRK"]
        assert triggered.entries[0].trigger == TRIGGER_BREAKOUT

    def test_missing_quote_fails_closed(self) -> None:
        result = run(signals=[breakout("BRK")], quotes={})
        assert result.entries == ()
        assert ("BRK", SKIP_NO_QUOTE) in result.skipped

    def test_missing_sizing_inputs_fail_closed(self) -> None:
        result = run(promoted=["NEW1"], candidates={})
        assert result.entries == ()
        assert ("NEW1", SKIP_NO_SIZING_INPUTS) in result.skipped

    def test_entries_run_the_full_cascade(self) -> None:
        result = run(promoted=["NEW1"])
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert 0.0 < entry.weight <= CONFIG.tactical_position_cap
        assert "cascade" in entry.rationale


class TestDailyCap:
    def test_cap_binds(self) -> None:
        promoted = [f"N{i}" for i in range(5)]
        result = run(promoted=promoted, cap=3)
        assert len(result.entries) == 3
        assert sum(1 for _, r in result.skipped if r == SKIP_CAP_REACHED) == 2

    def test_earlier_entries_consume_the_cap(self) -> None:
        result = run(promoted=["N1", "N2"], entries_today=2, cap=3)
        assert len(result.entries) == 1
        assert ("N2", SKIP_CAP_REACHED) in result.skipped or (
            "N1",
            SKIP_CAP_REACHED,
        ) in result.skipped

    def test_cap_already_exhausted(self) -> None:
        result = run(promoted=["N1"], entries_today=3, cap=3)
        assert result.entries == ()

    def test_already_entered_today_never_re_enters(self) -> None:
        result = run(promoted=["N1"], entered_today=frozenset({"N1"}))
        assert result.entries == ()
        assert ("N1", SKIP_ALREADY_ENTERED_TODAY) in result.skipped

    def test_deterministic_ranking_conviction_then_symbol(self) -> None:
        verdicts = {
            "AAA": make_verdict("AAA", action=Action.BUY, conviction=0.7),
            "BBB": make_verdict("BBB", action=Action.BUY, conviction=0.9),
            "CCC": make_verdict("CCC", action=Action.BUY, conviction=0.9),
        }
        result = run(promoted=["CCC", "AAA", "BBB"], verdicts=verdicts, cap=2)
        assert [e.symbol for e in result.entries] == ["BBB", "CCC"]  # tie → symbol asc
        again = run(promoted=["AAA", "BBB", "CCC"], verdicts=verdicts, cap=2)
        assert [e.symbol for e in again.entries] == [e.symbol for e in result.entries]


class TestDayTradeCounter:
    def test_buy_and_sell_same_day_is_one_day_trade(self) -> None:
        d = date(2026, 8, 4)
        assert count_day_trades([("AAPL", "buy", d), ("AAPL", "sell", d)]) == 1

    def test_different_days_do_not_count(self) -> None:
        assert (
            count_day_trades(
                [("AAPL", "buy", date(2026, 8, 3)), ("AAPL", "sell", date(2026, 8, 4))]
            )
            == 0
        )

    def test_min_of_buys_and_sells_per_symbol_day(self) -> None:
        d = date(2026, 8, 4)
        fills = [
            ("AAPL", "buy", d),
            ("AAPL", "buy", d),
            ("AAPL", "sell", d),
            ("MSFT", "buy", d),
            ("MSFT", "sell", d),
            ("MSFT", "sell", d),
        ]
        assert count_day_trades(fills) == 2


class TestPdtGuard:
    def test_not_entered_today_is_not_a_day_trade(self) -> None:
        decision = pdt_exit_allowed(
            equity=10_000.0,
            day_trades_in_window=PDT_MAX_DAY_TRADES,
            entered_today=False,
            is_disaster_stop=False,
        )
        assert decision.allowed and decision.reason == "not_a_day_trade"

    def test_disaster_stop_always_runs(self) -> None:
        decision = pdt_exit_allowed(
            equity=10_000.0,
            day_trades_in_window=99,
            entered_today=True,
            is_disaster_stop=True,
        )
        assert decision.allowed and decision.reason == "disaster_stop_overrides_pdt"

    def test_above_the_equity_floor_pdt_does_not_apply(self) -> None:
        decision = pdt_exit_allowed(
            equity=PDT_EQUITY_FLOOR,
            day_trades_in_window=99,
            entered_today=True,
            is_disaster_stop=False,
        )
        assert decision.allowed and decision.reason == "equity_above_pdt_floor"

    def test_would_trip_pdt_is_blocked(self) -> None:
        decision = pdt_exit_allowed(
            equity=10_000.0,
            day_trades_in_window=PDT_MAX_DAY_TRADES,
            entered_today=True,
            is_disaster_stop=False,
        )
        assert not decision.allowed and decision.reason == "would_trip_pdt"

    def test_within_the_allowance_is_permitted(self) -> None:
        decision = pdt_exit_allowed(
            equity=10_000.0,
            day_trades_in_window=PDT_MAX_DAY_TRADES - 1,
            entered_today=True,
            is_disaster_stop=False,
        )
        assert decision.allowed and decision.reason == "day_trade_within_pdt_allowance"

    def test_invalid_equity_fails_closed(self) -> None:
        decision = pdt_exit_allowed(
            equity=float("nan"),
            day_trades_in_window=0,
            entered_today=True,
            is_disaster_stop=False,
        )
        assert not decision.allowed
