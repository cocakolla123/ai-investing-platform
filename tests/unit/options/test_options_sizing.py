"""Options sizing: golden selection cases per structure, liquidity/DTE/IV
floors, hypothesis cap invariants, determinism, and NaN robustness (ADR-0010)."""

from __future__ import annotations

import math
from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.options import (
    CONTRACT_MULTIPLIER,
    OptionRight,
    OptionStructure,
)
from dynamo.domain.models.research import Action, QuantAlignment
from dynamo.risk.options_sizing import (
    SKIP_INSUFFICIENT_CASH,
    SKIP_NO_EXPIRY_IN_WINDOW,
    SKIP_NO_IV_RANK,
    SKIP_NO_LIQUID_STRIKE,
    SKIP_NO_VERDICT_MATCH,
    OptionsSizingParams,
    is_monthly_expiry,
    iv_rank,
    select_and_size_option_trades,
)
from tests.unit.options.helpers import (
    EQUITY,
    MONTHLY_EXPIRY,
    NOW,
    PARAMS,
    default_chain,
    make_chain_row,
    make_inputs,
    make_option_position,
    make_verdict,
)


class TestIvRank:
    def test_needs_min_observations(self) -> None:
        assert iv_rank(0.3, [0.2] * 9) is None
        assert iv_rank(0.3, [0.2] * 10) is not None

    def test_min_max_golden(self) -> None:
        history = [0.2 + 0.01 * i for i in range(21)]  # 0.20 .. 0.40
        assert math.isclose(iv_rank(0.30, history) or 0.0, 0.5)
        assert iv_rank(0.20, history) == 0.0
        assert iv_rank(0.50, history) == 1.0  # clamped

    def test_flat_history_is_uninformative(self) -> None:
        assert iv_rank(0.3, [0.3] * 20) == 0.5

    def test_nan_fails_closed(self) -> None:
        assert iv_rank(float("nan"), [0.3] * 20) is None


class TestMonthlyExpiry:
    def test_third_friday(self) -> None:
        assert is_monthly_expiry(date(2026, 9, 18))
        assert not is_monthly_expiry(date(2026, 9, 11))  # second Friday
        assert not is_monthly_expiry(date(2026, 9, 25))  # fourth Friday
        assert not is_monthly_expiry(date(2026, 9, 17))  # Thursday


class TestGoldenSelection:
    def test_long_call(self) -> None:
        """buy + conviction 0.8 + quant agrees + IV rank 0.4 → long call at the
        ~ITM strike (moneyness heuristic), 4 contracts at the 2.50 ask."""
        result = select_and_size_option_trades(make_inputs())
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.structure is OptionStructure.LONG_CALL
        assert d.right is OptionRight.CALL
        assert d.strike == 95.0  # ~5% ITM for the 0.6-delta target
        assert d.limit_price == 2.50  # buys price at the ask
        # per-position cap: 1% x 100k = $1,000; 2.50 x 100 = $250/contract → 4.
        assert d.contracts == 4
        assert d.premium_at_risk == 1_000.0
        assert d.dte == 45
        assert d.expiry == MONTHLY_EXPIRY

    def test_delta_targeting_when_greeks_present(self) -> None:
        chain = [
            make_chain_row(strike=90.0, delta=0.75),
            make_chain_row(strike=95.0, delta=0.62),
            make_chain_row(strike=100.0, delta=0.50),
            make_chain_row(strike=95.0, right=OptionRight.PUT, delta=-0.4),
        ]
        result = select_and_size_option_trades(make_inputs(chains={"AAPL": chain}))
        assert result.decisions[0].strike == 95.0  # |0.62 - 0.6| is closest

    def test_protective_long_put_on_held_name(self) -> None:
        inputs = make_inputs(
            verdicts={"AAPL": make_verdict(action=Action.SELL, conviction=0.75)},
            shares_held={"AAPL": 300.0},
        )
        result = select_and_size_option_trades(inputs)
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.structure is OptionStructure.LONG_PUT
        assert d.strike == 105.0  # ~5% ITM put

    def test_bearish_put_on_unheld_name_needs_quant_agreement(self) -> None:
        agree = make_inputs(
            verdicts={"AAPL": make_verdict(action=Action.AVOID, conviction=0.9)}
        )
        assert (
            select_and_size_option_trades(agree).decisions[0].structure
            is OptionStructure.LONG_PUT
        )
        conflict = make_inputs(
            verdicts={
                "AAPL": make_verdict(
                    action=Action.AVOID,
                    conviction=0.9,
                    quant_alignment=QuantAlignment.CONFLICT,
                )
            }
        )
        result = select_and_size_option_trades(conflict)
        assert result.decisions == ()
        assert ("AAPL", SKIP_NO_VERDICT_MATCH) in result.skipped

    def test_covered_call_on_long_term_holding_with_elevated_iv(self) -> None:
        inputs = make_inputs(
            verdicts={"AAPL": make_verdict(action=Action.HOLD, conviction=0.5)},
            iv_ranks={"AAPL": 0.6},
            shares_held={"AAPL": 250.0},
            long_term_holdings=frozenset({"AAPL"}),
        )
        result = select_and_size_option_trades(inputs)
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.structure is OptionStructure.COVERED_CALL
        assert d.strike == 105.0  # ~5% OTM short leg
        assert d.limit_price == 2.40  # sells price at the bid
        assert d.contracts == 2  # floor(250 / 100) — never uncovered
        assert d.premium_at_risk == 0.0  # share-covered
        assert d.collateral_shares == 250.0

    def test_covered_call_needs_elevated_iv(self) -> None:
        inputs = make_inputs(
            verdicts={"AAPL": make_verdict(action=Action.HOLD, conviction=0.5)},
            iv_ranks={"AAPL": 0.3},  # below the short-leg floor
            shares_held={"AAPL": 250.0},
            long_term_holdings=frozenset({"AAPL"}),
        )
        result = select_and_size_option_trades(inputs)
        assert ("AAPL", SKIP_NO_VERDICT_MATCH) in result.skipped

    def test_cash_secured_put_when_iv_too_rich_to_buy(self) -> None:
        """buy conviction + IV rank above the buy ceiling + long-term eligible
        → get paid to wait for a lower entry, with full cash reserved."""
        inputs = make_inputs(
            iv_ranks={"AAPL": 0.85},
            long_term_eligible=frozenset({"AAPL"}),
            equity=1_000_000.0,
            cash_available=500_000.0,
        )
        result = select_and_size_option_trades(inputs)
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.structure is OptionStructure.CASH_SECURED_PUT
        assert d.strike == 95.0  # ~5% OTM short put
        # per-position: 1% x 1M = $10,000; obligation 95 x 100 = $9,500 → 1.
        assert d.contracts == 1
        assert d.premium_at_risk == 9_500.0
        assert d.collateral_cash == 9_500.0

    def test_csp_without_cash_fails_closed(self) -> None:
        inputs = make_inputs(
            iv_ranks={"AAPL": 0.85},
            long_term_eligible=frozenset({"AAPL"}),
            equity=1_000_000.0,
            cash_available=1_000.0,  # cannot secure a single contract
        )
        result = select_and_size_option_trades(inputs)
        assert result.decisions == ()
        assert ("AAPL", SKIP_INSUFFICIENT_CASH) in result.skipped

    def test_never_buy_premium_at_extreme_iv_rank(self) -> None:
        inputs = make_inputs(iv_ranks={"AAPL": 0.85})  # buy verdict, not LT-eligible
        result = select_and_size_option_trades(inputs)
        assert result.decisions == ()
        assert ("AAPL", SKIP_NO_VERDICT_MATCH) in result.skipped


class TestFloors:
    def test_conviction_floor(self) -> None:
        inputs = make_inputs(verdicts={"AAPL": make_verdict(conviction=0.69)})
        result = select_and_size_option_trades(inputs)
        assert ("AAPL", SKIP_NO_VERDICT_MATCH) in result.skipped

    def test_missing_iv_rank_fails_closed(self) -> None:
        inputs = make_inputs(iv_ranks={})
        result = select_and_size_option_trades(inputs)
        assert ("AAPL", SKIP_NO_IV_RANK) in result.skipped

    def test_dte_window(self) -> None:
        # 2026-08-21 is a monthly expiry but only 17 DTE (< 21): skipped.
        near = [
            make_chain_row(expiry=date(2026, 8, 21), strike=95.0),
            make_chain_row(expiry=date(2026, 12, 18), strike=95.0),  # 136 DTE (> 90)
        ]
        result = select_and_size_option_trades(make_inputs(chains={"AAPL": near}))
        assert ("AAPL", SKIP_NO_EXPIRY_IN_WINDOW) in result.skipped

    def test_open_interest_floor(self) -> None:
        illiquid = [make_chain_row(strike=95.0, open_interest=99)]
        result = select_and_size_option_trades(make_inputs(chains={"AAPL": illiquid}))
        assert ("AAPL", SKIP_NO_LIQUID_STRIKE) in result.skipped

    def test_spread_floor(self) -> None:
        wide = [make_chain_row(strike=95.0, bid=2.0, ask=2.5)]  # 22% of mid
        result = select_and_size_option_trades(make_inputs(chains={"AAPL": wide}))
        assert ("AAPL", SKIP_NO_LIQUID_STRIKE) in result.skipped

    def test_missing_quote_fields_fail_closed(self) -> None:
        for row in (
            make_chain_row(strike=95.0, bid=None),
            make_chain_row(strike=95.0, ask=None),
            make_chain_row(strike=95.0, open_interest=None),
            make_chain_row(strike=95.0, bid=0.0),
            make_chain_row(strike=95.0, bid=float("nan")),
        ):
            result = select_and_size_option_trades(make_inputs(chains={"AAPL": [row]}))
            assert ("AAPL", SKIP_NO_LIQUID_STRIKE) in result.skipped


class TestCaps:
    def test_sleeve_cap_binds_across_underlyings(self) -> None:
        symbols = [f"SYM{i}" for i in range(8)]
        inputs = make_inputs(
            verdicts={s: make_verdict(s) for s in symbols},
            chains={s: default_chain(s) for s in symbols},
            spot_prices=dict.fromkeys(symbols, 100.0),
            iv_ranks=dict.fromkeys(symbols, 0.4),
        )
        result = select_and_size_option_trades(inputs)
        total = sum(d.premium_at_risk for d in result.decisions)
        assert total <= PARAMS.premium_at_risk_cap * EQUITY + 1e-6
        # 8 x $1,000 wanted, sleeve holds $5,000 → 5 full positions.
        assert len(result.decisions) == 5

    def test_existing_positions_consume_the_sleeve(self) -> None:
        held = tuple(
            make_option_position(underlying=f"HELD{i}", contracts=8, avg_price=5.0)
            for i in range(2)
        )  # 2 x $4,000 premium at risk
        result = select_and_size_option_trades(make_inputs(open_positions=held))
        # $5,000 cap - $8,000 held → nothing left.
        assert result.decisions == ()
        assert any(reason == "sleeve_cap_exhausted" for _, reason in result.skipped)

    def test_max_open_positions(self) -> None:
        held = tuple(
            make_option_position(underlying=f"HELD{i}", contracts=1, avg_price=0.05)
            for i in range(8)
        )
        result = select_and_size_option_trades(make_inputs(open_positions=held))
        assert result.decisions == ()
        assert any(reason == "max_open_positions" for _, reason in result.skipped)

    def test_invalid_equity_fails_closed(self) -> None:
        for equity in (0.0, -1.0, float("nan"), float("inf")):
            result = select_and_size_option_trades(make_inputs(equity=equity))
            assert result.decisions == ()
            assert ("*", "invalid_equity") in result.skipped


# --- hypothesis invariants -------------------------------------------------

maybe_price = st.one_of(
    st.none(),
    st.floats(min_value=0.01, max_value=50.0),
    st.just(float("nan")),
    st.just(0.0),
)
maybe_oi = st.one_of(st.none(), st.integers(min_value=0, max_value=5_000))
maybe_delta = st.one_of(
    st.none(), st.floats(min_value=-1.0, max_value=1.0), st.just(float("nan"))
)
actions = st.sampled_from(list(Action))
alignments = st.sampled_from(list(QuantAlignment))


@st.composite
def scenario(draw: st.DrawFn):
    n = draw(st.integers(min_value=1, max_value=5))
    verdicts = {}
    chains = {}
    spots = {}
    ranks = {}
    shares = {}
    lt_holdings = set()
    lt_eligible = set()
    for i in range(n):
        sym = f"S{i}"
        verdicts[sym] = make_verdict(
            sym,
            action=draw(actions),
            conviction=draw(st.floats(min_value=0.0, max_value=1.0)),
            quant_alignment=draw(alignments),
        )
        spot = draw(
            st.one_of(st.floats(min_value=1.0, max_value=500.0), st.just(float("nan")))
        )
        spots[sym] = spot
        if draw(st.booleans()):
            ranks[sym] = draw(st.floats(min_value=0.0, max_value=1.0))
        base = spot if math.isfinite(spot) and spot > 0 else 100.0
        rows = []
        for pct in (0.9, 0.95, 1.0, 1.05, 1.1):
            for right in (OptionRight.CALL, OptionRight.PUT):
                expiry = draw(
                    st.sampled_from(
                        [MONTHLY_EXPIRY, date(2026, 8, 21), NOW.date() + timedelta(days=40)]
                    )
                )
                rows.append(
                    make_chain_row(
                        sym,
                        expiry=expiry,
                        strike=round(base * pct, 2),
                        right=right,
                        bid=draw(maybe_price),
                        ask=draw(maybe_price),
                        open_interest=draw(maybe_oi),
                        iv=draw(maybe_delta),
                        delta=draw(maybe_delta),
                    )
                )
        chains[sym] = rows
        if draw(st.booleans()):
            shares[sym] = draw(st.floats(min_value=0.0, max_value=1_000.0))
            if draw(st.booleans()):
                lt_holdings.add(sym)
        if draw(st.booleans()):
            lt_eligible.add(sym)
    equity = draw(
        st.one_of(st.floats(min_value=1_000.0, max_value=1e7), st.just(float("nan")))
    )
    cash = draw(st.floats(min_value=0.0, max_value=1e6))
    return make_inputs(
        verdicts=verdicts,
        chains=chains,
        spot_prices=spots,
        iv_ranks=ranks,
        equity=equity,
        cash_available=cash,
        shares_held=shares,
        long_term_holdings=frozenset(lt_holdings),
        long_term_eligible=frozenset(lt_eligible),
    )


@given(scenario())
@settings(max_examples=150, deadline=None)
def test_invariants(inputs) -> None:
    result = select_and_size_option_trades(inputs)

    total = 0.0
    seen: set[str] = set()
    for d in result.decisions:
        # Only the four defined-risk structures; the leg matched at construction.
        assert d.structure in OptionStructure
        assert d.contracts >= 1
        assert d.limit_price > 0.0
        assert d.occ_symbol not in seen  # at most one decision per contract
        seen.add(d.occ_symbol)
        # Floors held.
        assert d.dte >= PARAMS.min_dte
        assert d.dte <= PARAMS.max_dte
        assert 0.0 <= d.iv_rank <= 1.0
        # Per-position cap (covered calls are zero-risk by construction).
        assert d.premium_at_risk <= PARAMS.per_position_cap * inputs.equity + 1e-6
        # Collateral proofs are carried on the decision itself.
        if d.structure is OptionStructure.COVERED_CALL:
            assert d.collateral_shares >= d.contracts * CONTRACT_MULTIPLIER
        if d.structure is OptionStructure.CASH_SECURED_PUT:
            assert d.collateral_cash >= d.contracts * d.strike * CONTRACT_MULTIPLIER - 1e-6
        total += d.premium_at_risk
    if result.decisions:
        existing = sum(p.premium_at_risk for p in inputs.open_positions)
        assert existing + total <= PARAMS.premium_at_risk_cap * inputs.equity + 1e-6
        assert len(seen) <= PARAMS.max_open_positions

    # CSP cash reservations never exceed available cash.
    reserved = sum(
        d.collateral_cash
        for d in result.decisions
        if d.structure is OptionStructure.CASH_SECURED_PUT
    )
    assert reserved <= inputs.cash_available + 1e-6

    # Deterministic: same inputs, same output.
    again = select_and_size_option_trades(inputs)
    assert again == result


def test_determinism_under_param_object_reuse() -> None:
    inputs = make_inputs()
    a = select_and_size_option_trades(inputs, OptionsSizingParams())
    b = select_and_size_option_trades(inputs, OptionsSizingParams())
    assert a == b
