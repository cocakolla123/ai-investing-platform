"""Options domain invariants (ADR-0010): unbounded-loss structures are
UNREPRESENTABLE, collateral is validated at proposal, max loss is always
finite, and OCC symbols round-trip."""

from __future__ import annotations

import math
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from dynamo.domain.models.options import (
    CONTRACT_MULTIPLIER,
    STRUCTURE_LEGS,
    CollateralShortfall,
    OptionOrder,
    OptionRight,
    OptionStructure,
    max_loss_dollars,
    occ_symbol,
    parse_occ_symbol,
    validate_covered_call_collateral,
    validate_csp_collateral,
)
from dynamo.domain.models.portfolio import OrderSide
from tests.unit.options.helpers import MONTHLY_EXPIRY, make_decision, make_option_position


class TestOccSymbol:
    def test_golden(self) -> None:
        assert (
            occ_symbol("AAPL", date(2026, 12, 18), OptionRight.CALL, 150.0) == "AAPL261218C00150000"
        )

    def test_round_trip(self) -> None:
        occ = occ_symbol("BRK", MONTHLY_EXPIRY, OptionRight.PUT, 372.5)
        assert parse_occ_symbol(occ) == ("BRK", MONTHLY_EXPIRY, OptionRight.PUT, 372.5)

    def test_malformed_raises(self) -> None:
        for bad in ("", "AAPL", "261218C00150000", "AAPL261218X00150000"):
            with pytest.raises(ValueError):
                parse_occ_symbol(bad)


ALL_COMBOS = [
    (structure, side, right)
    for structure in OptionStructure
    for side in OrderSide
    for right in OptionRight
]


class TestUnboundedLossUnrepresentable:
    """Every (structure, side, right) mismatch — including every naked-short
    shape — raises at construction. Only the four legal legs construct."""

    @pytest.mark.parametrize("structure,side,right", ALL_COMBOS)
    def test_only_legal_legs_construct(
        self, structure: OptionStructure, side: OrderSide, right: OptionRight
    ) -> None:
        build = lambda: OptionOrder(  # noqa: E731
            client_order_id="c1",
            structure=structure,
            side=side,
            right=right,
            occ_symbol=occ_symbol("AAPL", MONTHLY_EXPIRY, right, 100.0),
            underlying="AAPL",
            expiry=MONTHLY_EXPIRY,
            strike=100.0,
            contracts=1,
            limit_price=1.0,
        )
        if (side, right) == STRUCTURE_LEGS[structure]:
            assert build().structure is structure
        else:
            with pytest.raises(ValidationError, match="unrepresentable"):
                build()

    def test_no_naked_short_structure_exists(self) -> None:
        """The enum itself is the proof: every SELL leg is collateralized."""
        for structure, (side, _right) in STRUCTURE_LEGS.items():
            if side is OrderSide.SELL:
                assert structure in (
                    OptionStructure.COVERED_CALL,
                    OptionStructure.CASH_SECURED_PUT,
                )
        assert len(OptionStructure) == 4


class TestCollateralValidators:
    def test_covered_call_without_shares_raises(self) -> None:
        with pytest.raises(CollateralShortfall):
            validate_covered_call_collateral(2, 199.0)
        validate_covered_call_collateral(2, 200.0)  # exactly covered — fine

    def test_covered_call_nan_shares_fail_closed(self) -> None:
        with pytest.raises(CollateralShortfall):
            validate_covered_call_collateral(1, float("nan"))

    def test_csp_without_cash_raises(self) -> None:
        with pytest.raises(CollateralShortfall):
            validate_csp_collateral(1, 95.0, 9_499.0)
        validate_csp_collateral(1, 95.0, 9_500.0)  # exactly reserved — fine

    def test_csp_nan_cash_fails_closed(self) -> None:
        with pytest.raises(CollateralShortfall):
            validate_csp_collateral(1, 95.0, float("nan"))

    def test_decision_construction_is_the_proposal_check(self) -> None:
        """AT PROPOSAL: a decision without collateral cannot exist."""
        with pytest.raises(CollateralShortfall):
            make_decision(OptionStructure.COVERED_CALL, contracts=2, collateral_shares=150.0)
        with pytest.raises(CollateralShortfall):
            make_decision(
                OptionStructure.CASH_SECURED_PUT,
                strike=95.0,
                contracts=1,
                collateral_cash=1_000.0,
            )
        ok = make_decision(OptionStructure.COVERED_CALL, contracts=2, collateral_shares=200.0)
        assert ok.structure is OptionStructure.COVERED_CALL


class TestMaxLoss:
    @given(
        structure=st.sampled_from(list(OptionStructure)),
        strike=st.floats(min_value=0.01, max_value=10_000.0),
        premium=st.floats(min_value=0.0, max_value=1_000.0),
        contracts=st.integers(min_value=1, max_value=1_000),
    )
    @settings(max_examples=200, deadline=None)
    def test_always_finite_and_nonnegative(
        self, structure: OptionStructure, strike: float, premium: float, contracts: int
    ) -> None:
        loss = max_loss_dollars(structure, strike, premium, contracts)
        assert math.isfinite(loss)
        assert loss >= 0.0

    def test_golden_values(self) -> None:
        assert max_loss_dollars(OptionStructure.LONG_CALL, 100.0, 2.5, 4) == 1_000.0
        assert max_loss_dollars(OptionStructure.CASH_SECURED_PUT, 95.0, 2.0, 1) == 9_300.0
        assert max_loss_dollars(OptionStructure.COVERED_CALL, 105.0, 1.5, 2) == 0.0


class TestPremiumAtRisk:
    def test_long_premium(self) -> None:
        p = make_option_position(OptionStructure.LONG_CALL, contracts=4, avg_price=2.5)
        assert p.premium_at_risk == 4 * 2.5 * CONTRACT_MULTIPLIER

    def test_csp_assignment_obligation(self) -> None:
        p = make_option_position(
            OptionStructure.CASH_SECURED_PUT, strike=95.0, contracts=1, avg_price=2.0
        )
        assert p.premium_at_risk == 9_500.0

    def test_covered_call_is_share_covered(self) -> None:
        p = make_option_position(OptionStructure.COVERED_CALL, contracts=2, avg_price=1.5)
        assert p.premium_at_risk == 0.0
