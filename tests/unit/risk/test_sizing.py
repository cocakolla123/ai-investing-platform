"""Sizing cascade: hypothesis invariants + mutation-style per-clamp tests.

Doc 06 section 4: no input can produce a weight exceeding any cap; NaN/inf
inputs give 0, never an exception; higher sigma never yields a larger base.
Every binding clamp must appear in clamps_applied.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.sizing import (
    CLAMP_CLUSTER,
    CLAMP_FLOOR,
    CLAMP_GROSS,
    CLAMP_INVALID,
    CLAMP_KELLY,
    CLAMP_LIQUIDITY,
    CLAMP_PER_POSITION,
    CLAMP_SECTOR,
    CLAMP_TACTICAL,
    CLAMP_VETO,
    SizingCandidate,
    SizingContext,
    size_positions,
)

CFG = RiskConfig()
EQUITY = 100_000.0
BIG_ADV = 1e9  # liquidity cap never binds


def _wild(bounded: st.SearchStrategy[float]) -> st.SearchStrategy[float]:
    """Bounded floats plus the adversarial specials: NaN and +/-inf."""
    return st.one_of(
        bounded,
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
    )


wild_floats = _wild(st.floats(min_value=-10.0, max_value=10.0))
sigmas = _wild(st.floats(min_value=-1.0, max_value=3.0))
convictions = _wild(st.floats(min_value=-2.0, max_value=2.0))
advs = _wild(st.floats(min_value=-1e6, max_value=1e12))
maybe_mu = st.one_of(st.none(), wild_floats)


@st.composite
def candidates(draw: st.DrawFn, index: int = 0) -> SizingCandidate:
    return SizingCandidate(
        symbol=f"S{index}",
        sleeve=draw(st.sampled_from([Sleeve.LONG_TERM, Sleeve.TACTICAL])),
        conviction=draw(convictions),
        sigma_annual=draw(sigmas),
        adv_dollar_20d=draw(advs),
        expected_excess_return=draw(maybe_mu),
        vol_forecast_annual=draw(st.one_of(st.none(), sigmas)),
        sector=draw(st.sampled_from([None, "Tech", "Energy"])),
        cluster_id=draw(st.sampled_from([None, "c1", "c2"])),
    )


@st.composite
def candidate_lists(draw: st.DrawFn) -> list[SizingCandidate]:
    n = draw(st.integers(min_value=1, max_value=6))
    return [draw(candidates(index=i)) for i in range(n)]


@st.composite
def contexts(draw: st.DrawFn) -> SizingContext:
    return SizingContext(
        equity=draw(_wild(st.floats(min_value=-1e6, max_value=1e7))),
        existing_weights={"HELD": draw(st.floats(-0.5, 0.5))},
        sector_by_symbol={"HELD": "Tech"},
        cluster_by_symbol={"HELD": "c1"},
        vetoed=draw(st.sampled_from([frozenset(), frozenset({"S0"})])),
        sleeve_budget=draw(_wild(st.floats(min_value=-1.0, max_value=2.0))),
        risk_multiplier=draw(st.sampled_from([1.0, 0.5, 0.0])),
        tactical_multiplier=draw(st.sampled_from([1.0, 0.5])),
        tighten_factor=draw(st.sampled_from([1.0, 0.75])),
    )


@settings(max_examples=100, deadline=None)
@given(cands=candidate_lists(), ctx=contexts())
def test_sizing_never_exceeds_any_cap_and_never_raises(
    cands: list[SizingCandidate], ctx: SizingContext
) -> None:
    results = size_positions(cands, CFG, ctx)
    assert len(results) == len(cands)

    tighten = ctx.tighten_factor if ctx.tighten_factor in (1.0, 0.75) else 0.75
    sector_totals: dict[str, float] = {}
    cluster_totals: dict[str, float] = {}
    total = 0.0
    for cand, sized in zip(cands, results, strict=True):
        w = sized.weight
        assert math.isfinite(w) and w >= 0.0
        assert w <= CFG.per_position_hard + 1e-9
        if cand.sleeve is Sleeve.TACTICAL:
            assert w <= CFG.tactical_position_cap * ctx.tactical_multiplier + 1e-9
        else:
            assert w <= CFG.per_position_cap + 1e-9
        if sized.kelly_ceiling is not None:
            assert w <= sized.kelly_ceiling + 1e-9
        if w > 0.0:
            assert w >= CFG.min_position_weight - 1e-12
        if cand.symbol in ctx.vetoed:
            assert w == 0.0 and sized.clamps_applied == (CLAMP_VETO,)
        if cand.sector is not None:
            sector_totals[cand.sector] = sector_totals.get(cand.sector, 0.0) + w
        if cand.cluster_id is not None:
            cluster_totals[cand.cluster_id] = cluster_totals.get(cand.cluster_id, 0.0) + w
        total += w

    existing = {s: w for s, w in ctx.existing_weights.items() if math.isfinite(w) and w > 0.0}
    existing_gross = sum(existing.values())
    # New weight never exceeds the remaining headroom under each cap (existing
    # overweight cannot be fixed by sizing; it just leaves zero headroom).
    for sector, new_weight in sector_totals.items():
        used = sum(w for s, w in existing.items() if ctx.sector_by_symbol.get(s) == sector)
        assert new_weight <= max(0.0, CFG.sector_cap * tighten - used) + 1e-9
    for cluster, new_weight in cluster_totals.items():
        used = sum(w for s, w in existing.items() if ctx.cluster_by_symbol.get(s) == cluster)
        assert new_weight <= max(0.0, CFG.cluster_cap * tighten - used) + 1e-9
    assert total <= max(0.0, CFG.gross_cap - CFG.cash_floor - existing_gross) + 1e-9


@settings(max_examples=60, deadline=None)
@given(
    sigma_low=st.floats(min_value=0.05, max_value=1.0),
    bump=st.floats(min_value=1e-3, max_value=2.0),
    budget=st.floats(min_value=0.0, max_value=1.0),
)
def test_higher_sigma_never_bigger_base(sigma_low: float, bump: float, budget: float) -> None:
    def cand(symbol: str, sigma: float) -> SizingCandidate:
        return SizingCandidate(
            symbol=symbol,
            sleeve=Sleeve.LONG_TERM,
            conviction=0.5,
            sigma_annual=sigma,
            adv_dollar_20d=BIG_ADV,
        )

    ctx = SizingContext(equity=EQUITY, sleeve_budget=budget)
    low, high = size_positions([cand("LOW", sigma_low), cand("HIGH", sigma_low + bump)], CFG, ctx)
    assert high.base_weight <= low.base_weight + 1e-12


def _single(
    candidate: SizingCandidate, ctx: SizingContext
) -> tuple[float, tuple[str, ...], float, float | None]:
    (sized,) = size_positions([candidate], CFG, ctx)
    return sized.weight, sized.clamps_applied, sized.base_weight, sized.kelly_ceiling


def _plain(symbol: str = "AAA", **overrides: object) -> SizingCandidate:
    base: dict[str, object] = {
        "symbol": symbol,
        "sleeve": Sleeve.LONG_TERM,
        "conviction": 0.5,  # tilt exactly 1.0
        "sigma_annual": 0.20,
        "adv_dollar_20d": BIG_ADV,
    }
    base.update(overrides)
    return SizingCandidate(**base)  # type: ignore[arg-type]


class TestEachClampBindsExactlyWhereConstructed:
    """Mutation-style: for each clamp, an input where exactly it binds, with
    the pre-clamp weight provably above the cap."""

    def test_kelly_ceiling_binds(self) -> None:
        # ceiling = 0.25 * mu / (sigma^2 * 20/252) chosen inside (floor, budget)
        mu = 0.0004
        cand = _plain(expected_excess_return=mu, vol_forecast_annual=0.20)
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.06)
        weight, clamps, base, ceiling = _single(cand, ctx)
        assert ceiling is not None
        expected = 0.25 * mu / (0.20**2 * (20.0 / 252.0))
        assert math.isclose(ceiling, expected, rel_tol=1e-12)
        assert base * 1.0 > ceiling  # pre-clamp exceeded the cap
        assert weight == ceiling and clamps == (CLAMP_KELLY,)

    def test_negative_edge_kelly_zeroes(self) -> None:
        cand = _plain(expected_excess_return=-0.01)
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.06)
        weight, clamps, _, ceiling = _single(cand, ctx)
        assert ceiling == 0.0 and weight == 0.0
        assert CLAMP_KELLY in clamps

    def test_per_position_cap_binds(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.20)
        weight, clamps, base, _ = _single(_plain(), ctx)
        assert base > CFG.per_position_cap
        assert weight == CFG.per_position_cap and clamps == (CLAMP_PER_POSITION,)

    def test_tactical_position_cap_binds(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.20)
        weight, clamps, base, _ = _single(_plain(sleeve=Sleeve.TACTICAL), ctx)
        assert base > CFG.tactical_position_cap
        assert weight == CFG.tactical_position_cap and clamps == (CLAMP_TACTICAL,)

    def test_sector_cap_binds(self) -> None:
        ctx = SizingContext(
            equity=EQUITY,
            existing_weights={"HELD": 0.20},
            sector_by_symbol={"HELD": "Tech"},
            sleeve_budget=0.07,
        )
        weight, clamps, base, _ = _single(_plain(sector="Tech"), ctx)
        available = CFG.sector_cap - 0.20
        assert base > available
        assert math.isclose(weight, available) and clamps == (CLAMP_SECTOR,)

    def test_cluster_cap_binds(self) -> None:
        ctx = SizingContext(
            equity=EQUITY,
            existing_weights={"HELD": 0.27},
            cluster_by_symbol={"HELD": "c9"},
            sleeve_budget=0.07,
        )
        weight, clamps, base, _ = _single(_plain(cluster_id="c9"), ctx)
        available = CFG.cluster_cap - 0.27
        assert base > available
        assert math.isclose(weight, available) and clamps == (CLAMP_CLUSTER,)

    def test_liquidity_cap_binds(self) -> None:
        adv = 10_000.0  # cap = 3d * 10% * adv / equity = 0.03
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.07)
        weight, clamps, base, _ = _single(_plain(adv_dollar_20d=adv), ctx)
        cap = CFG.liquidity_max_days * CFG.liquidity_adv_participation * adv / EQUITY
        assert base > cap
        assert math.isclose(weight, cap) and clamps == (CLAMP_LIQUIDITY,)

    def test_gross_cap_binds(self) -> None:
        ctx = SizingContext(
            equity=EQUITY,
            existing_weights={"H1": 0.45, "H2": 0.45},
            sleeve_budget=0.07,
        )
        weight, clamps, base, _ = _single(_plain(), ctx)
        available = CFG.gross_cap - CFG.cash_floor - 0.90
        assert base > available
        assert math.isclose(weight, available) and clamps == (CLAMP_GROSS,)

    def test_min_weight_floor_zeroes(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.004)
        weight, clamps, base, _ = _single(_plain(), ctx)
        assert 0.0 < base < CFG.min_position_weight
        assert weight == 0.0 and clamps == (CLAMP_FLOOR,)

    def test_no_clamp_when_nothing_binds(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.06)
        weight, clamps, base, _ = _single(_plain(), ctx)
        assert weight == base and clamps == ()


class TestRobustness:
    def test_nan_sigma_gives_zero_not_exception(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.10)
        weight, clamps, _, _ = _single(_plain(sigma_annual=float("nan")), ctx)
        assert weight == 0.0 and clamps == (CLAMP_INVALID,)

    def test_negative_sigma_gives_zero(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.10)
        weight, clamps, _, _ = _single(_plain(sigma_annual=-0.2), ctx)
        assert weight == 0.0 and clamps == (CLAMP_INVALID,)

    def test_inf_conviction_gives_zero(self) -> None:
        ctx = SizingContext(equity=EQUITY, sleeve_budget=0.10)
        weight, clamps, _, _ = _single(_plain(conviction=float("inf")), ctx)
        assert weight == 0.0 and clamps == (CLAMP_INVALID,)

    def test_invalid_equity_zeroes_everything(self) -> None:
        ctx = SizingContext(equity=float("nan"), sleeve_budget=0.10)
        weight, clamps, _, _ = _single(_plain(), ctx)
        assert weight == 0.0 and clamps == (CLAMP_INVALID,)


def test_veto_contract_high_conviction_buy_still_zero() -> None:
    """Committee buy at conviction 0.9 on a vetoed symbol -> weight 0."""
    cand = _plain(symbol="EVIL", conviction=0.9)
    ctx = SizingContext(equity=EQUITY, vetoed=frozenset({"EVIL"}), sleeve_budget=0.10)
    weight, clamps, base, _ = _single(cand, ctx)
    assert weight == 0.0
    assert clamps == (CLAMP_VETO,)
    assert base == 0.0  # a veto is absolute: no base risk is even computed


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    budget=st.floats(min_value=0.0, max_value=1.0),
    used=st.floats(min_value=0.0, max_value=0.29),
)
def test_cluster_cap_respected_post_cascade(n: int, budget: float, used: float) -> None:
    """Many candidates in one correlation cluster are capped as one bet."""
    cands = [
        SizingCandidate(
            symbol=f"AI{i}",
            sleeve=Sleeve.LONG_TERM,
            conviction=0.9,
            sigma_annual=0.20 + 0.01 * i,
            adv_dollar_20d=BIG_ADV,
            cluster_id="ai",
        )
        for i in range(n)
    ]
    ctx = SizingContext(
        equity=EQUITY,
        existing_weights={"HELD": used},
        cluster_by_symbol={"HELD": "ai"},
        sleeve_budget=budget,
    )
    results = size_positions(cands, CFG, ctx)
    assert sum(r.weight for r in results) + used <= CFG.cluster_cap + 1e-9
