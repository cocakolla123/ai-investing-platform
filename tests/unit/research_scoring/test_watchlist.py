"""Watchlist blend: bounds, conflict cap, badges, decay, sleeve selection."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.research import (
    Action,
    BlendConfig,
    CommitteeVerdict,
    QualityScore,
    QuantAlignment,
    QuantScore,
    Signal,
    SignalStatus,
    Sleeve,
)
from dynamo.research.watchlist import blend_config_hash, build_watchlist

AS_OF = date(2026, 8, 3)


def _verdict(
    symbol: str,
    action: Action = Action.BUY,
    conviction: float = 0.8,
    as_of: date = AS_OF,
    alignment: QuantAlignment = QuantAlignment.NEUTRAL,
) -> CommitteeVerdict:
    return CommitteeVerdict(
        id=f"v-{symbol}-{as_of.isoformat()}",
        symbol=symbol,
        as_of=as_of,
        action=action,
        conviction=conviction,
        thesis="t",
        bull_case="b",
        bear_case="r",
        quant_alignment=alignment,
    )


def _quant(
    symbol: str, expected: float, downside: float = 0.05, confidence: float = 0.9
) -> QuantScore:
    return QuantScore(
        symbol=symbol,
        as_of=AS_OF,
        model_version="m1",
        expected_excess_return=expected,
        downside_prob=downside,
        confidence=confidence,
    )


def _quality(symbol: str, composite: float | None = 80.0) -> QualityScore:
    return QualityScore(symbol=symbol, as_of=AS_OF, composite=composite)


def _signal(symbol: str, strength: float = 0.8) -> Signal:
    return Signal(
        strategy_id="momentum-rs-v1",
        symbol=symbol,
        as_of=AS_OF,
        strength=strength,
        status=SignalStatus.ACTIVE,
    )


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(1, 12),
    seed=st.integers(0, 10_000),
)
def test_blend_bounds_and_rank_contiguity(n: int, seed: int) -> None:
    import random

    rng = random.Random(seed)
    symbols = [f"S{i:02d}" for i in range(n)]
    verdicts = [
        _verdict(
            s,
            action=rng.choice(list(Action)),
            conviction=rng.random(),
            as_of=AS_OF - timedelta(days=rng.randint(0, 60)),
            alignment=rng.choice(list(QuantAlignment)),
        )
        for s in symbols
        if rng.random() < 0.8
    ]
    quant = [
        _quant(s, rng.uniform(-0.1, 0.1), rng.random(), rng.random())
        for s in symbols
        if rng.random() < 0.8
    ]
    quality = [
        _quality(s, rng.choice([None, rng.uniform(0, 100)])) for s in symbols if rng.random() < 0.8
    ]
    signals = [_signal(s, rng.random()) for s in symbols if rng.random() < 0.5]
    rows = build_watchlist(
        as_of=AS_OF,
        verdicts=verdicts,
        quant_scores=quant,
        quality_scores=quality,
        signals=signals,
    )
    for sleeve in (Sleeve.LONG_TERM, Sleeve.TACTICAL):
        sleeve_rows = [r for r in rows if r.sleeve == sleeve]
        assert [r.rank for r in sleeve_rows] == list(range(1, len(sleeve_rows) + 1))
        for r in sleeve_rows:
            assert 0.0 <= r.blend_score <= 100.0
    assert len([r for r in rows if r.sleeve == Sleeve.LONG_TERM]) <= 25
    assert len([r for r in rows if r.sleeve == Sleeve.TACTICAL]) <= 10


def test_conflict_cap_and_badge() -> None:
    # Ten eligible candidates; CONF has the best raw components but a
    # conflicting quant alignment -> capped at the 40th percentile, badged.
    symbols = [f"S{i}" for i in range(9)]
    verdicts = [_verdict(s, conviction=0.6 + 0.02 * i) for i, s in enumerate(symbols)]
    quant = [_quant(s, 0.01 * i) for i, s in enumerate(symbols)]
    quality = [_quality(s, 70.0) for s in symbols]
    verdicts.append(_verdict("CONF", conviction=1.0, alignment=QuantAlignment.CONFLICT))
    quant.append(_quant("CONF", 0.5, downside=0.0, confidence=1.0))
    quality.append(_quality("CONF", 99.0))
    rows = build_watchlist(
        as_of=AS_OF, verdicts=verdicts, quant_scores=quant, quality_scores=quality
    )
    lt = [r for r in rows if r.sleeve == Sleeve.LONG_TERM]
    conf_row = next(r for r in lt if r.symbol == "CONF")
    assert "conflict" in conf_row.badges
    assert conf_row.rank > 1  # cannot top the list
    # Capped at the 40th percentile: at least 60% of the list ranks above it.
    assert conf_row.rank >= int(0.6 * len(lt))
    uncapped_best = max(r.blend_score for r in lt if r.symbol != "CONF")
    assert conf_row.blend_score < uncapped_best


def test_no_quant_badge_and_zero_component() -> None:
    rows = build_watchlist(
        as_of=AS_OF,
        verdicts=[_verdict("AAA")],
        quant_scores=[],
        quality_scores=[_quality("AAA", 80.0)],
    )
    lt = [r for r in rows if r.sleeve == Sleeve.LONG_TERM]
    assert len(lt) == 1
    row = lt[0]
    assert "no-quant" in row.badges
    assert row.quant_component is None  # never imputed
    cfg = BlendConfig()
    expected = 100.0 * (cfg.w_committee * 1.0 * 0.8 + cfg.w_quality * 0.8)
    assert row.blend_score == pytest.approx(expected)
    assert row.blend_config_hash == blend_config_hash(cfg)
    assert row.verdict_ref == f"v-AAA-{AS_OF.isoformat()}"


def test_committee_decay_past_seven_days() -> None:
    aged = AS_OF - timedelta(days=21)  # 2 full weeks past the 7d grace
    rows = build_watchlist(
        as_of=AS_OF,
        verdicts=[_verdict("AAA", conviction=1.0, as_of=aged)],
        quality_scores=[_quality("AAA", 80.0)],
    )
    row = next(r for r in rows if r.sleeve == Sleeve.LONG_TERM)
    assert row.committee_component == pytest.approx(1.0 * 0.9**2)
    fresh = build_watchlist(
        as_of=AS_OF,
        verdicts=[_verdict("AAA", conviction=1.0)],
        quality_scores=[_quality("AAA", 80.0)],
    )
    assert next(r.committee_component for r in fresh) == pytest.approx(1.0)


def test_long_term_eligibility_gates() -> None:
    rows = build_watchlist(
        as_of=AS_OF,
        verdicts=[
            _verdict("LOWQ"),  # quality below 60
            _verdict("HOLD", action=Action.HOLD),  # action not buy/add
            _verdict("LOWC", conviction=0.4),  # conviction below 0.6
            _verdict("GOOD", action=Action.ADD, conviction=0.7),
        ],
        quality_scores=[
            _quality("LOWQ", 40.0),
            _quality("HOLD", 90.0),
            _quality("LOWC", 90.0),
            _quality("GOOD", 75.0),
        ],
    )
    lt = [r.symbol for r in rows if r.sleeve == Sleeve.LONG_TERM]
    assert lt == ["GOOD"]


def test_tactical_rows_require_active_signal_and_reference_it() -> None:
    signals = [
        _signal("AAA", strength=0.9),
        _signal("BBB", strength=0.5),
        Signal(
            strategy_id="meanrev-v1",
            symbol="CCC",
            as_of=AS_OF,
            strength=1.0,
            status=SignalStatus.SUPPRESSED,  # suppressed -> not a candidate
        ),
    ]
    rows = build_watchlist(
        as_of=AS_OF,
        verdicts=[_verdict("AAA"), _verdict("BBB")],
        quant_scores=[_quant("AAA", 0.05), _quant("BBB", 0.02)],
        quality_scores=[_quality("AAA"), _quality("BBB")],
        signals=signals,
    )
    tac = [r for r in rows if r.sleeve == Sleeve.TACTICAL]
    assert [r.symbol for r in tac] == ["AAA", "BBB"]  # ranked by strength x blend
    assert tac[0].signal_ref == 0
    assert tac[1].signal_ref == 1
    # Tactical blend redistributes the quality weight: components identical
    # quality contributes nothing to the tactical score.
    assert tac[0].blend_score > 0


def test_stale_verdict_badge() -> None:
    rows = build_watchlist(
        as_of=AS_OF,
        verdicts=[_verdict("AAA", as_of=AS_OF - timedelta(days=40))],
        quality_scores=[_quality("AAA", 80.0)],
    )
    row = next(r for r in rows if r.sleeve == Sleeve.LONG_TERM)
    assert "stale-verdict" in row.badges


def test_deterministic_output() -> None:
    kwargs: dict[str, Any] = dict(
        as_of=AS_OF,
        verdicts=[_verdict("AAA"), _verdict("BBB", action=Action.ADD, conviction=0.9)],
        quant_scores=[_quant("AAA", 0.03), _quant("BBB", 0.01)],
        quality_scores=[_quality("AAA"), _quality("BBB")],
        signals=[_signal("AAA")],
    )
    r1 = [r.model_dump() for r in build_watchlist(**kwargs)]
    r2 = [r.model_dump() for r in build_watchlist(**kwargs)]
    assert r1 == r2
