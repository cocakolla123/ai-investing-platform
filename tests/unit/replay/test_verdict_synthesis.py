"""Quant-verdict synthesis mapping tests (plan Phase 4) — pure, no I/O."""

from __future__ import annotations

from datetime import date

from dynamo.domain.models.research import Action, QuantAlignment, QuantScore
from dynamo.replay.runner import (
    DEFAULT_EXPECTED_DOWNSIDE_PCT,
    synthesize_verdicts,
    synthesize_watchlist,
)

AS_OF = date(2026, 6, 1)


def _score(
    symbol: str,
    eer: float | None,
    downside: float | None = 0.1,
    coverage: float = 1.0,
) -> QuantScore:
    return QuantScore(
        symbol=symbol,
        as_of=AS_OF,
        model_version="test",
        expected_excess_return=eer,
        downside_prob=downside,
        confidence=0.5,
        feature_coverage=coverage,
    )


def _five() -> dict[str, QuantScore]:
    # Percentiles (rank by eer desc): TOP 1.0, HIGH 0.8, MID 0.6, LOW 0.4, BOT 0.2.
    return {
        "TOP": _score("TOP", 0.06),
        "HIGH": _score("HIGH", 0.05),
        "MID": _score("MID", 0.01),
        "LOW": _score("LOW", 0.005),
        "BOT": _score("BOT", -0.01),
    }


def test_actions_follow_percentile_thresholds() -> None:
    verdicts = synthesize_verdicts(_five(), AS_OF, 20)
    assert verdicts["TOP"].action is Action.BUY
    assert verdicts["HIGH"].action is Action.BUY  # percentile exactly 0.8
    assert verdicts["MID"].action is Action.HOLD
    assert verdicts["LOW"].action is Action.HOLD
    assert verdicts["BOT"].action is Action.AVOID  # percentile exactly 0.2


def test_conviction_comes_from_percentile() -> None:
    verdicts = synthesize_verdicts(_five(), AS_OF, 20)
    assert verdicts["TOP"].conviction == 1.0
    assert verdicts["HIGH"].conviction == 0.8
    assert verdicts["MID"].conviction == 0.5  # HOLD -> fixed 0.5
    assert verdicts["BOT"].conviction == 0.8  # AVOID -> 1 - percentile


def test_high_downside_prob_forces_avoid_even_at_top_rank() -> None:
    scores = _five()
    scores["TOP"] = _score("TOP", 0.06, downside=0.7)
    verdicts = synthesize_verdicts(scores, AS_OF, 20)
    assert verdicts["TOP"].action is Action.AVOID
    assert verdicts["TOP"].conviction == 0.7  # max(1 - pct, downside_prob)


def test_negative_expected_return_never_buys() -> None:
    scores = {"NEG": _score("NEG", -0.02), "ZERO": _score("ZERO", 0.0)}
    verdicts = synthesize_verdicts(scores, AS_OF, 20)
    assert all(v.action is not Action.BUY for v in verdicts.values())


def test_baseline_all_none_scores_synthesize_nothing() -> None:
    """baseline-v0 (no trained models) yields all-None expected returns —
    honest abstention, no verdicts, exactly like a quant-only night today."""
    scores = {s: _score(s, None, None) for s in ("AAA", "BBB")}
    assert synthesize_verdicts(scores, AS_OF, 20) == {}


def test_verdict_shape_and_expectations() -> None:
    verdicts = synthesize_verdicts(_five(), AS_OF, 15)
    v = verdicts["TOP"]
    assert v.id == f"rv-{AS_OF.isoformat()}-TOP"
    assert v.as_of == AS_OF
    assert v.horizon_days == 15
    assert v.expected_upside_pct == 6.0  # eer x 100
    assert v.expected_downside_pct == DEFAULT_EXPECTED_DOWNSIDE_PCT
    assert v.quant_alignment is QuantAlignment.AGREE
    assert v.model_id == "none"  # NO LLM anywhere


def test_synthesis_is_deterministic_and_tie_stable() -> None:
    scores = {s: _score(s, 0.05) for s in ("BBB", "AAA", "CCC")}  # exact ties
    first = synthesize_verdicts(scores, AS_OF, 20)
    second = synthesize_verdicts(dict(reversed(list(scores.items()))), AS_OF, 20)
    assert {s: v.model_dump() for s, v in first.items()} == {
        s: v.model_dump() for s, v in second.items()
    }
    # Ties break by symbol: AAA gets the best percentile.
    assert first["AAA"].conviction >= first["BBB"].conviction >= first["CCC"].conviction


# -- minimum-coverage gate (campaign findings Phase 1) -----------------------


def test_low_coverage_symbols_abstain_instead_of_being_ranked() -> None:
    """A name whose feature row is mostly holes should not compete for a
    deep-dive slot: it abstains entirely rather than being ranked."""
    scores = _five()
    scores["THIN"] = _score("THIN", 0.07, coverage=0.22)  # would rank FIRST
    verdicts = synthesize_verdicts(scores, AS_OF, 20, 0.35)
    assert "THIN" not in verdicts
    assert set(verdicts) == set(_five())
    # ...and the gate does not distort the survivors' percentiles.
    assert verdicts["TOP"].conviction == 1.0


def test_coverage_gate_defaults_off_and_is_inclusive_at_the_threshold() -> None:
    scores = {
        "AT": _score("AT", 0.05, coverage=0.35),
        "BELOW": _score("BELOW", 0.04, coverage=0.3499),
    }
    assert set(synthesize_verdicts(scores, AS_OF, 20)) == {"AT", "BELOW"}  # 0.0 default
    assert set(synthesize_verdicts(scores, AS_OF, 20, 0.35)) == {"AT"}


def test_coverage_gate_can_abstain_completely() -> None:
    scores = {s: _score(s, 0.05, coverage=0.1) for s in ("AAA", "BBB")}
    assert synthesize_verdicts(scores, AS_OF, 20, 0.35) == {}


def test_replay_config_carries_the_gate_into_the_config_hash() -> None:
    from dynamo.replay.runner import ReplayConfig

    base = dict(start=AS_OF, end=AS_OF, label="gate")
    default = ReplayConfig(**base)  # type: ignore[arg-type]
    assert default.min_feature_coverage == 0.35
    loosened = ReplayConfig(**base, min_feature_coverage=0.0)  # type: ignore[arg-type]
    assert loosened.run_id != default.run_id


def test_watchlist_rows_only_for_buys_ranked_by_conviction() -> None:
    verdicts = synthesize_verdicts(_five(), AS_OF, 20)
    rows = synthesize_watchlist(verdicts, AS_OF)
    assert [r.symbol for r in rows] == ["TOP", "HIGH"]
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].verdict_ref == verdicts["TOP"].id
    # Informational quality component present (constructor floor is disabled).
    assert rows[0].quality_component == 100.0
