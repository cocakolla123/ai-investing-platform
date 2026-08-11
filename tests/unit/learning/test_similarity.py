"""Pure similarity ranking for prior-case retrieval (plan Phase 2) — the
Python-side scorer behind ``ExperienceRepository.similar`` (no DB here)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.learning import DecisionExperience
from dynamo.persistence.repositories.experience import similarity_score

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)
QUERY_AS_OF = date(2026, 8, 10)


def _exp(**overrides: object) -> DecisionExperience:
    base: dict[str, object] = {
        "id": "exp:live:dc-1",
        "created_at": NOW,
        "symbol": "AAPL",
        "as_of": date(2026, 8, 1),
        "sleeve": "long_term",
        "regime": "risk_off",
        "sector": "tech",
        "resolved": True,
    }
    base.update(overrides)
    return DecisionExperience.model_validate(base)


def _score(exp: DecisionExperience, **query: object) -> float:
    defaults: dict[str, object] = {
        "symbol": "MSFT",
        "as_of": QUERY_AS_OF,
        "sector": None,
        "regime": None,
        "sleeve": None,
    }
    defaults.update(query)
    return similarity_score(exp, **defaults)  # type: ignore[arg-type]


def test_recency_decays_and_never_exceeds_one() -> None:
    fresh = _score(_exp(as_of=date(2026, 8, 9)))
    old = _score(_exp(as_of=date(2025, 8, 9)))
    assert 0.0 < old < fresh <= 1.0


def test_same_symbol_beats_recency() -> None:
    old_same_symbol = _score(_exp(symbol="MSFT", as_of=date(2026, 1, 1)))
    fresh_other = _score(_exp(symbol="NVDA", as_of=date(2026, 8, 9)))
    assert old_same_symbol > fresh_other


def test_sector_and_regime_boosts_stack() -> None:
    base = _score(_exp())
    sector = _score(_exp(), sector="tech")
    both = _score(_exp(), sector="tech", regime="risk_off")
    assert base < sector < both
    assert both - base == 1.0  # 0.5 sector + 0.5 regime


def test_no_boost_for_mismatched_dims() -> None:
    assert _score(_exp(sector="energy"), sector="tech") == _score(_exp(sector="energy"))
    assert _score(_exp(regime="risk_on"), regime="risk_off") == _score(_exp(regime="risk_on"))
