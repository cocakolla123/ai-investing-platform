"""Prior-similar-case evidence lines in the dossier journal_hits channel
(plan Phase 2) — pure helpers over a fake experience repository."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from orchestration.assets.research import (
    PRIOR_CASE_PREFIX,
    _prior_case_hits,
    _prior_case_line,
)

from dynamo.domain.models.learning import DecisionExperience, OutcomeQuality

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)


def _exp(**overrides: object) -> DecisionExperience:
    base: dict[str, object] = {
        "id": "exp:live:dc-1",
        "created_at": NOW,
        "symbol": "NVDA",
        "as_of": date(2026, 7, 1),
        "sleeve": "long_term",
        "regime": "risk_off",
        "sector": "tech",
        "action": "buy",
        "conviction": 0.7,
        "resolved": True,
        "pnl_pct": -3.2,
        "outcome_quality": OutcomeQuality.LOSS,
    }
    base.update(overrides)
    return DecisionExperience.model_validate(base)


def test_prior_case_line_embeds_evidence_not_rule_guidance() -> None:
    line = _prior_case_line(_exp())
    assert line.startswith("Prior similar case (evidence, not rule): ")
    assert line == (
        "Prior similar case (evidence, not rule): NVDA buy conviction 0.70 "
        "in risk_off/tech -> loss, pnl -3.2% (2026-07-01)"
    )


def test_prior_case_line_degrades_on_missing_fields() -> None:
    line = _prior_case_line(
        _exp(action=None, conviction=None, pnl_pct=None, regime=None, sector=None)
    )
    assert line.startswith(PRIOR_CASE_PREFIX)
    assert "unknown-action" in line and "conviction n/a" in line
    assert "unknown-regime/unknown-sector" in line and "pnl n/a" in line


class _FakeExpRepo:
    def __init__(self, cases: list[DecisionExperience]) -> None:
        self._cases = cases
        self.calls: list[dict[str, Any]] = []

    def similar(self, **kwargs: Any) -> list[DecisionExperience]:
        self.calls.append(kwargs)
        return self._cases[: kwargs["limit"]]


def test_prior_case_hits_render_per_symbol_with_prefix() -> None:
    repo = _FakeExpRepo([_exp(), _exp(id="exp:live:dc-2", symbol="AMD", pnl_pct=4.0)])
    hits = _prior_case_hits(
        repo,
        ["NVDA"],
        date(2026, 8, 10),
        top_k=2,
        sector_for={"NVDA": "tech"},
        regime="risk_off",
    )
    assert [eid for eid, _ in hits["NVDA"]] == ["exp:live:dc-1", "exp:live:dc-2"]
    assert all(text.startswith(PRIOR_CASE_PREFIX) for _, text in hits["NVDA"])
    # The repo received the query's point-in-time and boost context.
    assert repo.calls == [
        {
            "symbol": "NVDA",
            "as_of": date(2026, 8, 10),
            "sector": "tech",
            "regime": "risk_off",
            "limit": 2,
        }
    ]
