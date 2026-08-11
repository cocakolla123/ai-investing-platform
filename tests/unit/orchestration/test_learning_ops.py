"""Learning-ops helpers on in-memory fakes: the repaired decision-context
join, path-based calibration outcomes, and the new agent/strategy scopes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from orchestration.assets.learning_ops import (
    _calibration_observations,
    _decision_context_for,
    _outcome_for,
    _trade_id,
    hier_calibration_inputs,
    log_calibrated_probabilities,
)

from dynamo.domain.models.learning import DecisionExperience

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)


# --- tiny fakes for the decision-context join (post_trade_reviews path) ----


@dataclass(frozen=True)
class _Closed:
    symbol: str
    closed_at: datetime


@dataclass(frozen=True)
class _Fill:
    client_order_id: str
    filled_at: datetime


@dataclass(frozen=True)
class _Order:
    client_order_id: str
    decision_context_id: str


@dataclass(frozen=True)
class _Dc:
    id: str


class _TradesRepo:
    def __init__(self, fills: list[_Fill]) -> None:
        self._fills = fills

    def fills(self, symbol: str | None = None) -> list[_Fill]:
        return list(self._fills)


class _OrdersRepo:
    def __init__(self, orders: dict[str, _Order]) -> None:
        self._orders = orders

    def get(self, client_order_id: str) -> _Order | None:
        return self._orders.get(client_order_id)


class _DcRepo:
    """get() looks up by decision_context.id — exactly like the real repo."""

    def __init__(self, contexts: dict[str, _Dc]) -> None:
        self._contexts = contexts

    def get(self, dc_id: str) -> _Dc | None:
        return self._contexts.get(dc_id)


def test_post_trade_review_path_finds_decision_context_by_id() -> None:
    """Regression for the dc_id bug: orders store the id append() now returns,
    so the review path resolves the frozen context instead of skipping."""
    closed = _Closed(symbol="AAPL", closed_at=NOW)
    trades = _TradesRepo([_Fill("ord-1", NOW - timedelta(days=5)), _Fill("ord-2", NOW)])
    orders = _OrdersRepo(
        {
            "ord-1": _Order("ord-1", "dc-entry"),
            "ord-2": _Order("ord-2", "dc-exit"),
        }
    )
    dc_repo = _DcRepo({"dc-entry": _Dc("dc-entry"), "dc-exit": _Dc("dc-exit")})
    dc = _decision_context_for(closed, trades, orders, dc_repo)
    assert dc is not None and dc.id == "dc-exit"  # latest fill's context wins


def test_hash_valued_reference_still_skips() -> None:
    """A stale (unrepaired) hash reference resolves to no context — the old
    failure mode the migration + return-value fix eliminate for new rows."""
    closed = _Closed(symbol="AAPL", closed_at=NOW)
    trades = _TradesRepo([_Fill("ord-1", NOW)])
    orders = _OrdersRepo({"ord-1": _Order("ord-1", "a3f9...row-hash-not-an-id")})
    dc_repo = _DcRepo({"dc-1": _Dc("dc-1")})
    assert _decision_context_for(closed, trades, orders, dc_repo) is None


def test_trade_id_formula_is_shared() -> None:
    assert _trade_id(_Closed("AAPL", NOW)) == "AAPL-2026-08-10"


# --- calibration outcome: path-based where expectations exist --------------


def _dc_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trade_id": "AAPL-2026-08-10",
        "conviction": 0.8,
        "payload": {"outcome_pnl_pct": -2.0},
        "symbol": "AAPL",
        "symbol_v": "AAPL",
        "as_of": date(2026, 8, 1),
        "sleeve": "long_term",
        "created_at": NOW - timedelta(days=30),
        "expected_upside_pct": 10.0,
        "expected_downside_pct": 5.0,
        "horizon_days": 20,
    }
    row.update(overrides)
    return row


def test_outcome_uses_path_when_expectations_present() -> None:
    # Path reaches half the expected upside (5%) before the -5% floor →
    # achieved, even though the review's realized pnl was negative.
    path = [1.0, 3.0, 6.0, -1.0]
    outcome = _outcome_for(_dc_row(), lambda sym, start, horizon: path)
    assert outcome is True

    # Floor hit first → not achieved, even with positive realized pnl.
    losing_path = [-6.0, 8.0]
    outcome = _outcome_for(
        _dc_row(payload={"outcome_pnl_pct": 4.0}), lambda sym, start, horizon: losing_path
    )
    assert outcome is False


def test_outcome_falls_back_to_pnl_sign_for_zero_expectation_rows() -> None:
    # Pre-repair rows hardcoded (0, 0) expectations — path scoring undefined.
    row = _dc_row(expected_upside_pct=0.0, expected_downside_pct=0.0)
    assert _outcome_for(row, lambda *a: [5.0]) is False  # pnl -2.0
    row = _dc_row(
        expected_upside_pct=0.0,
        expected_downside_pct=0.0,
        payload={"outcome_pnl_pct": 3.0},
    )
    assert _outcome_for(row, lambda *a: []) is True

    # Missing bars → fall back too, never crash.
    assert _outcome_for(_dc_row(payload={"outcome_pnl_pct": 1.0}), lambda *a: []) is True


# --- calibration scopes: committee + strategy + per-agent ------------------


def _opinion(role: str, conviction: float | None, as_of: date = date(2026, 8, 1)) -> dict:
    payload = {} if conviction is None else {"conviction": conviction}
    return {"symbol": "AAPL", "as_of": as_of, "role": role, "payload": payload}


def test_calibration_emits_agent_and_strategy_scopes() -> None:
    merged = [_dc_row()]
    opinions = [
        _opinion("bull", 0.9),
        _opinion("bear", 0.3),
        _opinion("arbiter", 0.8),
        # Different day — must not join onto this trade's verdict.
        _opinion("bull", 0.1, as_of=date(2026, 7, 1)),
        # No conviction in payload — skipped.
        _opinion("bull", None),
    ]
    observations = _calibration_observations(merged, opinions, lambda row: True)
    scopes = {(o.scope_type, o.scope_id) for o in observations}
    assert scopes == {
        ("committee", "arbiter"),
        ("strategy", "long_term"),
        ("agent", "bull"),
        ("agent", "bear"),
        ("agent", "arbiter"),
    }
    by_scope = {(o.scope_type, o.scope_id): o for o in observations}
    assert by_scope[("agent", "bull")].conviction == 0.9
    assert by_scope[("strategy", "long_term")].conviction == 0.8  # committee conviction
    assert all(o.outcome for o in observations)  # shared trade outcome


def test_calibration_skips_rows_without_linked_verdict() -> None:
    row = _dc_row(conviction=None)
    assert _calibration_observations([row], [], lambda r: True) == []


# --- hierarchical scopes from decision_experience (plan Phase 2) ------------


def _experience(i: int, **overrides: Any) -> DecisionExperience:
    base: dict[str, Any] = {
        "id": f"exp:live:dc-{i}",
        "decision_context_id": f"dc-{i}",
        "created_at": NOW,
        "symbol": "AAPL",
        "as_of": date(2026, 8, i),
        "sleeve": "long_term",
        "regime": "risk_off",
        "sector": "tech",
        "quant_alignment": "agree",
        "conviction": 0.7,
        "resolved": True,
        "target_achieved": True,
        "label": "win",
    }
    base.update(overrides)
    return DecisionExperience.model_validate(base)


class _FakeExpRepo:
    def __init__(self, experiences: list[DecisionExperience]) -> None:
        self._experiences = experiences
        self.queried = False

    def resolved_live(self) -> list[DecisionExperience]:
        self.queried = True
        return list(self._experiences)


def test_hier_scopes_emitted_for_every_chain_prefix() -> None:
    repo = _FakeExpRepo([_experience(1), _experience(2, sector=None)])
    experiences, observations = hier_calibration_inputs(repo, enabled=True)
    assert len(experiences) == 2
    assert all(o.scope_type == "hier" for o in observations)
    scope_ids = [o.scope_id for o in observations]
    # Full key: 4 populated dims -> 5 prefixes; sector-less key -> 4.
    assert scope_ids[:5] == [
        "sleeve=long_term|regime=risk_off|sector=tech|quant_alignment=agree",
        "sleeve=long_term|regime=risk_off|sector=tech",
        "sleeve=long_term|regime=risk_off",
        "sleeve=long_term",
        "global",
    ]
    assert scope_ids[5:] == [
        "sleeve=long_term|regime=risk_off|quant_alignment=agree",
        "sleeve=long_term|regime=risk_off",
        "sleeve=long_term",
        "global",
    ]


def test_hier_scopes_gated_off_by_setting() -> None:
    repo = _FakeExpRepo([_experience(1)])
    experiences, observations = hier_calibration_inputs(repo, enabled=False)
    assert experiences == [] and observations == []
    assert repo.queried is False  # gate short-circuits before any read


def test_log_calibrated_probabilities_writes_only_uncalibrated_rows() -> None:
    experiences = [
        _experience(1),
        _experience(2, calibrated_probability=0.6, calibration_scope_used="global"),
        _experience(3, conviction=None),
    ]
    _, observations = hier_calibration_inputs(_FakeExpRepo(experiences), enabled=True)
    written: dict[str, tuple[float, str]] = {}
    count = log_calibrated_probabilities(
        experiences, observations, lambda i, p, s: written.__setitem__(i, (p, s))
    )
    # Row 2 already calibrated, row 3 has no conviction -> only row 1 logged.
    assert count == 1 and set(written) == {"exp:live:dc-1"}
    p, scope = written["exp:live:dc-1"]
    assert 0.0 <= p <= 1.0
    assert scope == "global"  # nodes are all below min_n=30 -> identity root
