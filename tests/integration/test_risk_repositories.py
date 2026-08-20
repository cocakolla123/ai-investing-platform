"""Ladder events, risk scenarios and the VaR backtest reader, against Postgres.

These are the storage halves of the four doc 06 mechanisms that previously
had no storage at all. Skipped unless ``INTEGRATION_DB_URL`` is set.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import Engine, create_engine, delete

from dynamo.persistence import schema
from dynamo.persistence.repositories.risk import (
    LadderEventRepository,
    RiskScenarioRepository,
    VarBacktestReader,
)
from dynamo.risk.var import VarTrafficLight, backtest_var

pytestmark = pytest.mark.integration

TABLES = [
    schema.ladder_events,
    schema.risk_scenarios,
    schema.risk_verdicts,
    schema.equity_curve,
]
AS_OF = date(2026, 8, 20)


@pytest.fixture()
def engine(integration_db_url: str) -> Iterator[Engine]:
    eng = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(eng, tables=TABLES)
    with eng.begin() as conn:
        for t in TABLES:
            conn.execute(delete(t))
    yield eng
    with eng.begin() as conn:
        for t in TABLES:
            conn.execute(delete(t))
    eng.dispose()


# --- ladder events -------------------------------------------------------


def _record(repo: LadderEventRepository, *, to: str, ack: bool, kill: bool = False) -> int:
    return repo.record(
        as_of=AS_OF, from_rung="normal", to_rung=to, drawdown=-0.2,
        requires_ack=ack, kill_switch_engaged=kill,
    )


def test_an_ack_requiring_event_halts_until_acknowledged(engine: Engine) -> None:
    repo = LadderEventRepository(engine)
    assert repo.is_halted() is False

    event_id = _record(repo, to="dd15", ack=True)
    assert repo.is_halted() is True
    assert [e["id"] for e in repo.unacked_blocking()] == [event_id]

    assert repo.acknowledge(event_id, "owner", "reviewed the drawdown") is True
    assert repo.is_halted() is False


def test_a_non_ack_event_never_halts(engine: Engine) -> None:
    repo = LadderEventRepository(engine)
    _record(repo, to="dd5", ack=False)
    assert repo.is_halted() is False


def test_acknowledging_twice_is_a_no_op(engine: Engine) -> None:
    """The ack is idempotent; the second call must not overwrite the first note."""
    repo = LadderEventRepository(engine)
    event_id = _record(repo, to="dd8", ack=True)
    assert repo.acknowledge(event_id, "owner", "first") is True
    assert repo.acknowledge(event_id, "someone-else", "second") is False
    row = next(e for e in repo.recent() if e["id"] == event_id)
    assert row["ack_note"] == "first" and row["acked_by"] == "owner"


def test_acknowledging_one_rung_does_not_clear_another(engine: Engine) -> None:
    repo = LadderEventRepository(engine)
    first = _record(repo, to="dd8", ack=True)
    _record(repo, to="dd15", ack=True)
    repo.acknowledge(first, "owner", "reviewed")
    assert repo.is_halted() is True  # the deeper rung is still pending


def test_the_dd20_row_records_that_the_switch_was_engaged(engine: Engine) -> None:
    repo = LadderEventRepository(engine)
    event_id = _record(repo, to="dd20", ack=True, kill=True)
    row = next(e for e in repo.recent() if e["id"] == event_id)
    assert row["kill_switch_engaged"] is True


# --- risk scenarios ------------------------------------------------------


def _rows(as_of: date, impact: float = -0.31) -> list[dict[str, object]]:
    return [
        {
            "as_of": as_of, "kind": "stress", "scenario": "2008_q4",
            "equity_impact_pct": impact, "ladder_implication": "dd20", "details": {},
        },
        {
            "as_of": as_of, "kind": "monte_carlo", "scenario": "20d_student_t_copula",
            "equity_impact_pct": -0.08, "ladder_implication": None,
            "details": {"n_paths": 10000},
        },
    ]


def test_scenarios_round_trip(engine: Engine) -> None:
    repo = RiskScenarioRepository(engine)
    assert repo.upsert(_rows(AS_OF)) == 2
    latest = repo.latest()
    assert {r["kind"] for r in latest} == {"stress", "monte_carlo"}
    assert repo.latest_as_of() == AS_OF


def test_rerunning_the_same_week_updates_rather_than_duplicates(engine: Engine) -> None:
    repo = RiskScenarioRepository(engine)
    repo.upsert(_rows(AS_OF))
    repo.upsert(_rows(AS_OF, impact=-0.42))
    latest = repo.latest()
    assert len(latest) == 2
    stress = next(r for r in latest if r["kind"] == "stress")
    assert stress["equity_impact_pct"] == pytest.approx(-0.42)


def test_only_the_newest_week_is_returned(engine: Engine) -> None:
    repo = RiskScenarioRepository(engine)
    repo.upsert(_rows(AS_OF - timedelta(days=7)))
    repo.upsert(_rows(AS_OF))
    assert all(r["as_of"] == AS_OF for r in repo.latest())


def test_no_scenarios_yet_reads_as_empty(engine: Engine) -> None:
    repo = RiskScenarioRepository(engine)
    assert repo.latest() == [] and repo.latest_as_of() is None


# --- VaR backtest reader -------------------------------------------------


def _seed_var_history(
    engine: Engine, *, sessions: int, breach_every: int | None = None
) -> int:
    """One verdict + one equity point per session; equity drifts -0.1%/day
    except on breach days, which lose 9% against a 2% VaR forecast.

    Returns the number of breaching *returns* — the last session's move has
    no following equity point, so it produces no observation.
    """
    verdicts = []
    curve = []
    equity = 100_000.0
    start = AS_OF - timedelta(days=sessions)
    breaches = 0
    for i in range(sessions + 1):
        day = start + timedelta(days=i)
        curve.append({"as_of": day, "equity": equity, "cash": 0.0, "gross_exposure": 1.0})
        verdicts.append(
            {
                "id": f"risk-{day.isoformat()}",
                "as_of": datetime(day.year, day.month, day.day, 21, tzinfo=UTC),
                "payload": {"metrics": {"var95_1d": 0.02}},
            }
        )
        breach = breach_every is not None and i > 0 and i % breach_every == 0
        if breach and i < sessions:
            breaches += 1
        equity *= 0.91 if breach else 0.999
    with engine.begin() as conn:
        conn.execute(schema.equity_curve.insert().values(curve))
        conn.execute(schema.risk_verdicts.insert().values(verdicts))
    return breaches


def test_a_quiet_year_produces_no_exceptions(engine: Engine) -> None:
    _seed_var_history(engine, sessions=120)
    observations = VarBacktestReader(engine).observations(today=AS_OF)
    assert len(observations) == 120
    result = backtest_var(observations)
    assert result.exceptions == 0
    assert result.light is VarTrafficLight.GREEN


def test_realized_breaches_are_counted_and_tighten_caps(engine: Engine) -> None:
    """The end-to-end claim: stored VaR + realized equity -> a tighter cap."""
    expected = _seed_var_history(engine, sessions=120, breach_every=4)
    result = backtest_var(VarBacktestReader(engine).observations(today=AS_OF))
    assert expected == 29
    assert result.exceptions == expected
    assert result.light is VarTrafficLight.RED
    assert result.tighten_factor == 0.75


def test_an_empty_database_yields_no_observations(engine: Engine) -> None:
    assert VarBacktestReader(engine).observations(today=AS_OF) == []
    # ...and that must not tighten anything.
    assert backtest_var([]).tighten_factor == 1.0


def test_sessions_without_a_stored_var_are_not_paired(engine: Engine) -> None:
    """Equity moves on a day the risk engine never ran are not evidence."""
    _seed_var_history(engine, sessions=30)
    with engine.begin() as conn:
        conn.execute(delete(schema.risk_verdicts))
    assert VarBacktestReader(engine).observations(today=AS_OF) == []
