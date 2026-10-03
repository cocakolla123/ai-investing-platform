"""Migration chain presence (file-level) + gated apply-to-Postgres check."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "infra" / "migrations" / "versions"


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"migration_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_chain_is_linear_through_the_head() -> None:
    """One chain, 0001 -> ... -> head, every step numerically adjacent.

    Stated generically rather than pinned to a revision number. Two agents
    landing a migration in the same session is exactly how a branched history
    happens, and a test that has to be RENAMED to notice is a test that gets
    renamed without being read.
    """
    modules = {m.revision: m for m in (_load(p) for p in sorted(VERSIONS.glob("0*.py")))}
    down = {rev: m.down_revision for rev, m in modules.items()}
    assert down["0001"] is None
    heads = set(down) - {d for d in down.values() if d is not None}
    assert len(heads) == 1, f"expected a single head, got {sorted(heads)}"
    assert heads == {max(modules)}, "the head is not the highest revision"
    parents = [d for d in down.values() if d is not None]
    assert len(parents) == len(set(parents)), "two revisions claim the same parent"
    for rev in sorted(modules):
        if rev != "0001":
            assert down[rev] == f"{int(rev) - 1:04d}"


def test_0021_targets_the_replay_flags_table() -> None:
    """The discovery firewall as a schema fact: replay-scoped rows with a
    deterministic id, so a run can record what its detectors found without
    ever writing the live ``surveillance_flags`` table that feeds production
    triage — and, through triage's promotions, every future replay's
    point-in-time universe."""
    from dynamo.persistence import schema

    assert "replay_flags" in schema.metadata.tables
    table = schema.metadata.tables["replay_flags"]
    assert [c.name for c in table.primary_key.columns] == ["id"]
    assert table.c.run_id.index
    for col in ("run_id", "as_of", "symbol", "detector", "severity", "promoted"):
        assert not table.c[col].nullable

    module = _load(VERSIONS / "0021_replay_flags.py")
    assert module.TABLE == "replay_flags"
    assert module.revision == "0021" and module.down_revision == "0020"


def test_0020_targets_short_interest_with_both_dates() -> None:
    """The point-in-time shape: keyed by (symbol, settlement_date) so a re-run
    upserts, but INDEXED on publication_date because that is the column every
    read bounds on (dynamo.data.finra)."""
    from dynamo.persistence import schema

    table = schema.metadata.tables["short_interest"]
    assert [c.name for c in table.primary_key.columns] == ["symbol", "settlement_date"]
    assert not table.c.publication_date.nullable
    assert table.c.publication_date.index
    assert not table.c.shares_short.nullable
    for col in ("previous_shares_short", "avg_daily_volume", "days_to_cover"):
        assert table.c[col].nullable

    module = _load(VERSIONS / "0020_short_interest.py")
    assert module.TABLE == "short_interest"
    assert module.INDEX == "ix_short_interest_publication_date"


def test_0017_targets_replay_equity_and_fills_tables() -> None:
    from dynamo.persistence import schema

    equity = schema.metadata.tables["replay_equity"]
    assert [c.name for c in equity.primary_key.columns] == ["run_id", "as_of"]
    for col in ("equity", "cash", "gross", "regime", "ladder_rung", "vetoes"):
        assert not equity.c[col].nullable

    fills = schema.metadata.tables["replay_fills"]
    assert [c.name for c in fills.primary_key.columns] == ["run_id", "session", "symbol"]
    for col in ("side", "qty", "price", "trade_value", "cost"):
        assert not fills.c[col].nullable
    assert fills.c.realized_pnl.nullable  # buys carry NULL

    module = _load(VERSIONS / "0017_replay_equity.py")
    assert module.EQUITY_TABLE == "replay_equity"
    assert module.FILLS_TABLE == "replay_fills"


def test_0015_targets_bankroll_context_columns() -> None:
    module = _load(VERSIONS / "0015_bankroll_context.py")
    assert set(module.COLUMN_TYPES) == {"equity_at_decision", "position_notional"}
    from dynamo.persistence import schema

    table = schema.metadata.tables["decision_experience"]
    assert table.c.equity_at_decision.nullable  # NULL on legacy rows
    assert table.c.position_notional.nullable


def test_0016_targets_replay_contexts_table() -> None:
    from dynamo.persistence import schema

    assert "replay_contexts" in schema.metadata.tables
    table = schema.metadata.tables["replay_contexts"]
    assert [c.name for c in table.primary_key.columns] == ["id"]
    assert table.c.run_id.index
    for col in ("run_id", "as_of", "symbol", "stage", "context_sha256", "context_text"):
        assert not table.c[col].nullable


def test_0014_targets_failure_clusters_table() -> None:
    from dynamo.persistence import schema

    assert "failure_clusters" in schema.metadata.tables
    table = schema.metadata.tables["failure_clusters"]
    assert [c.name for c in table.primary_key.columns] == ["id"]
    assert table.c.run_as_of.index
    assert {
        "label",
        "failure_class",
        "size",
        "example_refs",
        "suggested_kind",
        "proposal_id",
    } <= set(table.c.keys())
    assert table.c.proposal_id.nullable  # thin clusters open no proposal
    module = _load(VERSIONS / "0014_failure_clusters.py")
    assert {t.name for t in module.TABLES} == {"failure_clusters"}


def test_0013_targets_replay_verdicts_and_generation_tables() -> None:
    from sqlalchemy import UniqueConstraint

    from dynamo.persistence import schema

    verdicts = schema.metadata.tables["replay_verdicts"]
    assert [c.name for c in verdicts.primary_key.columns] == ["id"]
    assert verdicts.c.run_id.index
    assert {"symbol", "as_of", "action", "conviction", "disagreement_score", "payload"} <= set(
        verdicts.c.keys()
    )

    claims = schema.metadata.tables["evaluation_claims"]
    unique = [
        c
        for c in claims.constraints
        if isinstance(c, UniqueConstraint) and c.name == "uq_evaluation_claims_period_purpose"
    ]
    assert len(unique) == 1
    assert sorted(c.name for c in unique[0].columns) == [
        "period_end",
        "period_start",
        "purpose",
    ]

    generations = schema.metadata.tables["generation_results"]
    assert [c.name for c in generations.primary_key.columns] == ["id"]
    assert {
        "generation",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "model_version",
        "n_train",
        "n_test",
        "brier_meta",
        "brier_control",
        "precision_meta",
        "precision_control",
        "ev_meta",
        "ev_control",
        "value_of_learning",
        "payload",
    } <= set(generations.c.keys())

    module = _load(VERSIONS / "0013_replay_verdicts_generations.py")
    assert {t.name for t in module.TABLES} == {
        "replay_verdicts",
        "evaluation_claims",
        "generation_results",
    }


def test_0012_targets_replay_tables() -> None:
    from dynamo.persistence import schema

    assert "replay_runs" in schema.metadata.tables
    assert "replay_decisions" in schema.metadata.tables
    runs = schema.metadata.tables["replay_runs"]
    assert [c.name for c in runs.primary_key.columns] == ["id"]
    assert {"budget_usd", "spent_usd", "status", "config_json"} <= set(runs.c.keys())
    decisions = schema.metadata.tables["replay_decisions"]
    assert [c.name for c in decisions.primary_key.columns] == ["id"]
    assert decisions.c.run_id.index  # indexed FK-by-convention
    module = _load(VERSIONS / "0012_replay_runs.py")
    assert {t.name for t in module.TABLES} == {"replay_runs", "replay_decisions"}


def test_0011_targets_meta_decision_scores_table() -> None:
    from sqlalchemy import Index

    from dynamo.persistence import schema

    assert "meta_decision_scores" in schema.metadata.tables
    table = schema.metadata.tables["meta_decision_scores"]
    assert [c.name for c in table.primary_key.columns] == ["id"]
    index_names = {str(i.name) for i in table.indexes if isinstance(i, Index)}
    assert "ix_meta_decision_scores_as_of_symbol" in index_names
    assert table.c.source.default is not None  # 'live' default (advisory rows)


def test_0010_targets_calibration_columns() -> None:
    module = _load(VERSIONS / "0010_calibrated_p.py")
    assert set(module.COLUMN_TYPES) == {"calibrated_probability", "calibration_scope_used"}
    from dynamo.persistence import schema

    table = schema.metadata.tables["decision_experience"]
    assert table.c.calibrated_probability.nullable
    assert table.c.calibration_scope_used.nullable


def test_0009_targets_decision_experience_table() -> None:
    from sqlalchemy import Index

    from dynamo.persistence import schema

    assert "decision_experience" in schema.metadata.tables
    table = schema.metadata.tables["decision_experience"]
    assert [c.name for c in table.primary_key.columns] == ["id"]
    indexes = {str(i.name): i for i in table.indexes if isinstance(i, Index)}
    assert {
        "ix_decision_experience_source_resolved_created",
        "ix_decision_experience_symbol_as_of",
        "ix_decision_experience_regime_sector",
        "uq_decision_experience_live_context",
    } <= set(indexes)
    live_unique = indexes["uq_decision_experience_live_context"]
    assert live_unique.unique  # partial: WHERE source='live'
    assert "source = 'live'" in str(live_unique.dialect_options["postgresql"]["where"])


def test_0008_repairs_every_order_table_and_is_data_only() -> None:
    module = _load(VERSIONS / "0008_dc_id_repair.py")
    assert set(module.REPAIR_TABLES) == {"orders", "pm_orders", "options_orders"}
    source = (VERSIONS / "0008_dc_id_repair.py").read_text()
    # Data-only repair: no DDL in either direction.
    assert "create_table" not in source and "drop_table" not in source


def test_0007_targets_sim_ledger_table() -> None:
    from sqlalchemy import UniqueConstraint

    from dynamo.persistence import schema

    assert "pm_sim_ledger" in schema.metadata.tables
    table = schema.metadata.tables["pm_sim_ledger"]
    unique = [
        c
        for c in table.constraints
        if isinstance(c, UniqueConstraint) and c.name == "uq_pm_sim_ledger_movement"
    ]
    assert len(unique) == 1
    assert sorted(c.name for c in unique[0].columns) == ["kind", "ref", "venue"]


def test_0005_targets_cross_venue_table() -> None:
    from dynamo.persistence import schema

    assert "pm_cross_venue_spreads" in schema.metadata.tables
    table = schema.metadata.tables["pm_cross_venue_spreads"]
    assert [c.name for c in table.primary_key.columns] == ["kalshi_market_id", "ts"]


def test_0019_targets_estimate_vintages_table() -> None:
    from dynamo.persistence import schema

    assert "earnings_estimate_vintages" in schema.metadata.tables
    table = schema.metadata.tables["earnings_estimate_vintages"]
    assert [c.name for c in table.primary_key.columns] == [
        "symbol",
        "event_date",
        "fetched_at",
    ]
    module = _load(VERSIONS / "0019_estimate_vintages.py")
    assert module.TABLE == "earnings_estimate_vintages"
    assert module.revision == "0019" and module.down_revision == "0018"


def test_0006_targets_transcripts_table() -> None:
    from dynamo.persistence import schema

    assert "transcripts" in schema.metadata.tables
    table = schema.metadata.tables["transcripts"]
    assert [c.name for c in table.primary_key.columns] == ["symbol", "event_date", "source"]


@pytest.mark.integration
def test_alembic_upgrade_head_creates_table(
    integration_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    # env.py prefers DATABASE_URL over the config option — pin both.
    monkeypatch.setenv("DATABASE_URL", integration_db_url)
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "infra" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", integration_db_url)
    # Reused scratch DBs may sit at head with tables torn down by other
    # fixtures — walk down to base first so the upgrade is exercised for real.
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    engine = create_engine(integration_db_url)
    try:
        inspector = inspect(engine)
        assert inspector.has_table("pm_cross_venue_spreads")
        # 0010: advisory calibration columns present after upgrade head.
        experience_columns = {c["name"] for c in inspector.get_columns("decision_experience")}
        assert {"calibrated_probability", "calibration_scope_used"} <= experience_columns
        # 0012: replay bookkeeping tables present after upgrade head.
        assert inspector.has_table("replay_runs")
        assert inspector.has_table("replay_decisions")
        # 0013: committee-replay verdicts + generational-training tables.
        assert inspector.has_table("replay_verdicts")
        assert inspector.has_table("evaluation_claims")
        assert inspector.has_table("generation_results")
        # 0014: failure-cluster provenance (plan Phase 6).
        assert inspector.has_table("failure_clusters")
        # 0015: bankroll context on decision_experience (campaign Phase 3).
        assert {"equity_at_decision", "position_notional"} <= experience_columns
        # 0023: per-call tracing on cost_ledger (eval Phase 4).
        ledger_columns = {c["name"] for c in inspector.get_columns("cost_ledger")}
        assert {"role", "request_id", "prompt_hash", "retrieved_ids", "latency_ms"} <= (
            ledger_columns
        )
        assert {"attempt", "validation_ok", "equiv_cost_usd"} <= ledger_columns
    finally:
        engine.dispose()


@pytest.mark.integration
def test_ledger_writer_round_trips_a_call_trace(integration_db_url: str) -> None:
    """A traced row lands in every 0023 column; an untraced one leaves them NULL."""
    from datetime import UTC, datetime

    from sqlalchemy import create_engine, text

    from dynamo.llm.cost_meter import CallTrace, LedgerEntry, SqlAlchemyLedgerWriter
    from dynamo.persistence import schema

    engine = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(engine, tables=[schema.cost_ledger], checkfirst=True)
    ts = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
    base = dict(
        model_id="claude-sonnet-5",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cost_usd=0.0,
        batch=False,
    )
    try:
        writer = SqlAlchemyLedgerWriter(engine)
        writer.write(
            LedgerEntry(
                ts=ts,
                call_kind="it-trace:traced",
                trace=CallTrace(
                    role="bull",
                    request_id="run:2026-08-03-MRK-bull",
                    prompt_hash="abcd1234abcd1234",
                    retrieved_ids=("journal-1", "exp-2"),
                    latency_ms=1234.5,
                    attempt=2,
                    validation_ok=False,
                    equiv_cost_usd=0.0123,
                ),
                **base,  # type: ignore[arg-type]
            )
        )
        writer.write(LedgerEntry(ts=ts, call_kind="it-trace:plain", **base))  # type: ignore[arg-type]
        with engine.connect() as cx:
            rows = {
                r.call_kind: r
                for r in cx.execute(
                    text(
                        "select call_kind, role, request_id, prompt_hash, retrieved_ids, "
                        "latency_ms, attempt, validation_ok, equiv_cost_usd from cost_ledger "
                        "where call_kind like 'it-trace:%'"
                    )
                )
            }
        traced, plain = rows["it-trace:traced"], rows["it-trace:plain"]
        assert traced.role == "bull" and traced.request_id == "run:2026-08-03-MRK-bull"
        assert traced.retrieved_ids == ["journal-1", "exp-2"]
        assert traced.latency_ms == 1234.5 and traced.attempt == 2
        assert traced.validation_ok is False and traced.equiv_cost_usd == 0.0123
        assert plain.role is None and plain.retrieved_ids is None and plain.latency_ms is None
    finally:
        with engine.begin() as cx:
            cx.execute(text("delete from cost_ledger where call_kind like 'it-trace:%'"))
        engine.dispose()


@pytest.mark.integration
def test_0008_repair_rewrites_hash_refs_and_reports_orphans(integration_db_url: str) -> None:
    """dc-id repair fixture: order stored the row_hash -> repaired to dc.id;
    a reference matching nothing is counted as an orphan and left alone."""
    from datetime import UTC, datetime

    from sqlalchemy import create_engine, select

    from dynamo.domain.models.portfolio import DecisionContext
    from dynamo.domain.models.research import Sleeve
    from dynamo.persistence import schema
    from dynamo.persistence.repositories.learning import DecisionContextRepository

    module = _load(VERSIONS / "0008_dc_id_repair.py")
    engine = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(
        engine, tables=[schema.decision_context, schema.orders], checkfirst=True
    )
    now = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    try:
        repo = DecisionContextRepository(engine)
        assert (
            repo.append(
                DecisionContext(
                    id="dc-mig-1",
                    created_at=now,
                    env="PAPER",
                    symbol="AAPL",
                    sleeve=Sleeve.LONG_TERM,
                    verdict_id=None,
                    target_portfolio_id="tp-mig",
                    payload={},
                    expected_upside_pct=8.0,
                    expected_downside_pct=4.0,
                    horizon_days=20,
                )
            )
            == "dc-mig-1"
        )
        with engine.begin() as cx:
            row_hash = cx.execute(
                select(schema.decision_context.c.row_hash).where(
                    schema.decision_context.c.id == "dc-mig-1"
                )
            ).scalar_one()
            base = {
                "side": "buy",
                "qty": 1.0,
                "order_type": "marketable_limit",
                "state": "filled",
                "created_at": now,
                "updated_at": now,
            }
            cx.execute(
                schema.orders.insert(),
                [
                    {
                        "client_order_id": "ord-hash-ref",
                        "decision_context_id": row_hash,  # the pre-fix bug
                        "symbol": "AAPL",
                        **base,
                    },
                    {
                        "client_order_id": "ord-orphan",
                        "decision_context_id": "no-such-context",
                        "symbol": "MSFT",
                        **base,
                    },
                ],
            )
        with engine.begin() as cx:
            repaired, unmatched = module.repair_references(cx, "orders")
        assert (repaired, unmatched) == (1, 1)
        with engine.connect() as cx:
            rows = cx.execute(
                select(schema.orders.c.client_order_id, schema.orders.c.decision_context_id)
            ).all()
            refs = {str(r.client_order_id): str(r.decision_context_id) for r in rows}
        assert refs["ord-hash-ref"] == "dc-mig-1"  # repaired to the id get() uses
        assert refs["ord-orphan"] == "no-such-context"  # reported, never touched
    finally:
        schema.metadata.drop_all(
            engine, tables=[schema.decision_context, schema.orders], checkfirst=True
        )
        engine.dispose()


def test_0022_targets_ladder_events_and_risk_scenarios() -> None:
    """The blocking state DD8/DD15 never had, and a home for the weekly
    stress suite that ``/risk/stress`` can actually read (doc 06 §3, §6)."""
    from dynamo.persistence import schema

    events = schema.metadata.tables["ladder_events"]
    assert [c.name for c in events.primary_key.columns] == ["id"]
    assert events.c.as_of.index
    for col in ("as_of", "from_rung", "to_rung", "drawdown", "requires_ack"):
        assert not events.c[col].nullable
    # The ack columns are NULL until a human clears the halt — that NULL IS
    # the halt (mirrors the unacked-reconciliation pattern, doc 09 §5).
    for col in ("acked_by", "acked_at", "ack_note"):
        assert events.c[col].nullable

    scenarios = schema.metadata.tables["risk_scenarios"]
    assert [c.name for c in scenarios.primary_key.columns] == ["as_of", "kind", "scenario"]
    assert not scenarios.c.equity_impact_pct.nullable
    assert scenarios.c.ladder_implication.nullable  # Monte Carlo rows carry none

    module = _load(VERSIONS / "0022_ladder_events_risk_scenarios.py")
    assert module.LADDER_EVENTS == "ladder_events"
    assert module.RISK_SCENARIOS == "risk_scenarios"
    assert module.revision == "0022" and module.down_revision == "0021"
