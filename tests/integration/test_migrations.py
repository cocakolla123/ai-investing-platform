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


def test_revision_chain_is_linear_through_0014() -> None:
    modules = {m.revision: m for m in (_load(p) for p in sorted(VERSIONS.glob("0*.py")))}
    assert "0014" in modules, "0014_failure_clusters migration missing"
    # Single linear chain 0001 -> ... -> head.
    down = {rev: m.down_revision for rev, m in modules.items()}
    assert down["0001"] is None
    heads = set(down) - {d for d in down.values() if d is not None}
    assert heads == {"0014"}, f"expected 0014 as the single head, got {heads}"
    assert down["0005"] == "0004"
    assert down["0006"] == "0005"
    assert down["0007"] == "0006"
    assert down["0008"] == "0007"
    assert down["0009"] == "0008"
    assert down["0010"] == "0009"
    assert down["0011"] == "0010"
    assert down["0012"] == "0011"
    assert down["0013"] == "0012"
    assert down["0014"] == "0013"


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
    finally:
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
