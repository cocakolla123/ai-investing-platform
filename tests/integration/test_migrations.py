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


def test_revision_chain_is_linear_through_0006() -> None:
    modules = {m.revision: m for m in (_load(p) for p in sorted(VERSIONS.glob("0*.py")))}
    assert "0006" in modules, "0006_transcripts migration missing"
    # Single linear chain 0001 -> ... -> head.
    down = {rev: m.down_revision for rev, m in modules.items()}
    assert down["0001"] is None
    heads = set(down) - {d for d in down.values() if d is not None}
    assert heads == {"0006"}, f"expected 0006 as the single head, got {heads}"
    assert down["0005"] == "0004"
    assert down["0006"] == "0005"


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
    command.upgrade(cfg, "head")

    engine = create_engine(integration_db_url)
    try:
        assert inspect(engine).has_table("pm_cross_venue_spreads")
    finally:
        engine.dispose()
