"""Ladder transition events + weekly risk scenarios (doc 06 §3, §6).

Two tables for two risk mechanisms the design counted as mitigations but that
had no storage, and therefore no way to run:

``ladder_events`` — one row per drawdown-ladder transition. ``ladder_states``
records the daily state; this records the *move*, plus whether it engaged the
kill switch and whether it needs a human acknowledgment. For DD8/DD10/DD15/
DD20 an unacked row is a BLOCKING state: new orders are refused until a human
acks it with a note. Modelled on the unacked-divergent-reconciliation halt
(doc 09 §5) so the halt lives in the database and survives a restart, rather
than in a process-local registry.

``risk_scenarios`` — the weekly Monte Carlo + 7-scenario deterministic stress
results. ``risk/montecarlo.py`` and ``run_stress_suite`` had zero importers
outside their tests, and ``/risk/stress`` read a verdict key nothing wrote.

Idempotent: 0001 runs ``metadata.create_all`` over the whole schema module,
so a fresh database already has both tables and the inspector guards make
this a no-op there.

Revision ID: 0022
Revises: 0021
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None

LADDER_EVENTS = "ladder_events"
LADDER_INDEX = "ix_ladder_events_as_of"
RISK_SCENARIOS = "risk_scenarios"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table(LADDER_EVENTS):
        op.create_table(
            LADDER_EVENTS,
            sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("as_of", sa.Date, nullable=False),
            sa.Column("from_rung", sa.Text, nullable=False),
            sa.Column("to_rung", sa.Text, nullable=False),
            sa.Column("drawdown", sa.Float, nullable=False),
            sa.Column("requires_ack", sa.Boolean, nullable=False, server_default=sa.false()),
            sa.Column(
                "kill_switch_engaged", sa.Boolean, nullable=False, server_default=sa.false()
            ),
            sa.Column("acked_by", sa.Text),
            sa.Column("acked_at", sa.DateTime(timezone=True)),
            sa.Column("ack_note", sa.Text),
        )
    existing = {ix["name"] for ix in sa.inspect(bind).get_indexes(LADDER_EVENTS)}
    if LADDER_INDEX not in existing:
        op.create_index(LADDER_INDEX, LADDER_EVENTS, ["as_of"])

    if not inspector.has_table(RISK_SCENARIOS):
        op.create_table(
            RISK_SCENARIOS,
            sa.Column("as_of", sa.Date, primary_key=True),
            sa.Column("kind", sa.Text, primary_key=True),
            sa.Column("scenario", sa.Text, primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("equity_impact_pct", sa.Float, nullable=False, server_default="0"),
            sa.Column("ladder_implication", sa.Text),
            sa.Column(
                "details",
                JSONB,
                nullable=False,
                server_default="{}",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(RISK_SCENARIOS):
        op.drop_table(RISK_SCENARIOS)
    if inspector.has_table(LADDER_EVENTS):
        op.drop_table(LADDER_EVENTS)
