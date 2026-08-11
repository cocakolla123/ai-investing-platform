"""Historical replay foundation (plan Phase 4).

Adds the two replay bookkeeping tables:

- ``replay_runs`` — one row per manually-triggered replay run (quant-only in
  Phase 4; committee mode with ``budget_usd``/``spent_usd`` accounting lands
  in Phase 5).
- ``replay_decisions`` — one row per (run, session, symbol) synthesized
  decision, with the target weight and next-open fill price.

FIREWALL: these tables plus ``decision_experience`` rows with
``source='replay'`` are the ONLY tables replay writes; live tables (orders,
trades, decision_context, committee_verdicts, journal_entries) are never
touched by a replay run.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these tables — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get them here.

Revision ID: 0012
Revises: 0011
"""

from alembic import op

from dynamo.persistence.schema import metadata, replay_decisions, replay_runs

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

TABLES = [replay_runs, replay_decisions]


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=TABLES, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=TABLES, checkfirst=True)
