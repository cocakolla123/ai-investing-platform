"""DecisionExperience learning dataset (plan Phase 1).

Adds ``decision_experience`` — one row per decision → outcome, with quality
separation (forecast / decision / outcome) and counterfactual columns. The
``source`` column ('live' | 'replay') plus ``replay_run_id`` form the replay
firewall: live learning consumers filter ``source='live'`` from day one. The
partial unique index on ``decision_context_id WHERE source='live'`` makes
live inserts idempotent per decision context.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0009
Revises: 0008
"""

from alembic import op

from dynamo.persistence.schema import decision_experience, metadata

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[decision_experience], checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[decision_experience], checkfirst=True)
