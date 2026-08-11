"""Committee replay + generational training (plan Phase 5).

Adds three tables:

- ``replay_verdicts`` — committee-replay verdicts (same shape as
  ``committee_verdicts`` + ``run_id``). Replay verdicts NEVER land in
  ``committee_verdicts``: the firewall extension keeps live readers blind to
  replay output by construction.
- ``evaluation_claims`` — durable evaluation-period claims (the
  database-backed sibling of the in-memory ``EvaluationRegistry``): a
  (period_start, period_end, purpose) is claimed once, refused thereafter.
- ``generation_results`` — one row per training generation: meta-model vs
  frozen control (raw conviction, identity calibration) on identical test
  decisions; ``value_of_learning = brier_control - brier_meta``.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these tables — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get them here.

Revision ID: 0013
Revises: 0012
"""

from alembic import op

from dynamo.persistence.schema import (
    evaluation_claims,
    generation_results,
    metadata,
    replay_verdicts,
)

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

TABLES = [replay_verdicts, evaluation_claims, generation_results]


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=TABLES, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=TABLES, checkfirst=True)
