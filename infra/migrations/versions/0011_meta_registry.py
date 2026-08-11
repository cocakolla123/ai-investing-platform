"""Meta-model registry adoption (plan Phase 3).

Adds ``meta_decision_scores`` — one row per decision scored by the
production-stage advisory meta-model (nightly ``meta_scoring`` asset), with
the calibrated ``effective_probability`` next to the raw verdict. Sizing
consumes these rows ONLY behind ``meta_sizing_enabled`` (default False).

The registry itself (``model_versions`` / ``promotion_records`` /
``trial_counter``) has existed since 0001; this phase merely starts writing
meta candidates to it (``dynamo.cli.meta_train``) and gives the promotion
CLI a DB mode — no registry DDL is needed here.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0011
Revises: 0010
"""

from alembic import op

from dynamo.persistence.schema import meta_decision_scores, metadata

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[meta_decision_scores], checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[meta_decision_scores], checkfirst=True)
