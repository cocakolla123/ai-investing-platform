"""Failure-cluster provenance (plan Phase 6).

Adds ``failure_clusters`` — one row per cluster per clustering run. A table
(rather than more ``metrics_daily`` rows) is genuinely needed here because
clusters carry text/structured payloads (label, example refs, suggested
proposal kind) and the dashboard must browse ALL of the latest run's clusters,
including the thin ones below the proposal threshold that never reach
``improvement_proposals``. The durable sensor watermark itself stays in
``metrics_daily`` (metric ``learning.cluster_watermark``) — no table needed
for a single float.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0014
Revises: 0013
"""

from alembic import op

from dynamo.persistence.schema import failure_clusters, metadata

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

TABLES = [failure_clusters]


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=TABLES, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=TABLES, checkfirst=True)
