"""Initial schema: all tables + Timescale hypertables + pgvector.

Revision ID: 0001
Revises: None
"""

from alembic import op

from dynamo.persistence.schema import metadata

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

HYPERTABLES = ("bars_1d", "bars_1m")


def upgrade() -> None:
    bind = op.get_bind()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # timescaledb ships preloaded in the timescaledb-ha image
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    metadata.create_all(bind)
    for table in HYPERTABLES:
        op.execute(
            f"SELECT create_hypertable('{table}', 'ts', if_not_exists => TRUE, "
            "migrate_data => TRUE)"
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS journal_entries_embedding_idx ON journal_entries "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind)
