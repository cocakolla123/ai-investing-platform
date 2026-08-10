"""DuckDB/Parquet research sidecar export. Design: doc 02 §3.

Nightly (02:00 ET, doc 02 §4) export of the canonical ``bars_1d`` view and the
quant/derived tables to partitioned Parquet under ``settings.parquet_dir``.
Postgres remains the single system of record; the sidecar is disposable and
rebuildable — backtests and feature research read it via DuckDB without
hammering the OLTP instance.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from sqlalchemy import Engine, Table, select

from dynamo.persistence import schema
from dynamo.persistence.repositories.market import MarketDataRepository

log = structlog.get_logger(__name__)

# Derived tables exported alongside canonical bars (doc 02 §3: features/scores).
QUANT_TABLES: tuple[Table, ...] = (
    schema.quant_scores,
    schema.quality_scores,
    schema.watchlist,
    schema.signals,
)


def export_bars(
    repo: MarketDataRepository,
    parquet_dir: str | Path,
    *,
    start: date,
    end: date,
) -> Path:
    """Export the canonical bars view partitioned by year under ``bars_1d/``."""
    frame = repo.bars_frame(None, start, end)
    out_dir = Path(parquet_dir) / "bars_1d"
    out_dir.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        log.info("parquet_export_bars_empty")
        return out_dir
    frame = frame.copy()
    frame["year"] = pd.to_datetime(frame["ts"]).dt.year
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(out_dir),
        partition_cols=["year"],
        existing_data_behavior="delete_matching",
    )
    log.info("parquet_export_bars_done", rows=len(frame), path=str(out_dir))
    return out_dir


def export_quant_tables(
    engine: Engine,
    parquet_dir: str | Path,
    *,
    tables: tuple[Table, ...] = QUANT_TABLES,
) -> list[Path]:
    """Export each derived table as a single Parquet file per table."""
    root = Path(parquet_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in tables:
        with engine.connect() as conn:
            frame = pd.read_sql(select(table), conn)
        path = root / f"{table.name}.parquet"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), str(path))
        written.append(path)
        log.info("parquet_export_table_done", table=table.name, rows=len(frame))
    return written


def export_all(
    repo: MarketDataRepository,
    engine: Engine,
    parquet_dir: str | Path,
    *,
    start: date,
    end: date,
) -> None:
    """The 02:00 ET Dagster asset entry point (doc 02 §4)."""
    export_bars(repo, parquet_dir, start=start, end=end)
    export_quant_tables(engine, parquet_dir)
