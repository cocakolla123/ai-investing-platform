"""Engine and connection helpers."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from dynamo.config.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True, future=True)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)
