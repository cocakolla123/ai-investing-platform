"""Bind-parameter chunking — pure arithmetic, no database.

Lives outside test_replay_repository.py because that module is
INTEGRATION_DB_URL-gated and this must run everywhere.
"""

from __future__ import annotations


def test_row_chunks_respect_the_postgres_bind_parameter_cap() -> None:
    """The wide campaign died on its FIRST session: ~10,400 decision rows x 9
    columns = ~94k bind parameters against the wire protocol's 65,535 cap.
    Every bulk upsert now goes through _row_chunks; this pins the arithmetic."""
    from dynamo.persistence.repositories.replay import _BIND_PARAM_BUDGET, _row_chunks

    rows = [{f"c{i}": i for i in range(9)} for _ in range(10_400)]
    chunks = _row_chunks(rows)
    assert sum(len(c) for c in chunks) == len(rows), "no row lost or duplicated"
    assert len(chunks) > 1, "10,400 x 9 cannot fit one statement"
    for chunk in chunks:
        assert len(chunk) * 9 <= _BIND_PARAM_BUDGET

    assert _row_chunks([]) == []
    single = _row_chunks([{"a": 1}])
    assert single == [[{"a": 1}]]
