"""The nightly Form 4 asset ingests the universe, not an arbitrary 5%.

``EdgarClient.insider_transactions`` takes the first ``max_insider_filings``
(50) Form 4s of the day out of ~1,000 filed market-wide, unfiltered. That is
an arbitrary slice of an arbitrary slice, almost none of it names the platform
trades — and ``insider_cluster`` needs TWO insiders buying the SAME symbol
inside 14 days, which such a sample could only produce by luck. The asset now
filters on the tracked CIKs, before any ownership XML is downloaded.

No network, no database.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from dagster import build_asset_context
from orchestration.assets.ingestion import insider_tx

from dynamo.domain.models.market import InsiderTx

PARTITION = "2026-08-18"


class _Edgar:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def insider_transactions_in_range(
        self, start: date, end: date, *, ciks: Any = None
    ) -> list[InsiderTx]:
        self.calls.append({"start": start, "end": end, "ciks": ciks})
        return [
            InsiderTx(
                accession_no="0001-26-000001",
                symbol="ACME",
                insider_name="A Director",
                role="Director",
                tx_type="P",
                shares=100.0,
                price=10.0,
                filed_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
            )
        ]

    def insider_transactions(self, since: Any) -> list[InsiderTx]:  # pragma: no cover
        raise AssertionError("the unfiltered market-wide sweep must not be used")


class _Repo:
    def __init__(self) -> None:
        self.rows: list[InsiderTx] = []

    def upsert_insider_tx(self, txs: Any) -> int:
        self.rows = list(txs)
        return len(self.rows)


class _Dynamo:
    def __init__(self, edgar: _Edgar, ciks: dict[str, str]) -> None:
        self._edgar = edgar
        self._ciks = ciks
        self.repo = _Repo()

    def edgar(self) -> Any:
        return self._edgar

    def instrument_ciks(self) -> dict[str, str]:
        return dict(self._ciks)

    def market_repo(self) -> Any:
        return self.repo


def _ctx() -> Any:
    return build_asset_context(partition_key=PARTITION)


def test_the_sweep_is_scoped_to_the_tracked_ciks() -> None:
    edgar = _Edgar()
    dynamo = _Dynamo(edgar, {"ACME": "0000000042", "PEER": "0000000077"})
    result = insider_tx(_ctx(), dynamo)  # type: ignore[arg-type]
    assert edgar.calls[0]["ciks"] is not None
    assert sorted(edgar.calls[0]["ciks"]) == ["0000000042", "0000000077"]
    assert result.metadata == {"rows": 1, "scoped_to_ciks": 2}


def test_the_window_is_yesterday_through_the_partition_day() -> None:
    """Generous overlap on an idempotent upsert: a filing that lands after the
    previous run still gets picked up."""
    edgar = _Edgar()
    insider_tx(_ctx(), _Dynamo(edgar, {"ACME": "0000000042"}))  # type: ignore[arg-type]
    assert edgar.calls[0]["start"] == date(2026, 8, 17)
    assert edgar.calls[0]["end"] == date(2026, 8, 18)


def test_a_cik_less_database_falls_back_to_the_market_and_says_so() -> None:
    """A fresh database should ingest something rather than silently nothing,
    but the operator has to be told the sweep is unscoped."""
    edgar = _Edgar()
    dynamo = _Dynamo(edgar, {})
    result = insider_tx(_ctx(), dynamo)  # type: ignore[arg-type]
    assert edgar.calls[0]["ciks"] is None
    assert result.metadata["scoped_to_ciks"] == 0


def test_edgar_unconfigured_is_a_skip_not_a_failure() -> None:
    class _NoEdgar(_Dynamo):
        def edgar(self) -> Any:
            return None

    result = insider_tx(_ctx(), _NoEdgar(_Edgar(), {}))  # type: ignore[arg-type]
    assert result.metadata["rows"] == 0
    assert result.metadata["skipped"] == "edgar not configured"
