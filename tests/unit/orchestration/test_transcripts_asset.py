"""transcripts asset: AV-first, defeatbeta fallback, caps, graceful skips."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from dagster import MaterializeResult, build_asset_context
from orchestration.assets.ingestion import transcripts as transcripts_asset

from dynamo.domain.models.market import Transcript
from tests.unit.orchestration.conftest import FakeDynamo, FakeRepo

TODAY = datetime.now(UTC).date()


def _transcript(symbol: str, source: str) -> Transcript:
    return Transcript(
        symbol=symbol,
        event_date=TODAY,
        source=source,
        period="2026Q2",
        content=f"{symbol} call content",
    )


def _ctx():
    return build_asset_context(partition_key=TODAY.isoformat())


def test_skips_when_no_source_configured(fake_dynamo: FakeDynamo) -> None:
    result = cast(MaterializeResult, transcripts_asset(_ctx(), fake_dynamo))
    assert result.metadata is not None
    assert result.metadata["skipped"] == "no transcript source configured"


def test_av_first_then_defeatbeta_fallback(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo
) -> None:
    # AAA and BBB report today; AV only has AAA, defeatbeta backfills BBB.
    fake_dynamo.st.reporters = [
        {"symbol": "AAA", "event_date": TODAY, "period": "2026Q2", "call_transcript_ref": None},
        {"symbol": "BBB", "event_date": TODAY, "period": None, "call_transcript_ref": None},
        {  # not in the universe -> ignored
            "symbol": "ZZZ",
            "event_date": TODAY,
            "period": None,
            "call_transcript_ref": None,
        },
    ]
    fake_dynamo.st.av_transcripts = {"AAA": _transcript("AAA", "alphavantage")}
    fake_dynamo.st.defeatbeta_transcripts = [_transcript("BBB", "defeatbeta")]

    result = cast(MaterializeResult, transcripts_asset(_ctx(), fake_dynamo))
    assert result.metadata is not None
    assert result.metadata["reporters"] == 2
    assert result.metadata["alphavantage"] == 1
    assert result.metadata["defeatbeta"] == 1
    assert result.metadata["rows"] == 2

    persisted = dict(fake_repo.inserted_rows)["transcripts"]
    assert {t.symbol for t in persisted} == {"AAA", "BBB"}
    # earnings_events rows were pointed at the stored transcripts
    assert ("AAA", TODAY, "transcripts:AAA:" + TODAY.isoformat() + ":alphavantage") in (
        fake_repo.transcript_links
    )
    assert ("BBB", TODAY, "transcripts:BBB:" + TODAY.isoformat() + ":defeatbeta") in (
        fake_repo.transcript_links
    )


def test_daily_cap_limits_reporters(fake_dynamo: FakeDynamo, fake_repo: FakeRepo) -> None:
    fake_dynamo.st.reporters = [
        {"symbol": "AAA", "event_date": TODAY, "period": None, "call_transcript_ref": None},
        {
            "symbol": "BBB",
            "event_date": TODAY - timedelta(days=1),
            "period": None,
            "call_transcript_ref": None,
        },
    ]
    fake_dynamo.st.av_transcripts = {
        "AAA": _transcript("AAA", "alphavantage"),
        "BBB": _transcript("BBB", "alphavantage"),
    }

    class CappedDynamo(FakeDynamo):
        def settings(self):  # type: ignore[override]
            s = super().settings()
            object.__setattr__(s, "transcript_daily_cap", 1)
            return s

    capped = CappedDynamo(state_key=fake_dynamo.state_key)
    result = cast(MaterializeResult, transcripts_asset(_ctx(), capped))
    assert result.metadata is not None
    assert result.metadata["reporters"] == 1
    assert result.metadata["rows"] == 1


def test_no_reporters_is_zero_rows(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.av_transcripts = {}
    result = cast(MaterializeResult, transcripts_asset(_ctx(), fake_dynamo))
    assert result.metadata is not None
    assert result.metadata["rows"] == 0
    assert result.metadata["reporters"] == 0
