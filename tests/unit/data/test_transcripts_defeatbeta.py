"""defeatbeta transcript provider: guarded import + defensive parsing."""

from __future__ import annotations

from datetime import date

import pandas as pd

import dynamo.data.transcripts_defeatbeta as mod
from dynamo.data.transcripts_defeatbeta import DefeatBetaTranscripts


def test_unavailable_without_extra(monkeypatch) -> None:
    """No defeatbeta-api installed -> available False, empty result, no raise."""
    monkeypatch.setattr(mod, "_load_ticker_factory", lambda: None)
    provider = DefeatBetaTranscripts()
    assert provider.available is False
    assert provider.transcripts(["ACME"], since=date(2026, 8, 1)) == []


def test_bulk_fetch_with_stubbed_ticker_factory() -> None:
    """Exercise the parse path through a stub shaped like defeatbeta-api."""

    class _Holder:
        def get_transcripts_list(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "symbol": "ACME",
                        "fiscal_year": 2026,
                        "fiscal_quarter": 2,
                        "report_date": "2026-08-05",
                    },
                    {  # too old -> filtered by `since`
                        "symbol": "ACME",
                        "fiscal_year": 2026,
                        "fiscal_quarter": 1,
                        "report_date": "2026-05-01",
                    },
                ]
            )

        def get_transcript(self, fiscal_year: int, fiscal_quarter: int) -> pd.DataFrame:
            assert (fiscal_year, fiscal_quarter) == (2026, 2)
            return pd.DataFrame(
                [
                    {"paragraph_number": 1, "speaker": "Jane Roe", "content": "Welcome."},
                    {"paragraph_number": 2, "speaker": "John Doe", "content": "Margins up."},
                    {"paragraph_number": 3, "speaker": "Jane Roe", "content": ""},
                ]
            )

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def earning_call_transcripts(self) -> _Holder:
            return _Holder()

    provider = DefeatBetaTranscripts(ticker_factory=_Ticker)
    assert provider.available is True
    out = provider.transcripts(["ACME"], since=date(2026, 8, 1))
    assert len(out) == 1
    t = out[0]
    assert t.symbol == "ACME"
    assert t.event_date == date(2026, 8, 5)
    assert t.source == "defeatbeta"
    assert t.period == "2026Q2"
    assert t.content == "Jane Roe: Welcome.\n\nJohn Doe: Margins up."
    assert t.speakers == (
        {"speaker": "Jane Roe", "title": ""},
        {"speaker": "John Doe", "title": ""},
    )


def test_symbol_failure_degrades_to_empty() -> None:
    class _Ticker:
        def __init__(self, symbol: str) -> None:
            raise RuntimeError("dataset unavailable")

    provider = DefeatBetaTranscripts(ticker_factory=_Ticker)
    assert provider.transcripts(["ACME"], since=date(2026, 8, 1)) == []
