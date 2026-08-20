"""Event-table backfills — fake adapters, fake repo. No network, no database.

``insider_tx``, ``earnings_events`` and ``filings`` were all 0 rows, which is
why ``insider_cluster``, ``earnings_surprise`` and ``filing_event`` could
never fire. What these tests pin is the part a live run cannot re-check
cheaply: that a second sweep over the same window converges instead of
duplicating, that the universe filter reaches the adapter BEFORE it starts
downloading ownership XML, and that a truncated Finnhub window is reported as
a failure rather than passing for a complete one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from dynamo.cli import backfill_events as be
from dynamo.domain.models.market import EarningsEvent, Filing, InsiderTx

START = date(2026, 6, 1)  # a Monday
END = date(2026, 6, 3)


class FakeRepo:
    """Natural-key upserts, exactly like the real tables."""

    def __init__(self) -> None:
        self.insider: dict[tuple[str, str, str, str], InsiderTx] = {}
        self.filings: dict[str, Filing] = {}
        self.earnings: dict[tuple[str, date], EarningsEvent] = {}
        self.vintages = 0

    def upsert_insider_tx(self, txs: Any) -> int:
        txs = list(txs)
        for t in txs:
            self.insider[(t.accession_no, t.symbol, t.insider_name, t.tx_type)] = t
        return len(txs)

    def upsert_filings(self, filings: Any) -> int:
        filings = list(filings)
        for f in filings:
            self.filings[f.accession_no] = f
        return len(filings)

    def upsert_earnings(self, events: Any) -> int:
        events = list(events)
        for e in events:
            self.earnings[(e.symbol, e.event_date)] = e
        return len(events)

    def append_estimate_vintages(self, events: Any) -> int:
        self.vintages += len(list(events))
        return len(list(events))


class FakeEdgar:
    def __init__(self, *, by_day: dict[date, list[Any]] | None = None) -> None:
        self.by_day = by_day or {}
        self.cik_filters: list[Any] = []
        self.form_calls: list[tuple[str, date]] = []

    def insider_transactions_in_range(
        self, start: date, end: date, *, ciks: Any = None
    ) -> list[InsiderTx]:
        self.cik_filters.append(None if ciks is None else sorted(ciks))
        return list(self.by_day.get(start, []))

    def filings_in_range(self, form: str, start: date, end: date) -> list[Filing]:
        self.form_calls.append((form, start))
        return [f for f in self.by_day.get(start, []) if f.form_type == form]


class FakeFinnhub:
    def __init__(self, by_window: dict[tuple[date, date], list[EarningsEvent]]) -> None:
        self.by_window = by_window
        self.windows: list[tuple[date, date]] = []

    def earnings_calendar(self, start: date, end: date) -> list[EarningsEvent]:
        self.windows.append((start, end))
        return list(self.by_window.get((start, end), []))


def _tx(accession: str, symbol: str, insider: str, day: date) -> InsiderTx:
    return InsiderTx(
        accession_no=accession,
        symbol=symbol,
        insider_name=insider,
        role="Director",
        tx_type="P",
        shares=1_000.0,
        price=120.0,
        filed_at=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
    )


def _filing(accession: str, form: str, day: date, symbol: str | None) -> Filing:
    return Filing(
        accession_no=accession,
        cik="0000000042",
        symbol=symbol,
        form_type=form,
        filed_at=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
        url="https://example.invalid/filing",
        items=("1.01",),
    )


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    repo = FakeRepo()
    edgar = FakeEdgar()
    finnhub = FakeFinnhub({})
    universe = tmp_path / "universe.txt"
    universe.write_text("# a comment\nACME\npeer  # trailing comment\n\nACME\n")

    monkeypatch.setattr(be, "_repo", lambda _url: (object(), repo))
    monkeypatch.setattr(be, "_settings", lambda: _Settings())
    monkeypatch.setattr(be, "_edgar", lambda _s: edgar)
    monkeypatch.setattr(be, "_finnhub", lambda _s: finnhub)
    monkeypatch.setattr(
        be, "cik_map", lambda _e, symbols=None: {"ACME": "0000000042", "PEER": "0000000077"}
    )
    monkeypatch.setattr(
        be,
        "table_counts",
        lambda _e: {
            "insider_tx": len(repo.insider),
            "filings": len(repo.filings),
            "earnings_events": len(repo.earnings),
        },
    )
    return {"repo": repo, "edgar": edgar, "finnhub": finnhub, "universe": universe}


class _Settings:
    edgar_user_agent = "test"
    finnhub_api_key = "test-key"
    database_url = ""


def _args(*extra: str) -> list[str]:
    return [*extra, "--start", START.isoformat(), "--end", END.isoformat()]


# --- symbol file ------------------------------------------------------------


def test_symbol_file_skips_comments_and_normalizes(wired: dict[str, Any]) -> None:
    assert be.read_symbol_file(wired["universe"]) == ["ACME", "PEER"]


# --- insider ----------------------------------------------------------------


def test_insider_upserts_are_idempotent(wired: dict[str, Any]) -> None:
    """The natural key is (accession, symbol, insider, tx_type): sweeping the
    same window twice converges, it does not double the table."""
    edgar: FakeEdgar = wired["edgar"]
    edgar.by_day = {
        START: [_tx("0001-26-000001", "ACME", "A Director", START)],
        END: [_tx("0001-26-000002", "ACME", "B Officer", END)],
    }
    argv = ["insider", "--universe-file", str(wired["universe"]), *_args()]
    assert be.main(argv) == 0
    first = dict(wired["repo"].insider)
    assert len(first) == 2
    assert be.main(argv) == 0
    assert wired["repo"].insider == first


def test_insider_filters_ciks_before_any_xml_is_fetched(wired: dict[str, Any]) -> None:
    """The whole point of the filter: a market-wide Form 4 day is ~1,000
    ownership documents, and the ones outside the universe must never be
    downloaded at all."""
    be.main(["insider", "--universe-file", str(wired["universe"]), *_args()])
    assert wired["edgar"].cik_filters
    for passed in wired["edgar"].cik_filters:
        assert passed == ["0000000042", "0000000077"]


def test_all_symbols_lifts_the_filter(wired: dict[str, Any]) -> None:
    be.main(["insider", "--all-symbols", *_args()])
    assert wired["edgar"].cik_filters == [None] * len(wired["edgar"].cik_filters)


def test_insider_skips_weekends(wired: dict[str, Any]) -> None:
    """EDGAR accepts no filings at the weekend; a request per Saturday is
    pure rate-limit budget spent on a guaranteed empty response."""
    be.main(
        [
            "insider",
            "--universe-file",
            str(wired["universe"]),
            "--start",
            "2026-06-05",  # Friday
            "--end",
            "2026-06-08",  # Monday
        ]
    )
    assert len(wired["edgar"].cik_filters) == 2


def test_insider_refuses_when_no_symbol_carries_a_cik(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A CIK-less universe would silently sweep nothing and report success."""
    monkeypatch.setattr(be, "cik_map", lambda _e, symbols=None: {})
    argv = ["insider", "--universe-file", str(wired["universe"]), *_args()]
    assert be.main(argv) == 1
    assert wired["repo"].insider == {}


# --- filings ----------------------------------------------------------------


def test_filings_upserts_are_idempotent_and_cover_each_form(
    wired: dict[str, Any],
) -> None:
    wired["edgar"].by_day = {
        START: [
            _filing("0002-26-000001", "8-K", START, "ACME"),
            _filing("0002-26-000002", "S-1", START, "ACME"),
        ]
    }
    argv = ["filings", "--forms", "8-K", "S-1", *_args()]
    assert be.main(argv) == 0
    assert set(wired["repo"].filings) == {"0002-26-000001", "0002-26-000002"}
    assert be.main(argv) == 0
    assert len(wired["repo"].filings) == 2
    assert ("8-K", START) in wired["edgar"].form_calls


def test_filings_fill_a_missing_symbol_from_instruments(wired: dict[str, Any]) -> None:
    """``filing_event`` skips a row with no symbol outright, and the full-text
    index names one only when a display_name carries it."""
    wired["edgar"].by_day = {START: [_filing("0002-26-000003", "8-K", START, None)]}
    assert be.main(["filings", "--forms", "8-K", *_args()]) == 0
    assert wired["repo"].filings["0002-26-000003"].symbol == "ACME"


# --- earnings ---------------------------------------------------------------


def _event(symbol: str, day: date, *, act: float | None = None) -> EarningsEvent:
    return EarningsEvent(symbol=symbol, event_date=day, period="2026Q2", eps_est=1.0, eps_act=act)


def test_earnings_chunks_and_upserts_idempotently(wired: dict[str, Any]) -> None:
    lo = be.FINNHUB_CALENDAR_FLOOR
    hi = lo + timedelta(days=3)
    finnhub: FakeFinnhub = wired["finnhub"]
    finnhub.by_window = {
        (lo, lo + timedelta(days=1)): [_event("ACME", lo, act=1.4)],
        (lo + timedelta(days=2), hi): [_event("PEER", hi)],
    }
    argv = ["earnings", "--chunk-days", "2", "--start", lo.isoformat(), "--end", hi.isoformat()]
    assert be.main(argv) == 0
    assert set(wired["repo"].earnings) == {("ACME", lo), ("PEER", hi)}
    assert be.main(argv) == 0
    assert len(wired["repo"].earnings) == 2


def test_earnings_clamps_below_the_free_tier_floor(wired: dict[str, Any]) -> None:
    """Finnhub's free calendar returns nothing before the floor; asking for
    2024 would spend requests on guaranteed-empty windows."""
    be.main(["earnings", "--start", "2024-01-01", "--end", "2026-07-02"])
    assert wired["finnhub"].windows
    assert min(lo for lo, _ in wired["finnhub"].windows) == be.FINNHUB_CALENDAR_FLOOR


def test_a_truncated_window_is_a_failure_not_a_silent_gap(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Finnhub caps a response and does not say so. A truncated window looks
    identical to a complete one in the table, so the sweep has to shout."""
    day = be.FINNHUB_CALENDAR_FLOOR
    wired["finnhub"].by_window = {
        (day, day): [
            _event(f"SYM{i}", day + timedelta(days=i % 5)) for i in range(be.FINNHUB_RESPONSE_CAP)
        ]
    }
    exit_code = be.main(
        ["earnings", "--chunk-days", "1", "--start", day.isoformat(), "--end", day.isoformat()]
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "INCOMPLETE" in err and "--chunk-days" in err


def test_earnings_refuses_without_a_finnhub_key(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    class _NoKey(_Settings):
        finnhub_api_key = ""

    monkeypatch.setattr(be, "_settings", lambda: _NoKey())
    assert be.main(["earnings"]) == 1
    assert wired["repo"].earnings == {}


# --- transient provider failures --------------------------------------------


class _FlakyEdgar(FakeEdgar):
    """Fails on ``bad`` and serves everything else — EDGAR's actual behaviour
    over a multi-hour sweep."""

    def __init__(self, bad: set[date], payload: dict[date, list[Any]]) -> None:
        super().__init__(by_day=payload)
        self.bad = bad

    def _guard(self, day: date) -> None:
        from dynamo.data.errors import ProviderDown

        if day in self.bad:
            raise ProviderDown("edgar", "500 from the full-text index")

    def insider_transactions_in_range(
        self, start: date, end: date, *, ciks: Any = None
    ) -> list[InsiderTx]:
        self._guard(start)
        return super().insider_transactions_in_range(start, end, ciks=ciks)

    def filings_in_range(self, form: str, start: date, end: date) -> list[Filing]:
        self._guard(start)
        return super().filings_in_range(form, start, end)


def test_a_provider_500_skips_the_day_and_keeps_going(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Aborting throws away the hours already spent; swallowing the error
    leaves a hole that looks exactly like a quiet day. Neither is acceptable:
    continue, then exit non-zero naming the days to re-run."""
    good = _tx("0001-26-000009", "ACME", "A Director", END)
    edgar = _FlakyEdgar(bad={START}, payload={END: [good]})
    monkeypatch.setattr(be, "_edgar", lambda _s: edgar)

    argv = ["insider", "--universe-file", str(wired["universe"]), *_args()]
    assert be.main(argv) == 1
    # The healthy day still landed.
    assert len(wired["repo"].insider) == 1
    out, err = capsys.readouterr()
    assert "SKIPPED" in out
    assert "INCOMPLETE" in err and START.isoformat() in err


def test_filings_sweep_survives_a_provider_500_too(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    edgar = _FlakyEdgar(bad={START}, payload={END: [_filing("0002-26-000009", "8-K", END, "ACME")]})
    monkeypatch.setattr(be, "_edgar", lambda _s: edgar)
    assert be.main(["filings", "--forms", "8-K", *_args()]) == 1
    assert set(wired["repo"].filings) == {"0002-26-000009"}


def test_a_clean_sweep_still_exits_zero(wired: dict[str, Any]) -> None:
    wired["edgar"].by_day = {START: [_tx("0001-26-000010", "ACME", "A Director", START)]}
    argv = ["insider", "--universe-file", str(wired["universe"]), *_args()]
    assert be.main(argv) == 0


def test_the_index_root_form_is_stored_as_the_sec_form_code(
    wired: dict[str, Any],
) -> None:
    """EDGAR's full-text index calls it "SCHEDULE 13D"; the detector's
    MATERIAL_FORMS says "SC 13D", and startswith("SC 13D") is False on the
    index's spelling. Storing the index name would produce rows
    ``filing_event`` reads and silently ignores."""
    wired["edgar"].by_day = {START: [_filing("0002-26-000011", "SCHEDULE 13D", START, "ACME")]}
    assert be.main(["filings", "--forms", "SCHEDULE 13D", *_args()]) == 0
    assert wired["repo"].filings["0002-26-000011"].form_type == "SC 13D"


def test_the_stored_form_codes_are_the_ones_the_detector_matches() -> None:
    """Belt and braces on the mapping itself, against the real constant."""
    from dynamo.data.detectors import MATERIAL_FORMS

    for form in be.DEFAULT_FILING_FORMS:
        stored = be.normalized_form_type(form).upper()
        if stored.startswith("8-K"):
            continue  # 8-K is material via its item codes, not its form
        assert any(stored.startswith(m) for m in MATERIAL_FORMS), stored


def test_an_unmapped_form_is_stored_verbatim(wired: dict[str, Any]) -> None:
    assert be.normalized_form_type("8-K") == "8-K"
    assert be.normalized_form_type("S-1") == "S-1"
