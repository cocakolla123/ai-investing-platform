"""Market-wide CIK/sector seeding CLI — fake repo + fake SEC adapters.

No network and no database: the point of these tests is the *scope* and the
*additivity* of the writes, which is exactly where the old path (``backfill
fundamentals --seed-ciks``, active universe only) fell short.
"""

from __future__ import annotations

from typing import Any

import pytest

from dynamo.cli import reference


class FakeRepo:
    def __init__(self) -> None:
        self.ciks: dict[str, str] = {}
        self.sectors: dict[str, str] = {}
        self.industries: dict[str, str] = {}
        self.etf_flags: dict[str, bool] = {}

    def seed_instrument_ciks(self, ciks: dict[str, str]) -> int:
        self.ciks.update(ciks)
        return len(ciks)

    def seed_instrument_sectors(self, sectors: dict[str, str]) -> int:
        self.sectors.update(sectors)
        return len(sectors)

    def seed_instrument_industries(self, industries: dict[str, str]) -> int:
        self.industries.update(industries)
        return len(industries)

    def seed_instrument_etf_flags(self, flags: dict[str, bool]) -> int:
        self.etf_flags.update(flags)
        return len(flags)


class FakeEdgar:
    """``company_ticker_map`` over SEC's company_tickers.json."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls = 0

    def company_ticker_map(self) -> dict[str, str]:
        self.calls += 1
        return dict(self._mapping)


class FakeSecReference:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.quarters: list[str] = []

    def cik_to_sic(self, quarters: list[str]) -> dict[str, str]:
        self.quarters = list(quarters)
        return dict(self._mapping)


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = FakeRepo()
    # The whole market: 5 equities the platform knows about, only one of
    # which is in the active universe. Plus a crypto pair, which must be
    # excluded (no EDGAR filings; its sector is owned by crypto_bars).
    known = ["AAPL", "BIOTECHCO", "MSFT", "OLDCO", "REITCO"]
    on_file = {"AAPL": "0000320193"}  # the pre-existing 1-of-5 coverage

    edgar = FakeEdgar(
        {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "BIOTECHCO": "0000000042",
            "REITCO": "0000000077",
            "NOTOURS": "0000000999",  # SEC knows it; we do not track it
        }
    )
    sec = FakeSecReference(
        {
            "0000320193": "3571",  # computer equipment -> Information Technology
            "0000789019": "7372",  # software -> Information Technology
            "0000000042": "8731",  # commercial research -> Health Care
            "0000000077": "6798",  # REIT -> Real Estate
        }
    )

    monkeypatch.setattr(reference, "_repo", lambda _url: (object(), repo))
    monkeypatch.setattr(reference, "_settings", lambda: object())
    monkeypatch.setattr(reference, "_edgar", lambda _s: edgar)
    monkeypatch.setattr(reference, "_sec_reference", lambda _s: sec)
    monkeypatch.setattr(reference, "known_symbols", lambda _e: list(known))
    monkeypatch.setattr(
        reference, "instrument_ciks", lambda _e: {**on_file, **repo.ciks}
    )
    monkeypatch.setattr(reference, "instrument_sectors", lambda _e: dict(repo.sectors))
    monkeypatch.setattr(
        reference,
        "_coverage",
        lambda _e: {
            "rows": len(known),
            "cik": len({**on_file, **repo.ciks}),
            "sector": len(repo.sectors),
            "industry": len(repo.industries),
            "is_etf": sum(1 for f in repo.etf_flags.values() if f),
        },
    )
    return {"repo": repo, "edgar": edgar, "sec": sec, "known": known}


def test_ciks_seeds_every_known_symbol_not_just_the_active_universe(
    wired: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert reference.main(["ciks"]) == 0
    repo: FakeRepo = wired["repo"]
    # All four tracked symbols SEC knows about — this is the whole fix.
    assert repo.ciks == {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "BIOTECHCO": "0000000042",
        "REITCO": "0000000077",
    }
    # A symbol SEC maps but the platform does not track is never inserted.
    assert "NOTOURS" not in repo.ciks
    out = capsys.readouterr().out
    assert "1 of 5 symbols unmapped by SEC" in out  # OLDCO has no SEC mapping
    assert wired["edgar"].calls == 1  # one request for the whole market


def test_sectors_maps_cik_to_sic_to_gics(wired: dict[str, Any]) -> None:
    assert reference.main(["all"]) == 0
    repo: FakeRepo = wired["repo"]
    assert repo.sectors == {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "BIOTECHCO": "Health Care",
        "REITCO": "Real Estate",
    }
    assert repo.industries["BIOTECHCO"] == (
        "Engineering, Accounting, Research and Management"
    )


def test_sectors_defaults_to_the_four_most_recent_published_quarters(
    wired: dict[str, Any],
) -> None:
    reference.main(["ciks"])
    reference.main(["sectors"])
    quarters = wired["sec"].quarters
    assert len(quarters) == 4
    assert all(len(q) == 6 and q[4] == "q" for q in quarters)
    # Newest first: cik_to_sic lets earlier entries win, so ordering is
    # load-bearing (a re-classified filer must keep its newest SIC).
    assert quarters == sorted(quarters, reverse=True)


def test_explicit_quarters_are_passed_through(wired: dict[str, Any]) -> None:
    reference.main(["ciks"])
    reference.main(["sectors", "--quarters", "2025q4", "2025q3"])
    assert wired["sec"].quarters == ["2025q4", "2025q3"]


def test_dry_run_writes_nothing(wired: dict[str, Any]) -> None:
    assert reference.main(["--dry-run", "ciks"]) == 0
    assert wired["repo"].ciks == {}


def test_sectors_refuses_when_no_cik_is_on_file(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """Sector seeding is CIK-mediated; without CIKs it is a refusal, not a
    silent no-op that looks like success."""
    monkeypatch.setattr(reference, "instrument_ciks", lambda _e: {})
    assert reference.main(["sectors"]) == 1
    assert wired["repo"].sectors == {}


def test_each_seed_touches_exactly_one_column(wired: dict[str, Any]) -> None:
    """Additivity: the three seeds compose in any order because none of them
    can clobber another's column."""
    reference.main(["sectors"])  # sectors first, off the pre-existing AAPL CIK
    repo: FakeRepo = wired["repo"]
    assert repo.ciks == {} and repo.sectors == {"AAPL": "Information Technology"}
    reference.main(["ciks"])
    assert repo.sectors == {"AAPL": "Information Technology"}  # untouched


# --------------------------------------------------------------------------
# the fund / operating-company classifier (item 2)
# --------------------------------------------------------------------------

# The two real spot-check groups. SPY and QQQ are the interesting ones: they
# DO hold a CIK (they are EDGAR filers), so only signal 2 catches them. IWM
# and the sampled unmapped names fall to signal 1.
SEC_CIKS = {
    "AAPL": "0000320193",
    "JPM": "0000019617",
    "XOM": "0002115436",
    "SPY": "0000884394",
    "QQQ": "0001067839",
}
DERA_SIC = {  # only operating companies file XBRL financial statements
    "0000320193": "3571",
    "0000019617": "6022",
    "0002115436": "2911",
}
UNIVERSE = [
    "AAPL", "JPM", "XOM",                      # operating companies
    "SPY", "QQQ",                              # funds WITH a CIK
    "IWM", "VTWO", "IEFA", "EWA", "GSY", "AVLV", "BSCZ", "ROM",  # no CIK
]


def _classify(symbols: Any = None) -> Any:
    return reference.classify_etfs(
        symbols if symbols is not None else UNIVERSE,
        sec_ciks=SEC_CIKS,
        sic_by_cik=DERA_SIC,
    )


def test_etf_spot_check_groups() -> None:
    """SPY/QQQ/IWM true; AAPL/JPM/XOM false — the acceptance criterion."""
    flags = _classify().flags
    for symbol in ("SPY", "QQQ", "IWM"):
        assert flags[symbol] is True, symbol
    for symbol in ("AAPL", "JPM", "XOM"):
        assert flags[symbol] is False, symbol


def test_absence_from_company_tickers_is_the_fund_proxy() -> None:
    """Signal 1: the sampled unmapped symbols, all funds."""
    result = _classify()
    for symbol in ("VTWO", "IEFA", "EWA", "GSY", "AVLV", "BSCZ", "ROM", "IWM"):
        assert result.flags[symbol] is True
        assert result.reasons[symbol] == reference.ETF_ABSENT_FROM_SEC


def test_a_cik_without_a_dera_row_is_the_second_signal() -> None:
    """Signal 2 exists because signal 1 misses SPY: it files with EDGAR and
    holds a CIK, it just never files an XBRL financial statement."""
    result = _classify()
    for symbol in ("SPY", "QQQ"):
        assert result.flags[symbol] is True
        assert result.reasons[symbol] == reference.ETF_CIK_WITHOUT_SIC
    for symbol in ("AAPL", "JPM", "XOM"):
        assert result.reasons[symbol] == reference.EQUITY_CIK_WITH_SIC


def test_crypto_pairs_are_not_classified_at_all() -> None:
    """Crypto is neither a fund nor an issuer; is_crypto_symbol decides its
    asset class upstream of is_etf, so it must not appear here."""
    result = _classify([*UNIVERSE, "BTC/USD", "ETH/USD"])
    assert "BTC/USD" not in result.flags and "ETH/USD" not in result.flags
    assert result.flags["AAPL"] is False


def test_classification_is_idempotent_and_total() -> None:
    first, second = _classify(), _classify()
    assert first.flags == second.flags
    assert set(first.flags) == set(UNIVERSE)
    assert len(first.etfs) == 10 and len(first.equities) == 3
    assert first.reason_counts() == {
        reference.ETF_ABSENT_FROM_SEC: 8,
        reference.ETF_CIK_WITHOUT_SIC: 2,
        reference.EQUITY_CIK_WITH_SIC: 3,
    }


# --- the CLI around it ------------------------------------------------------


@pytest.fixture
def etf_wired(monkeypatch: pytest.MonkeyPatch) -> FakeRepo:
    repo = FakeRepo()
    repo.sectors.update(
        {"AAPL": "Information Technology", "JPM": "Financials", "XOM": "Energy"}
    )
    monkeypatch.setattr(reference, "_repo", lambda _url: (object(), repo))
    monkeypatch.setattr(reference, "_settings", lambda: object())
    monkeypatch.setattr(reference, "known_symbols", lambda _e: list(UNIVERSE))
    monkeypatch.setattr(reference, "instrument_ciks", lambda _e: dict(SEC_CIKS))
    monkeypatch.setattr(reference, "instrument_sectors", lambda _e: dict(repo.sectors))
    monkeypatch.setattr(
        reference,
        "_coverage",
        lambda _e: {
            "rows": len(UNIVERSE),
            "cik": len(SEC_CIKS),
            "sector": len(repo.sectors),
            "industry": 0,
            "is_etf": sum(1 for f in repo.etf_flags.values() if f),
        },
    )
    return repo


def test_etfs_command_writes_every_flag(
    etf_wired: FakeRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    assert reference.main(["etfs"]) == 0
    assert etf_wired.etf_flags["SPY"] is True
    assert etf_wired.etf_flags["AAPL"] is False
    assert len(etf_wired.etf_flags) == len(UNIVERSE)
    out = capsys.readouterr().out
    assert "spot-check SPY: is_etf=True" in out
    assert "spot-check AAPL: is_etf=False" in out


def test_etfs_dry_run_writes_nothing(etf_wired: FakeRepo) -> None:
    assert reference.main(["--dry-run", "etfs"]) == 0
    assert etf_wired.etf_flags == {}


def test_etfs_refuses_to_write_when_a_spot_check_fails(
    monkeypatch: pytest.MonkeyPatch, etf_wired: FakeRepo
) -> None:
    """A signal regression that reclassified SPY as an operating company
    would corrupt 11,505 rows. The command stops instead."""
    monkeypatch.setattr(
        reference, "instrument_sectors", lambda _e: {**etf_wired.sectors, "SPY": "Financials"}
    )
    assert reference.main(["etfs"]) == 1
    assert etf_wired.etf_flags == {}
