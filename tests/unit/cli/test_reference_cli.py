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

    def seed_instrument_ciks(self, ciks: dict[str, str]) -> int:
        self.ciks.update(ciks)
        return len(ciks)

    def seed_instrument_sectors(self, sectors: dict[str, str]) -> int:
        self.sectors.update(sectors)
        return len(sectors)

    def seed_instrument_industries(self, industries: dict[str, str]) -> int:
        self.industries.update(industries)
        return len(industries)


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
    monkeypatch.setattr(
        reference,
        "_coverage",
        lambda _e: {
            "rows": len(known),
            "cik": len({**on_file, **repo.ciks}),
            "sector": len(repo.sectors),
            "industry": len(repo.industries),
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
