"""SIC → GICS-sector mapping: the cases where SIC and GICS disagree."""

from __future__ import annotations

import pytest

from dynamo.data.sic import (
    _MAJOR_GROUPS,
    _SECTOR_RULES,
    industry_for_sic,
    normalize_sic,
    sector_for_sic,
)
from dynamo.data.universe import default_sector_etf_map


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        # The exceptions are the whole point: SIC files these by production
        # process, GICS by end market, and getting them wrong silently
        # corrupts every sector-relative feature and the risk sector caps.
        ("2834", "Health Care"),  # pharmaceutical preparations — inside Chemicals
        ("2836", "Health Care"),  # biological products
        ("8731", "Health Care"),  # commercial research — where biotech files
        ("3841", "Health Care"),  # surgical & medical instruments
        ("2844", "Consumer Staples"),  # cosmetics — inside Chemicals
        ("2860", "Materials"),  # industrial organic chemicals
        ("3711", "Consumer Discretionary"),  # motor vehicles — inside Transport Equip
        ("3721", "Industrials"),  # aircraft
        ("3674", "Information Technology"),  # semiconductors
        ("7372", "Information Technology"),  # prepackaged software
        ("7310", "Communication Services"),  # advertising — inside Business Services
        ("4813", "Communication Services"),  # telephone communications
        ("7812", "Communication Services"),  # motion picture production
        ("6021", "Financials"),  # national commercial banks
        ("6770", "Financials"),  # blank checks (SPACs)
        ("6798", "Real Estate"),  # REITs — inside Holding/Investment Offices
        ("6512", "Real Estate"),  # operators of apartment buildings
        ("1311", "Energy"),  # crude petroleum & natural gas
        ("2911", "Energy"),  # petroleum refining
        ("4610", "Energy"),  # crude petroleum pipelines — NOT Utilities
        ("4911", "Utilities"),  # electric services
        ("4924", "Utilities"),  # natural gas distribution
        ("5912", "Consumer Staples"),  # drug stores — inside Misc Retail
        ("5812", "Consumer Discretionary"),  # eating places
        ("5122", "Health Care"),  # drugs wholesale — inside Nondurable Wholesale
        ("2011", "Consumer Staples"),  # meat packing
        ("3312", "Materials"),  # steel works
        ("1531", "Industrials"),  # operative builders
        ("4512", "Industrials"),  # air transportation, scheduled
    ],
)
def test_sector_for_sic(sic: str, sector: str) -> None:
    assert sector_for_sic(sic) == sector


@pytest.mark.parametrize("value", [None, "", "  ", "abc", "0000", "9995"])
def test_unmappable_returns_none_never_a_placeholder(value: str | None) -> None:
    """A missing sector makes a feature abstain; a wrong one makes it lie."""
    assert normalize_sic(value) is None
    assert sector_for_sic(value) is None
    assert industry_for_sic(value) is None


def test_every_sector_name_is_one_the_platform_already_speaks() -> None:
    """The mapping must land on names ``default_sector_etf_map`` recognizes,
    or the rel_strength detector silently drops every symbol again."""
    produced = {sector for _, _, sector in _SECTOR_RULES}
    assert len(produced) == 11  # the eleven GICS sectors, no invented ones
    unmapped = {s for s in produced if not default_sector_etf_map({"X": s})}
    assert unmapped == set(), f"sectors with no sector SPDR: {unmapped}"


def test_rules_are_ordered_most_specific_first() -> None:
    """Exceptions must precede the broad range they sit inside, else the
    broad rule wins and the exception is dead code."""
    for index, (low, high, _) in enumerate(_SECTOR_RULES):
        for other_low, other_high, _ in _SECTOR_RULES[index + 1 :]:
            enclosed = other_low <= low and high <= other_high
            assert not (enclosed and (other_low, other_high) == (low, high)), (
                f"duplicate range {low}-{high}"
            )
            # A later rule may enclose an earlier one (that is the point);
            # an earlier rule must never strictly enclose a later one.
            assert not (low < other_low and other_high < high), (
                f"rule {low}-{high} shadows the later, narrower {other_low}-{other_high}"
            )


def test_industry_is_the_sic_major_group() -> None:
    assert industry_for_sic("2834") == "Chemicals and Allied Products"
    assert industry_for_sic("7372") == "Business Services"
    assert industry_for_sic("6021") == "Depository Institutions"
    # An unknown major group yields nothing rather than a guess.
    assert industry_for_sic("9721") is None
    assert all(0 < group < 100 for group in _MAJOR_GROUPS)
