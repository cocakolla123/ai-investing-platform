"""Wide-universe backfill CLI paths: `universe-wide`, `bars --provider alpaca`,
and `surveillance-sweep`.

Every adapter is a fake — these tests never touch the network or a database.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from dynamo.cli import backfill
from dynamo.domain.models.market import Bar, Instrument, Tier, UniverseMembership

# -- fixtures / fakes -------------------------------------------------------


class FakeSettings:
    tiingo_api_key = "tiingo-key"
    alpaca_api_key = "alpaca-key"
    alpaca_api_secret = "alpaca-secret"
    crypto_enabled = False
    crypto_universe: tuple[str, ...] = ()


class FakeRepo:
    """Records every write; serves point-in-time reads from in-memory state."""

    def __init__(
        self,
        *,
        members: list[UniverseMembership] | None = None,
        bars: pd.DataFrame | None = None,
        coverage: dict[str, tuple[date, date]] | None = None,
    ) -> None:
        self.members = list(members or [])
        self._bars = bars if bars is not None else pd.DataFrame()
        self.coverage = dict(coverage or {})
        self.instrument_rows: list[dict[str, Any]] = []
        self.bar_rows: list[Bar] = []
        self.flags: list[Any] = []

    def universe_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]:
        return [
            m
            for m in self.members
            if m.valid_from <= as_of
            and (m.valid_to is None or m.valid_to >= as_of)
            and (tier is None or int(m.tier) == int(tier))
        ]

    def upsert_rows(self, table: Any, rows: list[dict[str, Any]]) -> int:
        self.instrument_rows.extend(rows)
        return len(rows)

    def upsert_universe_membership(self, memberships: Any) -> int:
        new = list(memberships)
        self.members.extend(new)
        return len(new)

    def upsert_bars(self, bars: Any) -> int:
        new = list(bars)
        self.bar_rows.extend(new)
        return len(new)

    def bars_frame(self, symbols: Any, start: date, end: date) -> pd.DataFrame:
        return self._bars

    def insert_surveillance_flags(self, flags: Any) -> int:
        new = list(flags)
        self.flags.extend(new)
        return len(new)


class FakeAlpaca:
    """Multi-symbol daily bars + a tradable-asset list."""

    def __init__(
        self,
        instruments: list[Instrument] | None = None,
        *,
        fail_on: set[str] | None = None,
        no_data: set[str] | None = None,
    ) -> None:
        self._instruments = list(instruments or [])
        self._fail_on = fail_on or set()
        self._no_data = no_data or set()
        self.requests: list[list[str]] = []

    def tradable_instruments(self) -> list[Instrument]:
        return list(self._instruments)

    def daily_bars(self, symbols: Any, start: date, end: date) -> list[Bar]:
        chunk = list(symbols)
        self.requests.append(chunk)
        if self._fail_on & set(chunk):
            raise RuntimeError("alpaca boom")
        return [
            Bar(
                symbol=s,
                ts=datetime(end.year, end.month, end.day, tzinfo=UTC),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=100,
                source="alpaca",
            )
            for s in chunk
            if s not in self._no_data
        ]


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the CLI's composition seams; return a setter for repo/adapter."""

    state: dict[str, Any] = {}
    monkeypatch.setattr(backfill, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill, "_repo", lambda url: state["repo"])
    monkeypatch.setattr(backfill, "_alpaca_market", lambda settings: state["alpaca"])
    monkeypatch.setattr(
        backfill,
        "_bar_coverage",
        lambda repo, source: dict(getattr(repo, "coverage", {})),
    )
    return state


def _inst(symbol: str, name: str, exchange: str) -> Instrument:
    return Instrument(symbol=symbol, name=name, exchange=exchange)


# -- pure filter ------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "name", "expected"),
    [
        ("AAPL", "Apple Inc. Common Stock", False),
        ("SPY", "State Street SPDR S&P 500 ETF Trust", False),
        ("BABA", "Alibaba Group Holding Limited American Depositary Shares", False),
        # Real common stocks whose ticker happens to end in W / R / U.
        ("UNIT", "Uniti Group Inc. Common Stock", False),
        ("RIGL", "Rigel Pharmaceuticals, Inc. Common Stock", False),
        ("UNTY", "Unity Bancorp, Inc. Common Stock", False),
        ("BRT", "BRT Apartments Corp. Common Stock", False),
        # Word-boundary traps that must NOT be excluded.
        ("WRB", "W. R. Berkley Corporation Common Stock", False),
        ("CBU", "Community Bank System, Inc. Common Stock", False),
        ("UAL", "United Airlines Holdings, Inc. Common Stock", False),
        # The classes we do want gone.
        ("BFRIW", "Biofrontera Inc. Warrants", True),
        ("HTZWW", "Hertz Global Holdings, Inc Warrant", True),
        ("KOYNU", "CSLM Digital Asset Acquisition Corp III Units", True),
        ("ABCR", "Acme Acquisition Corp. Rights", True),
        ("BMNP", "BitMine Immersion Technologies, Inc. 9.5% Series A Perpetual Preferred", True),
        ("MTB.PRJ", "M&T Bank Corporation Depositary Shares representing Preferred", True),
        ("ABC.WS", "Some Corp Class A", True),
        ("XYZ.U", "Some Corp Class A", True),
        ("GSBD", "Goldman Sachs BDC 6.375% Notes due 2030", True),
    ],
)
def test_is_non_common_share_classifies_security_types(
    symbol: str, name: str, expected: bool
) -> None:
    assert backfill.is_non_common_share(symbol, name) is expected


def test_filter_surveillance_candidates_reports_the_funnel() -> None:
    kept, counts = backfill.filter_surveillance_candidates(
        [
            _inst("AAPL", "Apple Inc. Common Stock", "NASDAQ"),
            _inst("SPY", "SPDR S&P 500 ETF Trust", "ARCA"),
            _inst("UPMMY", "UPM KYMMENE CORP ADR -Unsponsored (Finland)", "OTC"),
            _inst("BFRIW", "Biofrontera Inc. Warrants", "NASDAQ"),
            _inst("AAPL", "Apple Inc. Common Stock", "NASDAQ"),
        ]
    )
    assert [i.symbol for i in kept] == ["AAPL", "SPY"]
    assert counts == {
        "fetched": 5,
        "dropped_exchange": 1,
        "dropped_class": 1,
        "dropped_duplicate": 1,
        "kept": 2,
    }


def test_filter_surveillance_candidates_honors_custom_exchanges() -> None:
    kept, counts = backfill.filter_surveillance_candidates(
        [_inst("AAPL", "Apple Inc. Common Stock", "NASDAQ")], exchanges=["NYSE"]
    )
    assert kept == [] and counts["dropped_exchange"] == 1


# -- chunking / resume ------------------------------------------------------


def test_chunked_splits_without_losing_symbols() -> None:
    assert backfill.chunked(["A", "B", "C"], 2) == [["A", "B"], ["C"]]
    assert backfill.chunked([], 2) == []
    assert backfill.chunked(["A", "B"], 0) == [["A"], ["B"]]  # degenerate size clamps to 1


def test_covered_symbols_uses_tolerance_at_both_edges() -> None:
    start, end = date(2024, 1, 1), date(2026, 8, 16)
    coverage = {
        "FULL": (date(2024, 1, 2), date(2026, 8, 14)),  # inside tolerance both ends
        "LATE_START": (date(2025, 6, 1), date(2026, 8, 14)),  # listed mid-range
        "STALE": (date(2024, 1, 2), date(2026, 1, 1)),  # stopped updating
    }
    assert backfill.covered_symbols(coverage, start, end) == {"FULL"}
    assert backfill.covered_symbols(coverage, start, end, tolerance_days=0) == set()


# -- universe-wide ----------------------------------------------------------


def test_universe_wide_writes_tier2_and_leaves_tier1_untouched(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    core = UniverseMembership(
        symbol="AAPL",
        tier=Tier.CORE,
        valid_from=date(2024, 1, 1),
        reason="seed",
        approved_by="seed",
    )
    wired["repo"] = FakeRepo(members=[core])
    wired["alpaca"] = FakeAlpaca(
        [
            _inst("AAPL", "Apple Inc. Common Stock", "NASDAQ"),
            _inst("PLTR", "Palantir Technologies Inc. Class A Common Stock", "NASDAQ"),
            _inst("BFRIW", "Biofrontera Inc. Warrants", "NASDAQ"),
            _inst("UPMMY", "UPM KYMMENE ADR", "OTC"),
        ]
    )
    rc = backfill.main(["universe-wide", "--as-of", "2026-08-17"])
    repo: FakeRepo = wired["repo"]
    assert rc == 0
    # AAPL is already CORE -> only PLTR gets a Tier-2 row; CORE is unchanged.
    tier2 = [m for m in repo.members if int(m.tier) == int(Tier.SURVEILLANCE)]
    assert [m.symbol for m in tier2] == ["PLTR"]
    assert tier2[0].valid_from == date(2026, 8, 17) and tier2[0].valid_to is None
    assert [m.symbol for m in repo.members if int(m.tier) == int(Tier.CORE)] == ["AAPL"]
    # Instruments upsert is ADDITIVE: no sector/cik/status keys are supplied,
    # so the ON CONFLICT DO UPDATE cannot clobber enrichment already on file.
    assert {r["symbol"] for r in repo.instrument_rows} == {"AAPL", "PLTR"}
    assert all(set(r) == {"symbol", "name", "exchange"} for r in repo.instrument_rows)
    out = capsys.readouterr().out
    assert "UNCHANGED" in out and "4 active/tradable US-equity assets" in out


def test_universe_wide_dry_run_writes_nothing(wired: Any) -> None:
    wired["repo"] = FakeRepo()
    wired["alpaca"] = FakeAlpaca([_inst("PLTR", "Palantir Class A Common Stock", "NASDAQ")])
    assert backfill.main(["universe-wide", "--dry-run"]) == 0
    repo: FakeRepo = wired["repo"]
    assert repo.members == [] and repo.instrument_rows == []


def test_universe_wide_defaults_valid_from_to_today(wired: Any) -> None:
    wired["repo"] = FakeRepo()
    wired["alpaca"] = FakeAlpaca([_inst("PLTR", "Palantir Class A Common Stock", "NASDAQ")])
    assert backfill.main(["universe-wide"]) == 0
    assert wired["repo"].members[0].valid_from == date.today()


def test_universe_wide_refuses_to_write_when_the_filter_empties_the_list(wired: Any) -> None:
    wired["repo"] = FakeRepo()
    wired["alpaca"] = FakeAlpaca([_inst("BFRIW", "Biofrontera Inc. Warrants", "NASDAQ")])
    assert backfill.main(["universe-wide"]) == 1
    assert wired["repo"].members == []


def test_universe_wide_requires_alpaca_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoKeys(FakeSettings):
        alpaca_api_key = ""

    monkeypatch.setattr(backfill, "_settings", lambda: NoKeys())
    monkeypatch.setattr(
        backfill, "_repo", lambda url: (_ for _ in ()).throw(AssertionError("no DB"))
    )
    assert backfill.main(["universe-wide"]) == 1


# -- bars --provider alpaca -------------------------------------------------


def _bars_argv(*extra: str) -> list[str]:
    return [
        "bars",
        "--provider",
        "alpaca",
        "--start",
        "2024-01-01",
        "--end",
        "2026-08-16",
        *extra,
    ]


def test_bars_alpaca_batches_many_symbols_per_request(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    wired["repo"] = FakeRepo()
    wired["alpaca"] = FakeAlpaca()
    symbols = [f"S{i:03d}" for i in range(5)]
    rc = backfill.main(_bars_argv("--chunk-size", "2", "--symbols", *symbols))
    alpaca: FakeAlpaca = wired["alpaca"]
    assert rc == 0
    assert alpaca.requests == [["S000", "S001"], ["S002", "S003"], ["S004"]]
    assert len(wired["repo"].bar_rows) == 5
    assert "rows/s" in capsys.readouterr().out


def test_bars_alpaca_skips_symbols_already_covered(wired: Any) -> None:
    wired["repo"] = FakeRepo(coverage={"COVERED": (date(2024, 1, 2), date(2026, 8, 14))})
    wired["alpaca"] = FakeAlpaca()
    rc = backfill.main(_bars_argv("--symbols", "COVERED", "NEW"))
    assert rc == 0
    assert wired["alpaca"].requests == [["NEW"]]


def test_bars_alpaca_force_refetches_covered_symbols(wired: Any) -> None:
    wired["repo"] = FakeRepo(coverage={"COVERED": (date(2024, 1, 2), date(2026, 8, 14))})
    wired["alpaca"] = FakeAlpaca()
    assert backfill.main(_bars_argv("--force", "--symbols", "COVERED")) == 0
    assert wired["alpaca"].requests == [["COVERED"]]


def test_bars_alpaca_bad_symbol_degrades_to_per_symbol_and_keeps_going(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    wired["repo"] = FakeRepo()
    wired["alpaca"] = FakeAlpaca(fail_on={"BAD"})
    rc = backfill.main(_bars_argv("--chunk-size", "3", "--symbols", "A", "BAD", "C"))
    alpaca: FakeAlpaca = wired["alpaca"]
    assert rc == 1  # the failure is reported, not swallowed
    assert alpaca.requests == [["A", "BAD", "C"], ["A"], ["BAD"], ["C"]]
    assert {b.symbol for b in wired["repo"].bar_rows} == {"A", "C"}
    assert "1 symbols failed" in capsys.readouterr().out


def test_bars_alpaca_reports_symbols_that_returned_no_bars(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    wired["repo"] = FakeRepo()
    wired["alpaca"] = FakeAlpaca(no_data={"QUIET"})
    assert backfill.main(_bars_argv("--symbols", "A", "QUIET")) == 0
    out = capsys.readouterr().out
    assert "1 returned none" in out and "IEX-only" in out


def test_bars_alpaca_universe_can_target_a_single_tier(wired: Any) -> None:
    wired["repo"] = FakeRepo(
        members=[
            UniverseMembership(
                symbol="CORE1", tier=Tier.CORE, valid_from=date(2024, 1, 1), reason="seed"
            ),
            UniverseMembership(
                symbol="WIDE1", tier=Tier.SURVEILLANCE, valid_from=date(2024, 1, 1), reason="wide"
            ),
        ]
    )
    wired["alpaca"] = FakeAlpaca()
    assert backfill.main(_bars_argv("--universe", "--tiers", "2")) == 0
    assert wired["alpaca"].requests == [["WIDE1"]]


def test_bars_default_provider_is_still_tiingo() -> None:
    args = backfill._build_parser().parse_args(
        ["bars", "--start", "2024-01-01", "--end", "2026-01-01", "--symbols", "AAPL"]
    )
    assert args.provider == "tiingo" and args.tiers is None


def test_bars_alpaca_requires_alpaca_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoKeys(FakeSettings):
        alpaca_api_secret = ""

    monkeypatch.setattr(backfill, "_settings", lambda: NoKeys())
    monkeypatch.setattr(
        backfill, "_repo", lambda url: (_ for _ in ()).throw(AssertionError("no DB"))
    )
    assert backfill.main(_bars_argv("--symbols", "AAPL")) == 1


# -- surveillance-sweep -----------------------------------------------------


def _bars_frame(symbol: str, closes: list[float], volumes: list[int]) -> pd.DataFrame:
    base = date(2026, 8, 14)
    n = len(closes)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts": [
                datetime.combine(base - timedelta(days=n - 1 - i), datetime.min.time(), tzinfo=UTC)
                for i in range(n)
            ],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "volume": volumes,
            "source": ["alpaca"] * n,
        }
    )


# price_zscore now requires a tradable name (D1/D2/D5): a $1M/day
# consolidated-equivalent floor, which on these source='alpaca' rows means
# ~$66k/day printed. The old fixture traded $10k/day — genuinely untradable,
# so the detector was right to drop it. Base volume is realistic instead, and
# the 20x spike ratio volume_spike keys off is unchanged.
BASE_VOLUME = 200_000
SPIKE_VOLUME = BASE_VOLUME * 20


def _spike_volumes() -> list[int]:
    return [BASE_VOLUME] * 70 + [SPIKE_VOLUME]


def test_surveillance_sweep_writes_real_flags(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # 70 flat sessions then a +40% jump on 20x volume: price_zscore,
    # volume_spike and range_break should all fire.
    closes = [10.0 + 0.01 * (i % 3) for i in range(70)] + [14.0]
    volumes = _spike_volumes()
    wired["repo"] = FakeRepo(
        members=[
            UniverseMembership(
                symbol="SPIKE", tier=Tier.SURVEILLANCE, valid_from=date(2024, 1, 1), reason="wide"
            )
        ],
        bars=_bars_frame("SPIKE", closes, volumes),
    )
    wired["alpaca"] = FakeAlpaca()
    rc = backfill.main(["surveillance-sweep", "--as-of", "2026-08-14"])
    repo: FakeRepo = wired["repo"]
    assert rc == 0
    detectors = {f.detector for f in repo.flags}
    assert detectors == {"price_zscore", "volume_spike", "range_break"}
    assert all(f.symbol == "SPIKE" for f in repo.flags)
    assert all(f.ts == datetime(2026, 8, 14, 21, 0, tzinfo=UTC) for f in repo.flags)
    out = capsys.readouterr().out
    assert "FLAG SPIKE" in out
    # The LLM-dependent and event-table detectors are named as NOT run.
    assert "news_burst" in out and "rel_strength" in out


def test_surveillance_sweep_dry_run_writes_nothing(wired: Any) -> None:
    closes = [10.0 + 0.01 * (i % 3) for i in range(70)] + [14.0]
    wired["repo"] = FakeRepo(
        members=[
            UniverseMembership(
                symbol="SPIKE", tier=Tier.SURVEILLANCE, valid_from=date(2024, 1, 1), reason="wide"
            )
        ],
        bars=_bars_frame("SPIKE", closes, _spike_volumes()),
    )
    assert backfill.main(["surveillance-sweep", "--as-of", "2026-08-14", "--dry-run"]) == 0
    assert wired["repo"].flags == []


def test_surveillance_sweep_applies_the_liquidity_floor(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    closes = [10.0 + 0.01 * (i % 3) for i in range(70)] + [14.0]
    wired["repo"] = FakeRepo(
        members=[
            UniverseMembership(
                symbol="SPIKE", tier=Tier.SURVEILLANCE, valid_from=date(2024, 1, 1), reason="wide"
            )
        ],
        bars=_bars_frame("SPIKE", closes, _spike_volumes()),
    )
    rc = backfill.main(
        [
            "surveillance-sweep",
            "--as-of",
            "2026-08-14",
            "--min-dollar-volume",
            "1000000000",
        ]
    )
    assert rc == 0
    assert wired["repo"].flags == []
    assert "liquidity floor removed every symbol" in capsys.readouterr().out


def test_surveillance_sweep_no_members_is_a_clean_no_op(
    wired: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    wired["repo"] = FakeRepo()
    assert backfill.main(["surveillance-sweep", "--as-of", "2026-08-14"]) == 0
    assert "nothing to scan" in capsys.readouterr().out
