"""CLI tests for replay + backfill (plan Phase 4): arg parsing, the
committee Phase-5 stub, graceful no-key skips, and adapter-mocked upsert
paths. NO network, NO database — every seam is monkeypatched."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from dynamo.cli import backfill as backfill_cli
from dynamo.cli import replay as replay_cli

# --------------------------------------------------------------------------
# replay CLI
# --------------------------------------------------------------------------


def test_replay_run_requires_start_end_and_label() -> None:
    with pytest.raises(SystemExit):
        replay_cli._build_parser().parse_args(["run", "--start", "2026-06-01"])
    args = replay_cli._build_parser().parse_args(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert args.start == date(2026, 6, 1)
    assert args.end == date(2026, 6, 10)
    assert args.mode == "quant"
    assert args.universe_tier is None


def test_replay_committee_mode_requires_max_usd(capsys: pytest.CaptureFixture[str]) -> None:
    """--mode committee without --max-usd is an argparse ERROR (exit 2) —
    there is no default cap, ever. Fails before touching engine/DB/LLM."""
    with pytest.raises(SystemExit) as exc:
        replay_cli.main(
            [
                "run",
                "--start", "2026-06-01",
                "--end", "2026-06-10",
                "--mode", "committee",
                "--run-label", "x",
            ]
        )
    assert exc.value.code == 2
    assert "--max-usd" in capsys.readouterr().err


def test_replay_committee_max_usd_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        replay_cli.main(
            [
                "run",
                "--start", "2026-06-01",
                "--end", "2026-06-10",
                "--mode", "committee",
                "--run-label", "x",
                "--max-usd", "0",
            ]
        )


def test_replay_committee_refuses_without_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With a cap but no ANTHROPIC_API_KEY the committee run refuses before
    touching the database or constructing any client."""

    class KeylessSettings:
        anthropic_api_key = ""

    monkeypatch.setattr("dynamo.config.settings.Settings", lambda: KeylessSettings())
    rc = replay_cli.main(
        [
            "run",
            "--start", "2026-06-01",
            "--end", "2026-06-10",
            "--mode", "committee",
            "--run-label", "x",
            "--max-usd", "5.0",
        ]
    )
    assert rc == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_estimate_prints_projection_without_any_api_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The estimator is pure math over on-disk prompt sizes + the pricing
    table — no client is constructed, no request object even built."""
    rc = replay_cli.main(
        ["estimate", "--start", "2026-06-01", "--end", "2026-06-10"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "NO API calls" in out
    assert "projected total:" in out
    assert "$" in out


def test_estimate_scales_with_deep_dive_cap(capsys: pytest.CaptureFixture[str]) -> None:
    def total_for(cap: str) -> float:
        rc = replay_cli.main(
            [
                "estimate",
                "--start", "2026-06-01",
                "--end", "2026-06-10",
                "--deep-dives-per-day", cap,
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        line = next(ln for ln in out.splitlines() if "projected total:" in ln)
        return float(line.split("$")[-1])

    # Printed at 2 decimals — allow rounding slack around the exact 2x.
    assert total_for("8") == pytest.approx(2.0 * total_for("4"), abs=0.03)


def test_replay_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        replay_cli._build_parser().parse_args(
            ["run", "--start", "2026-06-01", "--end", "2026-06-10",
             "--mode", "yolo", "--run-label", "x"]
        )


def test_replay_list_and_show_parse() -> None:
    args = replay_cli._build_parser().parse_args(["list", "--limit", "5"])
    assert args.command == "list" and args.limit == 5
    args = replay_cli._build_parser().parse_args(["show", "--run-id", "replay-abc"])
    assert args.command == "show" and args.run_id == "replay-abc"


# --------------------------------------------------------------------------
# backfill CLI — shared fakes
# --------------------------------------------------------------------------


@dataclass
class FakeSettings:
    tiingo_api_key: str = ""
    fred_api_key: str = ""
    edgar_user_agent: str = "test agent test@example.com"


@dataclass
class FakeRepo:
    bars: list[Any] = field(default_factory=list)
    facts: list[Any] = field(default_factory=list)
    macro: list[Any] = field(default_factory=list)

    def upsert_bars(self, bars: Any) -> int:
        batch = list(bars)
        self.bars.extend(batch)
        return len(batch)

    def upsert_xbrl_facts(self, facts: Any) -> int:
        batch = list(facts)
        self.facts.extend(batch)
        return len(batch)

    def upsert_macro(self, points: Any) -> int:
        batch = list(points)
        self.macro.extend(batch)
        return len(batch)

    def universe_at(self, as_of: date) -> list[Any]:
        @dataclass
        class M:
            symbol: str

        return [M("AAA"), M("BBB")]


def test_backfill_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        backfill_cli._build_parser().parse_args([])


def test_backfill_bars_requires_symbols_or_universe() -> None:
    with pytest.raises(SystemExit):
        backfill_cli._build_parser().parse_args(
            ["bars", "--start", "2026-01-01", "--end", "2026-02-01"]
        )
    args = backfill_cli._build_parser().parse_args(
        ["bars", "--start", "2026-01-01", "--end", "2026-02-01", "--universe"]
    )
    assert args.universe is True and args.symbols is None


def test_backfill_bars_skips_gracefully_without_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        backfill_cli, "_repo", lambda url: pytest.fail("must not touch the database")
    )
    rc = backfill_cli.main(
        ["bars", "--start", "2026-01-01", "--end", "2026-02-01", "--symbols", "AAA"]
    )
    assert rc == 0
    assert "TIINGO_API_KEY is not set" in capsys.readouterr().out


def test_backfill_macro_skips_gracefully_without_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    rc = backfill_cli.main(["macro", "--start", "2020-01-01"])
    assert rc == 0
    assert "FRED_API_KEY is not set" in capsys.readouterr().out


def test_backfill_bars_upserts_range_per_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    calls: list[tuple[list[str], date, date]] = []

    class FakeTiingo:
        def daily_bars(self, symbols: Any, start: date, end: date) -> Any:
            calls.append((list(symbols), start, end))
            yield {"symbol": symbols[0]}

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(tiingo_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_tiingo", lambda s: FakeTiingo())
    rc = backfill_cli.main(
        [
            "bars", "--start", "2025-01-01", "--end", "2026-01-01",
            "--symbols", "AAA", "BBB",
        ]
    )
    assert rc == 0
    assert calls == [
        (["AAA"], date(2025, 1, 1), date(2026, 1, 1)),
        (["BBB"], date(2025, 1, 1), date(2026, 1, 1)),
    ]
    assert len(repo.bars) == 2


def test_backfill_bars_universe_resolves_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    seen: list[str] = []

    class FakeTiingo:
        def daily_bars(self, symbols: Any, start: date, end: date) -> Any:
            seen.extend(symbols)
            return iter(())

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(tiingo_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_tiingo", lambda s: FakeTiingo())
    rc = backfill_cli.main(
        ["bars", "--start", "2025-01-01", "--end", "2026-01-01", "--universe"]
    )
    assert rc == 0
    assert seen == ["AAA", "BBB"]


def test_backfill_fundamentals_uses_cik_map(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    fetched: list[str] = []

    class FakeEdgar:
        def company_facts(self, cik: str) -> Any:
            fetched.append(cik)
            yield {"cik": cik}

    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_edgar", lambda s: FakeEdgar())
    monkeypatch.setattr(
        backfill_cli, "_cik_map", lambda repo, symbols: {"AAA": "0000000001"}
    )
    rc = backfill_cli.main(["fundamentals", "--symbols", "AAA", "NOCIK"])
    assert rc == 0
    assert fetched == ["0000000001"]  # NOCIK skipped with a message
    assert len(repo.facts) == 1


def test_backfill_macro_fetches_vintages_for_default_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeRepo()
    calls: list[tuple[str, date, date]] = []

    class FakeFred:
        def series_vintages(self, series_id: str, start: date, realtime_start: date) -> Any:
            calls.append((series_id, start, realtime_start))
            yield {"series_id": series_id}

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(fred_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_fred", lambda s: FakeFred())
    rc = backfill_cli.main(["macro", "--start", "2020-01-01"])
    assert rc == 0
    assert [c[0] for c in calls] == list(backfill_cli.DEFAULT_FRED_SERIES)
    assert all(c[1] == c[2] == date(2020, 1, 1) for c in calls)
    assert len(repo.macro) == len(backfill_cli.DEFAULT_FRED_SERIES)


def test_backfill_macro_custom_series(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    calls: list[str] = []

    class FakeFred:
        def series_vintages(self, series_id: str, start: date, realtime_start: date) -> Any:
            calls.append(series_id)
            return iter(())

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(fred_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_fred", lambda s: FakeFred())
    rc = backfill_cli.main(["macro", "--start", "2024-01-01", "--series", "VIXCLS"])
    assert rc == 0
    assert calls == ["VIXCLS"]


def test_backfill_default_series_matches_production_ingestion() -> None:
    from orchestration.resources import DEFAULT_FRED_SERIES

    assert backfill_cli.DEFAULT_FRED_SERIES == DEFAULT_FRED_SERIES
