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

# Captured BEFORE any test monkeypatches dynamo.config.settings.Settings —
# the backend tests build real Settings instances after patching the module.
from dynamo.config.settings import Settings as RealSettings

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
        research_backend = "api"

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
                # Exact 2x scaling only holds without the flat per-session
                # macro-digest overhead (Phase 4) — drop it here.
                "--no-macro-digest",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        line = next(ln for ln in out.splitlines() if "projected total:" in ln)
        return float(line.split("$")[-1])

    # Printed at 2 decimals — allow rounding slack around the exact 2x.
    assert total_for("8") == pytest.approx(2.0 * total_for("4"), abs=0.03)


def test_replay_run_quant_bundle_flags_parse_with_safe_defaults() -> None:
    """Phase-2 flags: defaults preserve old behavior (baseline-v0, age 5,
    empty code version); explicit values parse through."""
    parser = replay_cli._build_parser()
    args = parser.parse_args(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert args.quant_model_version == ""
    assert args.quant_score_max_age_days == 5
    assert args.code_version == ""
    args = parser.parse_args(
        [
            "run",
            "--start", "2026-06-01",
            "--end", "2026-06-10",
            "--run-label", "x",
            "--quant-model-version", "campaign-v1",
            "--quant-score-max-age-days", "7",
            "--code-version", "abc123",
        ]
    )
    assert args.quant_model_version == "campaign-v1"
    assert args.quant_score_max_age_days == 7
    assert args.code_version == "abc123"


def test_replay_portfolio_flags_parse_with_safe_defaults() -> None:
    """Phase-3 flags: defaults preserve old behavior; explicit values parse."""
    parser = replay_cli._build_parser()
    args = parser.parse_args(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert args.source_run_id == ""
    assert args.whole_shares is False
    assert args.min_order_notional == 0.0
    assert args.commission_per_order == 0.0
    args = parser.parse_args(
        [
            "run",
            "--start", "2026-02-02",
            "--end", "2026-07-31",
            "--run-label", "1k",
            "--mode", "portfolio",
            "--source-run-id", "replay-abc123def456",
            "--whole-shares",
            "--min-order-notional", "25",
            "--commission-per-order", "1.0",
        ]
    )
    assert args.mode == "portfolio"
    assert args.source_run_id == "replay-abc123def456"
    assert args.whole_shares is True
    assert args.min_order_notional == 25.0
    assert args.commission_per_order == 1.0


def test_replay_portfolio_mode_requires_source_run_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        replay_cli.main(
            [
                "run",
                "--start", "2026-06-01",
                "--end", "2026-06-10",
                "--mode", "portfolio",
                "--run-label", "x",
            ]
        )
    assert exc.value.code == 2
    assert "--source-run-id" in capsys.readouterr().err


def test_replay_portfolio_mode_never_constructs_an_llm_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI token firewall: portfolio mode routes through the quant/
    portfolio command, where the LLM wiring is structurally unreachable —
    with the AnthropicClient constructor rigged to explode, a portfolio run
    still reaches the runner (patched here) and completes."""
    import dynamo.llm.anthropic_client as llm_mod
    import dynamo.replay.preflight as preflight_mod
    import dynamo.replay.runner as runner_mod

    monkeypatch.setattr(
        llm_mod.AnthropicClient,
        "__init__",
        lambda *a, **k: pytest.fail("portfolio mode must never construct an LLM client"),
    )
    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())
    monkeypatch.setattr(
        preflight_mod, "run_preflight", lambda engine, config: pytest.fail("skipped")
    )
    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, progress: Any = None) -> Any:
            @dataclass
            class R:
                run_id: str = "replay-fake"
                status: str = "completed"
                sessions: tuple = ()
                decisions: tuple = ()
                fills: tuple = ()
                experiences: tuple = ()
                equity_curve: tuple = ()

            return R()

    monkeypatch.setattr(runner_mod, "ReplayRunner", FakeRunner)
    rc = replay_cli.main(
        [
            "run",
            "--start", "2026-06-01",
            "--end", "2026-06-10",
            "--mode", "portfolio",
            "--source-run-id", "replay-src",
            "--run-label", "x",
            "--skip-preflight",
        ]
    )
    assert rc == 0
    assert captured["config"].mode == "portfolio"
    assert captured["config"].source_run_id == "replay-src"
    # The composition layer wires the repository verdict read as the source.
    assert captured["verdict_source"] is not None


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
    finnhub_api_key: str = ""
    edgar_user_agent: str = "test agent test@example.com"
    crypto_enabled: bool = True
    crypto_universe: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""


@dataclass
class FakeRepo:
    bars: list[Any] = field(default_factory=list)
    facts: list[Any] = field(default_factory=list)
    macro: list[Any] = field(default_factory=list)
    news: list[Any] = field(default_factory=list)
    transcripts: list[Any] = field(default_factory=list)
    linked: list[tuple[str, date, str]] = field(default_factory=list)
    seeded_ciks: dict[str, str] = field(default_factory=dict)

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

    def upsert_news(self, items: Any) -> int:
        batch = list(items)
        self.news.extend(batch)
        return len(batch)

    def upsert_transcripts(self, transcripts: Any) -> int:
        batch = list(transcripts)
        self.transcripts.extend(batch)
        return len(batch)

    def link_transcript_ref(self, symbol: str, event_date: date, ref: str) -> int:
        self.linked.append((symbol, event_date, ref))
        return 1

    def seed_instrument_ciks(self, ciks: Any) -> int:
        self.seeded_ciks.update(dict(ciks))
        return len(ciks)

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


def test_backfill_bars_fails_loudly_without_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Phase 1 hardening: a missing Tiingo key is a NONZERO exit — a bars
    backfill that silently 'succeeds' would let replay preflight blame the
    wrong layer."""
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        backfill_cli, "_repo", lambda url: pytest.fail("must not touch the database")
    )
    rc = backfill_cli.main(
        ["bars", "--start", "2026-01-01", "--end", "2026-02-01", "--symbols", "AAA"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "TIINGO_API_KEY is not set" in out
    assert "--allow-missing-tiingo" in out


def test_backfill_bars_allow_missing_tiingo_restores_graceful_skip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        backfill_cli, "_repo", lambda url: pytest.fail("must not touch the database")
    )
    rc = backfill_cli.main(
        [
            "bars", "--start", "2026-01-01", "--end", "2026-02-01",
            "--symbols", "AAA", "--allow-missing-tiingo",
        ]
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


# --------------------------------------------------------------------------
# backfill universe --as-of (replay campaign backdate)
# --------------------------------------------------------------------------


class FakeUniverseRepo:
    """Stateful membership store mimicking the repo's backdate semantics."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, date], Any] = {}
        self.sectors: dict[str, str] = {}

    def seed_instrument_sectors(self, sectors: Any) -> int:
        self.sectors = dict(sectors)
        return len(self.sectors)

    def backdate_core_universe(
        self, memberships: Any, *, delete_approved_by: Any = ("seed", "replay-backdate")
    ) -> tuple[int, int]:
        ms = list(memberships)
        symbols = {m.symbol for m in ms}
        doomed = [
            key
            for key, row in self.rows.items()
            if row.symbol in symbols
            and int(row.tier) == 1
            and row.approved_by in delete_approved_by
        ]
        for key in doomed:
            del self.rows[key]
        for m in ms:
            self.rows[(m.symbol, m.valid_from)] = m
        return len(doomed), len(ms)

    def universe_at(self, as_of: date) -> list[Any]:
        return [
            m
            for m in self.rows.values()
            if m.valid_from <= as_of and (m.valid_to is None or m.valid_to >= as_of)
        ]


def test_backfill_universe_requires_as_of() -> None:
    with pytest.raises(SystemExit):
        backfill_cli._build_parser().parse_args(["universe"])
    args = backfill_cli._build_parser().parse_args(["universe", "--as-of", "2024-12-01"])
    assert args.as_of == date(2024, 12, 1)


def test_backfill_universe_backdates_and_verifies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dynamo.domain.models.market import Tier, UniverseMembership

    repo = FakeUniverseRepo()
    # Pre-existing 2026-08 seed rows (approved_by='seed') + one human add.
    for sym in ("AAA", "SPY", "BTC/USD"):
        repo.rows[(sym, date(2026, 8, 1))] = UniverseMembership(
            symbol=sym, tier=Tier.CORE, valid_from=date(2026, 8, 1),
            valid_to=None, reason="core seed", approved_by="seed",
        )
    repo.rows[("HUMAN", date(2026, 8, 5))] = UniverseMembership(
        symbol="HUMAN", tier=Tier.CORE, valid_from=date(2026, 8, 5),
        valid_to=None, reason="approved add", approved_by="user",
    )
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(
        "dynamo.data.universe.seed_symbol_origins",
        lambda *, include_crypto, seed_path="unused": [
            ("AAA", "built-in starter list"),
            ("SPY", "built-in starter list"),
            *([("BTC/USD", "crypto starter list")] if include_crypto else []),
        ],
    )
    rc = backfill_cli.main(["universe", "--as-of", "2024-12-01"])
    out = capsys.readouterr().out
    assert rc == 0
    # Old seed intervals replaced by one backdated open interval per symbol.
    assert sorted(k for k in repo.rows if k[0] != "HUMAN") == [
        ("AAA", date(2024, 12, 1)),
        ("BTC/USD", date(2024, 12, 1)),
        ("SPY", date(2024, 12, 1)),
    ]
    row = repo.rows[("AAA", date(2024, 12, 1))]
    assert row.approved_by == "replay-backdate"
    assert row.valid_to is None
    assert "replay backfill campaign" in row.reason
    # Human-approved rows are never touched.
    assert ("HUMAN", date(2026, 8, 5)) in repo.rows
    assert "verify universe_at(as-of=2024-12-01): 3/3" in out
    assert "OK" in out


def test_backfill_universe_is_idempotent_and_rebases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeUniverseRepo()
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(
        "dynamo.data.universe.seed_symbol_origins",
        lambda *, include_crypto, seed_path="unused": [("AAA", "built-in starter list")],
    )
    assert backfill_cli.main(["universe", "--as-of", "2024-12-01"]) == 0
    assert backfill_cli.main(["universe", "--as-of", "2024-12-01"]) == 0
    # Same as-of twice -> still exactly one interval.
    assert list(repo.rows) == [("AAA", date(2024, 12, 1))]
    # Different as-of -> replaced (single interval again, new valid_from).
    assert backfill_cli.main(["universe", "--as-of", "2025-06-01"]) == 0
    assert list(repo.rows) == [("AAA", date(2025, 6, 1))]


def test_backfill_universe_excludes_crypto_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeUniverseRepo()
    seen: list[bool] = []

    def fake_origins(*, include_crypto: bool, seed_path: str = "unused") -> list[Any]:
        seen.append(include_crypto)
        return [("AAA", "built-in starter list")]

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(crypto_enabled=False)
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr("dynamo.data.universe.seed_symbol_origins", fake_origins)
    assert backfill_cli.main(["universe", "--as-of", "2024-12-01"]) == 0
    assert seen == [False]


# --------------------------------------------------------------------------
# backfill news
# --------------------------------------------------------------------------


def test_backfill_news_parses_args() -> None:
    with pytest.raises(SystemExit):
        backfill_cli._build_parser().parse_args(
            ["news", "--start", "2026-01-01", "--end", "2026-02-01"]
        )
    args = backfill_cli._build_parser().parse_args(
        ["news", "--start", "2026-01-01", "--end", "2026-02-01", "--universe"]
    )
    assert args.universe is True and args.start == date(2026, 1, 1)


def test_backfill_news_fails_loudly_without_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        backfill_cli, "_repo", lambda url: pytest.fail("must not touch the database")
    )
    rc = backfill_cli.main(
        ["news", "--start", "2026-01-01", "--end", "2026-02-01", "--symbols", "AAA"]
    )
    assert rc == 1
    assert "FINNHUB_API_KEY" in capsys.readouterr().out


def test_backfill_news_upserts_range_and_skips_crypto(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = FakeRepo()
    calls: list[tuple[list[str], date, date]] = []

    class FakeFinnhub:
        def news_range(self, symbols: Any, start: date, end: date) -> Any:
            calls.append((list(symbols), start, end))
            yield {"id": f"finnhub:{symbols[0]}"}

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(finnhub_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_finnhub", lambda s: FakeFinnhub())
    rc = backfill_cli.main(
        [
            "news", "--start", "2026-01-01", "--end", "2026-08-01",
            "--symbols", "AAA", "BTC/USD", "BBB",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == [
        (["AAA"], date(2026, 1, 1), date(2026, 8, 1)),
        (["BBB"], date(2026, 1, 1), date(2026, 8, 1)),
    ]
    assert len(repo.news) == 2
    assert backfill_cli.CRYPTO_NEWS_SKIP_MESSAGE in out
    assert "BTC/USD" in out


# --------------------------------------------------------------------------
# backfill transcripts (guarded optional extra)
# --------------------------------------------------------------------------


class FakeDefeatBeta:
    def __init__(self, available: bool, transcripts: list[Any] | None = None) -> None:
        self.available = available
        self._transcripts = transcripts or []
        self.calls: list[tuple[list[str], date]] = []

    def transcripts(self, symbols: Any, *, since: date) -> list[Any]:
        self.calls.append((list(symbols), since))
        return list(self._transcripts)


def test_backfill_transcripts_parses_args() -> None:
    with pytest.raises(SystemExit):
        backfill_cli._build_parser().parse_args(["transcripts"])
    args = backfill_cli._build_parser().parse_args(
        ["transcripts", "--symbols", "AAA", "--since", "2024-01-01"]
    )
    assert args.symbols == ["AAA"] and args.since == date(2024, 1, 1)


def test_backfill_transcripts_missing_extra_is_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backfill_cli, "_defeatbeta", lambda: FakeDefeatBeta(False))
    monkeypatch.setattr(
        backfill_cli, "_repo", lambda url: pytest.fail("must not touch the database")
    )
    rc = backfill_cli.main(["transcripts", "--symbols", "AAA"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "transcripts" in out and "extra" in out
    assert "--allow-missing-extra" in out


def test_backfill_transcripts_allow_missing_extra_skips(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backfill_cli, "_defeatbeta", lambda: FakeDefeatBeta(False))
    monkeypatch.setattr(
        backfill_cli, "_repo", lambda url: pytest.fail("must not touch the database")
    )
    rc = backfill_cli.main(
        ["transcripts", "--symbols", "AAA", "--allow-missing-extra"]
    )
    assert rc == 0
    assert "skipping" in capsys.readouterr().out


def test_backfill_transcripts_upserts_and_links(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dynamo.domain.models.market import Transcript

    transcript = Transcript(
        symbol="AAA",
        event_date=date(2025, 5, 1),
        source="defeatbeta",
        period="2025Q1",
        content="CEO: strong quarter.",
    )
    provider = FakeDefeatBeta(True, [transcript])
    repo = FakeRepo()
    monkeypatch.setattr(backfill_cli, "_defeatbeta", lambda: provider)
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    rc = backfill_cli.main(
        ["transcripts", "--symbols", "AAA", "BTC/USD", "--since", "2024-06-01"]
    )
    assert rc == 0
    assert provider.calls == [(["AAA"], date(2024, 6, 1))]  # crypto filtered out
    assert repo.transcripts == [transcript]
    assert repo.linked == [("AAA", date(2025, 5, 1), transcript.ref)]
    assert "skipped" in capsys.readouterr().out  # crypto skip note


# --------------------------------------------------------------------------
# backfill crypto-bars
# --------------------------------------------------------------------------


def test_backfill_crypto_bars_parses_args() -> None:
    args = backfill_cli._build_parser().parse_args(
        ["crypto-bars", "--start", "2024-01-01", "--end", "2026-08-01"]
    )
    assert args.symbols is None
    args = backfill_cli._build_parser().parse_args(
        [
            "crypto-bars", "--start", "2024-01-01", "--end", "2026-08-01",
            "--symbols", "BTC/USD",
        ]
    )
    assert args.symbols == ["BTC/USD"]


def test_backfill_crypto_bars_defaults_to_settings_universe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @dataclass
    class FakeBar:
        symbol: str

    repo = FakeRepo()
    calls: list[tuple[list[str], date, date]] = []

    class FakeCrypto:
        def daily_bars(self, symbols: Any, start: date, end: date) -> Any:
            calls.append((list(symbols), start, end))
            for sym in symbols:
                yield FakeBar(sym)
                yield FakeBar(sym)

    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_alpaca_crypto", lambda s: FakeCrypto())
    rc = backfill_cli.main(
        ["crypto-bars", "--start", "2024-01-01", "--end", "2026-08-01"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == [(["BTC/USD", "ETH/USD"], date(2024, 1, 1), date(2026, 8, 1))]
    assert len(repo.bars) == 4
    assert "BTC/USD 2024-01-01 -> 2026-08-01: 2 bars" in out


def test_backfill_crypto_bars_explicit_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    seen: list[str] = []

    class FakeCrypto:
        def daily_bars(self, symbols: Any, start: date, end: date) -> Any:
            seen.extend(symbols)
            return iter(())

    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_alpaca_crypto", lambda s: FakeCrypto())
    rc = backfill_cli.main(
        [
            "crypto-bars", "--start", "2024-01-01", "--end", "2026-08-01",
            "--symbols", "SOL/USD",
        ]
    )
    assert rc == 0
    assert seen == ["SOL/USD"]


# --------------------------------------------------------------------------
# fundamentals --seed-ciks
# --------------------------------------------------------------------------


def test_backfill_fundamentals_seed_ciks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = FakeRepo()

    class FakeEdgar:
        def company_ticker_map(self) -> dict[str, str]:
            return {"AAA": "0000000001", "ZZZ": "0000000009"}

        def company_facts(self, cik: str) -> Any:
            yield {"cik": cik}

    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_edgar", lambda s: FakeEdgar())
    monkeypatch.setattr(
        backfill_cli, "_cik_map", lambda repo, symbols: {"AAA": "0000000001"}
    )
    rc = backfill_cli.main(["fundamentals", "--symbols", "AAA", "NOMAP", "--seed-ciks"])
    out = capsys.readouterr().out
    assert rc == 0
    assert repo.seeded_ciks == {"AAA": "0000000001"}  # only requested symbols
    assert "no SEC ticker mapping for ['NOMAP']" in out
    assert len(repo.facts) == 1


# --------------------------------------------------------------------------
# replay preflight wiring (Phase 1)
# --------------------------------------------------------------------------


def _preflight_error() -> Any:
    from dynamo.replay.preflight import PreflightCheck, PreflightError, PreflightReport

    report = PreflightReport(
        checks=(PreflightCheck("universe", "error", "universe is EMPTY at 2026-02-02"),)
    )
    return PreflightError(report)


def test_replay_run_parses_skip_preflight() -> None:
    args = replay_cli._build_parser().parse_args(
        [
            "run", "--start", "2026-06-01", "--end", "2026-06-10",
            "--run-label", "x", "--skip-preflight",
        ]
    )
    assert args.skip_preflight is True
    args = replay_cli._build_parser().parse_args(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert args.skip_preflight is False


def test_replay_quant_run_refuses_on_preflight_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import dynamo.replay.preflight as preflight_mod

    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())

    def failing_preflight(engine: Any, config: Any) -> Any:
        raise _preflight_error()

    monkeypatch.setattr(preflight_mod, "run_preflight", failing_preflight)
    rc = replay_cli.main(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "preflight FAILED" in out
    assert "universe is EMPTY" in out


def test_replay_skip_preflight_records_skip_in_run_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--skip-preflight never calls run_preflight and hands the runner a
    {"preflight": {"skipped": True}} extra for config_json (NOT ReplayConfig)."""
    import dynamo.replay.preflight as preflight_mod
    import dynamo.replay.runner as runner_mod

    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())
    monkeypatch.setattr(
        preflight_mod,
        "run_preflight",
        lambda engine, config: pytest.fail("preflight must not run when skipped"),
    )
    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, progress: Any = None) -> Any:
            @dataclass
            class R:
                run_id: str = "replay-fake"
                status: str = "completed"
                sessions: tuple = ()
                decisions: tuple = ()
                fills: tuple = ()
                experiences: tuple = ()
                equity_curve: tuple = ()

            return R()

    monkeypatch.setattr(runner_mod, "ReplayRunner", FakeRunner)
    rc = replay_cli.main(
        [
            "run", "--start", "2026-06-01", "--end", "2026-06-10",
            "--run-label", "x", "--skip-preflight",
        ]
    )
    assert rc == 0
    assert captured["run_config_extras"] == {"preflight": {"skipped": True}}
    # run_id stability: the skip lives in metadata, never in the config.
    assert "skip" not in captured["config"].model_dump_json()
    assert "SKIPPED" in capsys.readouterr().out


def test_replay_committee_preflight_runs_before_any_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Committee mode preflights right after config construction — a failure
    exits before the budget, meter, or Anthropic client are ever built."""
    import dynamo.replay.preflight as preflight_mod

    class KeyedSettings:
        anthropic_api_key = "k"
        research_backend = "api"
        model_screen = "claude-haiku-4-5"
        model_analyst = "claude-sonnet-5"
        model_arbiter = "claude-opus-5"

    monkeypatch.setattr("dynamo.config.settings.Settings", lambda: KeyedSettings())
    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())

    def failing_preflight(engine: Any, config: Any) -> Any:
        raise _preflight_error()

    monkeypatch.setattr(preflight_mod, "run_preflight", failing_preflight)
    rc = replay_cli.main(
        [
            "run", "--start", "2026-06-01", "--end", "2026-06-10",
            "--mode", "committee", "--run-label", "x", "--max-usd", "5.0",
        ]
    )
    assert rc == 1
    assert "preflight FAILED" in capsys.readouterr().out


def test_estimate_surfaces_model_knowledge_gate_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A June-2026 window sits inside opus-5's documented training window:
    the estimate surfaces BLOCKED (zero API calls, zero spend). A window
    safely past every cutoff + margin reads OK."""
    rc = replay_cli.main(["estimate", "--start", "2026-06-01", "--end", "2026-06-10"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model knowledge gate" in out
    assert "claude-opus-5: cutoff 2026-05-31 [BLOCKS]" in out
    assert "gate verdict:        BLOCKED" in out
    assert "--acknowledge-model-knowledge" in out

    rc = replay_cli.main(["estimate", "--start", "2027-01-04", "--end", "2027-01-15"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gate verdict:        OK" in out


def test_estimate_macro_digest_overhead_included_unless_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def total(extra: list[str]) -> float:
        rc = replay_cli.main(
            ["estimate", "--start", "2026-06-01", "--end", "2026-06-10", *extra]
        )
        assert rc == 0
        out = capsys.readouterr().out
        line = next(ln for ln in out.splitlines() if "projected total:" in ln)
        return float(line.split("$")[-1])

    with_macro = total([])
    without = total(["--no-macro-digest"])
    assert with_macro > without


def test_run_parser_accepts_phase4_flags() -> None:
    args = replay_cli._build_parser().parse_args(
        [
            "run",
            "--start", "2026-06-01",
            "--end", "2026-06-10",
            "--run-label", "x",
            "--acknowledge-model-knowledge",
            "--no-macro-digest",
            "--no-within-run-learning",
        ]
    )
    assert args.acknowledge_model_knowledge is True
    assert args.no_macro_digest is True
    assert args.no_within_run_learning is True
    defaults = replay_cli._build_parser().parse_args(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert defaults.acknowledge_model_knowledge is False
    assert defaults.no_macro_digest is False


def test_main_sets_offline_env_before_any_embedder_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    import os

    rc = replay_cli.main(["estimate", "--start", "2027-01-04", "--end", "2027-01-08"])
    assert rc == 0
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_backfill_news_retries_on_rate_limit_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finnhub free tier 429s mid-campaign: honor Retry-After, sleep, retry
    the symbol (idempotent upserts), and only then move on."""
    from dynamo.data.errors import RateLimited

    repo = FakeRepo()
    attempts: list[str] = []
    sleeps: list[float] = []

    class FlakyFinnhub:
        def news_range(self, symbols: Any, start: date, end: date) -> Any:
            attempts.append(symbols[0])
            if attempts.count(symbols[0]) == 1 and symbols[0] == "AAA":
                raise RateLimited("finnhub", "429", retry_after_s=7.0)
            yield {"id": f"finnhub:{symbols[0]}"}

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(finnhub_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_finnhub", lambda s: FlakyFinnhub())
    monkeypatch.setattr("dynamo.cli.backfill.time.sleep", sleeps.append)
    rc = backfill_cli.main(
        ["news", "--start", "2026-01-01", "--end", "2026-08-16", "--symbols", "AAA", "BBB"]
    )
    assert rc == 0
    assert attempts == ["AAA", "AAA", "BBB"]  # retried AAA once, then moved on
    assert sleeps == [7.0]  # honored Retry-After
    assert len(repo.news) == 2


def test_backfill_news_gives_up_after_retries_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dynamo.data.errors import RateLimited

    repo = FakeRepo()

    class AlwaysLimited:
        def news_range(self, symbols: Any, start: date, end: date) -> Any:
            raise RateLimited("finnhub", "429")

    monkeypatch.setattr(
        backfill_cli, "_settings", lambda: FakeSettings(finnhub_api_key="k")
    )
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_finnhub", lambda s: AlwaysLimited())
    monkeypatch.setattr("dynamo.cli.backfill.time.sleep", lambda s: None)
    rc = backfill_cli.main(
        ["news", "--start", "2026-01-01", "--end", "2026-08-16", "--symbols", "AAA"]
    )
    out = capsys.readouterr().out
    assert rc == 1  # incomplete backfill is a loud nonzero exit
    assert "INCOMPLETE" in out and "AAA" in out


def test_backfill_fundamentals_skips_symbols_without_companyfacts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ETF trusts (DIA/SPY/QQQ) sit in SEC's ticker map but 404 on
    companyfacts: skip loudly per symbol, keep going, exit 0."""
    from dynamo.data.errors import ProviderDown

    repo = FakeRepo()

    class FakeEdgar:
        def company_ticker_map(self) -> dict[str, str]:
            return {"AAA": "0000000001", "DIA": "0000000002"}

        def company_facts(self, cik: str) -> Any:
            if cik == "0000000002":
                raise ProviderDown("edgar", f"404 for CIK {cik}")
            yield {"cik": cik}

    monkeypatch.setattr(backfill_cli, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(backfill_cli, "_repo", lambda url: repo)
    monkeypatch.setattr(backfill_cli, "_edgar", lambda s: FakeEdgar())
    monkeypatch.setattr(
        backfill_cli,
        "_cik_map",
        lambda repo, symbols: {"AAA": "0000000001", "DIA": "0000000002"},
    )
    rc = backfill_cli.main(["fundamentals", "--symbols", "AAA", "DIA", "--seed-ciks"])
    out = capsys.readouterr().out
    assert rc == 0
    assert len(repo.facts) == 1  # AAA landed despite DIA's 404
    assert "DIA" in out and "skipped" in out


# --------------------------------------------------------------------------
# committee research-backend selection (ADR-0014): api vs claude_code
# --------------------------------------------------------------------------


def _backend_settings(tmp_path: Any, **overrides: Any) -> Any:
    """Real Settings with an executable fake `claude` binary on disk."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    kwargs: dict[str, Any] = {
        "ANTHROPIC_API_KEY": "test-key",
        "CLAUDE_CODE_BINARY": str(binary),
    }
    kwargs.update(overrides)
    return RealSettings(**kwargs)


def _committee_harness(
    monkeypatch: pytest.MonkeyPatch,
    settings: Any,
    *,
    run_row: dict[str, Any] | None = None,
    verdict_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Monkeypatch every seam behind _cmd_run_committee; capture the kwargs
    handed to CommitteeReplay. NO network, NO database, NO CLI processes."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr("dynamo.config.settings.Settings", lambda: settings)
    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())

    class FakeReplayRepo:
        def __init__(self, engine: Any) -> None:
            pass

        def get_run(self, run_id: str) -> dict[str, Any] | None:
            return run_row

        def verdicts_for(self, run_id: str) -> list[dict[str, Any]]:
            return list(verdict_rows or [])

        def record_spend(self, run_id: str, delta_usd: float) -> None:
            pass

    monkeypatch.setattr(
        "dynamo.persistence.repositories.replay.ReplayRepository", FakeReplayRepo
    )
    monkeypatch.setattr(
        "dynamo.persistence.repositories.experience.ExperienceRepository",
        lambda engine: object(),
    )

    class FakeEmbedder:
        backend = "hash-fallback"

    monkeypatch.setattr("dynamo.llm.embeddings.LocalEmbedder", FakeEmbedder)

    class FakeCommitteeReplay:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, progress: Any = None) -> Any:
            @dataclass
            class R:
                run_id: str = "replay-fake"
                status: str = "completed"
                sessions: tuple = ()
                deep_dives: tuple = ()
                experiences: tuple = ()
                spent_usd: float = 0.0
                cap_usd: float = 5.0

            return R()

    monkeypatch.setattr("dynamo.replay.committee.CommitteeReplay", FakeCommitteeReplay)
    return captured


def _committee_argv(*extra: str) -> list[str]:
    return [
        "run",
        "--start", "2026-03-16",
        "--end", "2026-03-20",
        "--mode", "committee",
        "--run-label", "x",
        "--max-usd", "5.0",
        "--skip-preflight",
        *extra,
    ]


def test_committee_claude_code_backend_selected_from_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """RESEARCH_BACKEND=claude_code routes the committee through the CLI
    client with the claude_code tier models (fable arbiter), no flag needed."""
    from dynamo.llm.claude_code_client import ClaudeCodeResearchModel

    settings = _backend_settings(tmp_path, RESEARCH_BACKEND="claude_code")
    captured = _committee_harness(monkeypatch, settings)
    rc = replay_cli.main(_committee_argv())
    assert rc == 0
    assert isinstance(captured["research_model"], ClaudeCodeResearchModel)
    assert captured["tier_models"] == {
        "screen": "claude-haiku-4-5",
        "analyst": "claude-sonnet-5",
        "arbiter": "claude-fable-5",
    }
    assert captured["run_config_extras"]["research_backend"] == "claude_code"
    out = capsys.readouterr().out
    assert "backend=claude_code" in out
    assert "flat-rate subscription" in out and "metered cost $0" in out
    assert "effectively inert" in out


def test_committee_backend_flag_overrides_settings_both_ways(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    from dynamo.llm.anthropic_client import AnthropicClient
    from dynamo.llm.claude_code_client import ClaudeCodeResearchModel

    # Settings say api; --backend claude_code wins.
    settings = _backend_settings(tmp_path, RESEARCH_BACKEND="api")
    captured = _committee_harness(monkeypatch, settings)
    assert replay_cli.main(_committee_argv("--backend", "claude_code")) == 0
    assert isinstance(captured["research_model"], ClaudeCodeResearchModel)
    assert captured["tier_models"]["arbiter"] == "claude-fable-5"
    capsys.readouterr()

    # Settings say claude_code; --backend api wins (and uses the API tiers).
    settings = _backend_settings(tmp_path, RESEARCH_BACKEND="claude_code")
    captured = _committee_harness(monkeypatch, settings)
    assert replay_cli.main(_committee_argv("--backend", "api")) == 0
    assert isinstance(captured["research_model"], AnthropicClient)
    assert captured["tier_models"]["arbiter"] == "claude-opus-5"
    assert "backend=api" in capsys.readouterr().out


def test_committee_api_backend_default_regression(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """Default settings (research_backend='api') keep the AnthropicClient
    path byte-for-byte: API tier models, no claude_code note printed."""
    from dynamo.llm.anthropic_client import AnthropicClient

    settings = _backend_settings(tmp_path)
    assert settings.research_backend == "api"
    captured = _committee_harness(monkeypatch, settings)
    rc = replay_cli.main(_committee_argv())
    assert rc == 0
    assert isinstance(captured["research_model"], AnthropicClient)
    assert captured["tier_models"] == {
        "screen": "claude-haiku-4-5",
        "analyst": "claude-sonnet-5",
        "arbiter": "claude-opus-5",
    }
    assert captured["resume_skip_days"] == frozenset()
    out = capsys.readouterr().out
    assert "backend=api" in out
    assert "flat-rate subscription" not in out


def test_committee_claude_code_refuses_without_binary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    settings = RealSettings(
        ANTHROPIC_API_KEY="",
        RESEARCH_BACKEND="claude_code",
        CLAUDE_CODE_BINARY=str(tmp_path / "no-such-claude"),
    )
    captured = _committee_harness(monkeypatch, settings)
    rc = replay_cli.main(_committee_argv())
    assert rc == 2
    out = capsys.readouterr().out
    assert "claude" in out and "binary" in out
    assert captured == {}  # never reached CommitteeReplay construction


def test_committee_resume_skips_days_with_persisted_verdicts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """A re-run of the same deterministic run id hands CommitteeReplay the
    days that already have persisted verdicts (deduplicated) and resumes the
    durable spend — the incremental-resume contract."""
    settings = _backend_settings(tmp_path, RESEARCH_BACKEND="claude_code")
    d1, d2 = date(2026, 3, 16), date(2026, 3, 17)
    captured = _committee_harness(
        monkeypatch,
        settings,
        run_row={"spent_usd": 1.25},
        verdict_rows=[
            {"as_of": d1, "symbol": "AAA"},
            {"as_of": d1, "symbol": "BBB"},
            {"as_of": d2, "symbol": "AAA"},
        ],
    )
    rc = replay_cli.main(_committee_argv())
    assert rc == 0
    assert captured["resume_skip_days"] == frozenset({d1, d2})
    assert captured["budget"].spent_usd == pytest.approx(1.25)
    out = capsys.readouterr().out
    assert "skipping 2 day(s) with persisted verdicts" in out
    assert "resuming from $1.25 spent" in out


# --------------------------------------------------------------------------
# estimate: claude_code backend (flat-rate — no dollar projection)
# --------------------------------------------------------------------------


def test_estimate_claude_code_prints_flat_rate_and_fable_cutoff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """claude_code estimate: backend named, NO dollar projection, the gate
    resolves the claude_code tiers' cutoffs — a March-2026 start clears
    fable/sonnet (cutoffs 2026-01-31 + 30d margin) and reads ALLOWED."""
    settings = RealSettings(ANTHROPIC_API_KEY="", RESEARCH_BACKEND="claude_code")
    monkeypatch.setattr("dynamo.config.settings.Settings", lambda: settings)
    rc = replay_cli.main(["estimate", "--start", "2026-03-16", "--end", "2026-03-27"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "backend:             claude_code" in out
    assert "flat-rate subscription — metered cost $0" in out
    assert "consumes Claude plan usage windows" in out
    assert "effectively inert" in out
    assert "projected total:" not in out
    assert "per deep dive:" not in out
    assert "suggested --max-usd" not in out
    # The gate resolved the claude_code tiers — fable included — and ALLOWED.
    assert "claude-fable-5: cutoff 2026-01-31 [ok]" in out
    assert "claude-sonnet-5: cutoff 2026-01-31 [ok]" in out
    assert "claude-opus-5" not in out
    assert "gate verdict:        OK" in out


def test_estimate_backend_flag_overrides_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = RealSettings(ANTHROPIC_API_KEY="", RESEARCH_BACKEND="api")
    monkeypatch.setattr("dynamo.config.settings.Settings", lambda: settings)
    rc = replay_cli.main(
        [
            "estimate",
            "--start", "2026-03-16",
            "--end", "2026-03-27",
            "--backend", "claude_code",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "backend:             claude_code" in out and "projected total:" not in out

    # And the api projection is unchanged when selected explicitly.
    rc = replay_cli.main(
        ["estimate", "--start", "2026-03-16", "--end", "2026-03-27", "--backend", "api"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "backend:             api" in out
    assert "projected total:" in out and "claude-opus-5" in out


def test_replay_ballast_flag_parses_and_defaults_off() -> None:
    parser = replay_cli._build_parser()
    args = parser.parse_args(
        ["run", "--start", "2026-06-01", "--end", "2026-06-10", "--run-label", "x"]
    )
    assert args.ballast == ""
    args = parser.parse_args(
        [
            "run",
            "--start", "2026-06-01",
            "--end", "2026-06-10",
            "--run-label", "x",
            "--ballast", "SPY",
        ]
    )
    assert args.ballast == "SPY"


def test_replay_committee_mode_refuses_ballast(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Committee runs produce verdicts, not portfolios: --ballast is an error
    there (exit 2), pointing at --mode portfolio instead."""
    rc = replay_cli.main(
        [
            "run",
            "--start", "2026-06-01",
            "--end", "2026-06-10",
            "--mode", "committee",
            "--run-label", "x",
            "--max-usd", "5",
            "--ballast", "SPY",
        ]
    )
    assert rc == 2
    assert "--mode portfolio" in capsys.readouterr().err
