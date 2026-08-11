"""Daily report builder + renderer against an in-memory fake ReportSource."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from dynamo.reporting.daily import build_daily_report, cache_savings_usd, render_markdown

DAY = date(2026, 8, 4)  # a Tuesday

SECTION_HEADERS = [
    "## At a glance",
    "## Decisions",
    "## Trades executed",
    "## Claude usage & cost",
    "## Watchlist movers",
    "## Surveillance & promotions",
    "## Alerts & issues",
    "## Learning",
]


class FakeSource:
    """ReportSource backed by a plain dict; anything unset is empty."""

    def __init__(self, **data: Any) -> None:
        self._data = data

    def _get(self, key: str, default: Any) -> Any:
        return self._data.get(key, default)

    def equity_points(self, day: date) -> list[dict[str, Any]]:
        return self._get("equity_points", [])

    def open_position_counts(self) -> dict[str, int]:
        return self._get("open_position_counts", {})

    def regime_on(self, day: date) -> dict[str, Any] | None:
        return self._get("regime", None)

    def ladder_on(self, day: date) -> dict[str, Any] | None:
        return self._get("ladder", None)

    def verdicts_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("verdicts", [])

    def watchlist_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("watchlist", [])

    def watchlist_prior(self, day: date) -> list[dict[str, Any]]:
        return self._get("watchlist_prior", [])

    def equity_trades_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("equity_trades", [])

    def options_trades_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("options_trades", [])

    def pm_trades_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("pm_trades", [])

    def cost_entries_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("cost_entries", [])

    def cost_month_to_date(self, day: date) -> float:
        return self._get("mtd", 0.0)

    def monthly_budget(self) -> float:
        return self._get("budget", 90.0)

    def flags_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("flags", [])

    def universe_changes_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("universe_changes", [])

    def pending_universe_recommendations(self) -> list[str]:
        return self._get("pending_recommendations", [])

    def unacked_alerts(self) -> list[dict[str, Any]]:
        return self._get("alerts", [])

    def open_quarantines(self) -> list[dict[str, Any]]:
        return self._get("quarantines", [])

    def batch_failures_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("batch_failures", [])

    def reconciliation_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("reconciliation", [])

    def reviews_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("reviews", [])

    def calibration_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("calibration", [])

    def proposals_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("proposals", [])

    def meta_scores_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("meta_scores", [])

    def learning_metrics_on(self, day: date) -> list[dict[str, Any]]:
        return self._get("learning_metrics", [])

    def open_proposal_count(self) -> int:
        return self._get("open_proposals", 0)


def _verdict(symbol: str, action: str, conviction: float, **payload: Any) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "action": action,
        "conviction": conviction,
        "disagreement_score": payload.pop("disagreement", 0.2),
        "payload": {
            "thesis": payload.pop("thesis", f"{symbol} looks attractive"),
            "risks": payload.pop("risks", ("Macro turns risk-off",)),
            "expected_upside_pct": payload.pop("up", 12.0),
            "expected_downside_pct": payload.pop("down", 6.0),
        },
    }


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 8, 4, hour, minute, tzinfo=UTC)


def full_day_source() -> FakeSource:
    return FakeSource(
        equity_points=[
            {"as_of": date(2026, 8, 3), "equity": 100_800.0, "cash": 24_000.0},
            {"as_of": DAY, "equity": 102_000.0, "cash": 25_000.0},
        ],
        open_position_counts={"equities": 7, "options": 2, "kalshi": 1},
        regime={"as_of": DAY, "regime": "neutral"},
        ladder={"as_of": DAY, "rung": "normal", "drawdown": -0.012},
        verdicts=[
            _verdict("AAPL", "buy", 0.72, thesis="Services growth keeps compounding",
                     risks=("Hardware cycle rolls over",), up=18.0, down=9.0,
                     disagreement=0.35),
            _verdict("MSFT", "trim", 0.55, thesis="Position outgrew its risk budget"),
            _verdict("NVDA", "hold", 0.50),
            _verdict("TSM", "add", 0.80, thesis="Foundry pricing power is intact"),
            _verdict("ZZZ", "no_action", 0.30),
        ],
        watchlist=[
            {"sleeve": "long_term", "symbol": "AAPL", "rank": 1, "blend_score": 0.82,
             "badges": ("conflict",)},
            {"sleeve": "long_term", "symbol": "TSM", "rank": 2, "blend_score": 0.79,
             "badges": ()},
            {"sleeve": "tactical", "symbol": "SMCI", "rank": 1, "blend_score": 0.66,
             "badges": ()},
        ],
        watchlist_prior=[
            {"sleeve": "long_term", "symbol": "AAPL", "rank": 3, "blend_score": 0.70,
             "badges": ()},
            {"sleeve": "long_term", "symbol": "TSM", "rank": 2, "blend_score": 0.75,
             "badges": ()},
        ],
        equity_trades=[
            {"filled_at": _ts(14, 31), "symbol": "AAPL", "side": "buy", "qty": 10.0,
             "fill_price": 210.55, "slippage_bps": 3.2, "sleeve": "long_term"},
        ],
        options_trades=[
            {"filled_at": _ts(14, 35), "occ_symbol": "AAPL260918C00230000",
             "underlying": "AAPL", "side": "buy", "contracts": 2, "price": 4.15},
        ],
        pm_trades=[
            {"filled_at": _ts(13, 5), "market_id": "KXFED-26SEP-CUT", "side": "yes",
             "contracts": 40, "price_cents": 42.0},
        ],
        cost_entries=[
            {"model_id": "claude-sonnet-5", "call_kind": "dossier", "input_tokens": 500_000,
             "output_tokens": 40_000, "cache_read_tokens": 1_000_000, "cost_usd": 2.4,
             "batch": True},
            {"model_id": "claude-haiku-4-5", "call_kind": "triage", "input_tokens": 200_000,
             "output_tokens": 10_000, "cache_read_tokens": 0, "cost_usd": 0.25,
             "batch": False},
            {"model_id": "claude-opus-5", "call_kind": "arbiter", "input_tokens": 100_000,
             "output_tokens": 20_000, "cache_read_tokens": 400_000, "cost_usd": 1.2,
             "batch": False},
        ],
        mtd=42.5,
        budget=90.0,
        flags=[
            {"detector": "volume_spike", "symbol": "GME", "severity": 0.8,
             "triage_verdict": "dismissed"},
            {"detector": "volume_spike", "symbol": "AMC", "severity": 0.7,
             "triage_verdict": "escalated"},
        ],
        universe_changes=[
            {"symbol": "GME", "tier": 3, "reason": "volume burst", "kind": "promoted"},
        ],
        pending_recommendations=["Add SMCI to Tier 1 (core universe)"],
        alerts=[{"severity": "warn", "name": "provider_gap",
                 "message": "Tiingo missing 2 symbols"}],
        quarantines=[{"scope": "symbol", "key": "XYZ", "reason": "stale price"}],
        batch_failures=[{"batch_id": "b-9", "stage": "dossier", "status": "expired"}],
        reconciliation=[{"scope": "equities", "result": "clean", "divergences": []}],
        reviews=[{"symbol": "AAPL", "lesson": "Exited too early on a strong trend"}],
        calibration=[{"scope_id": "arbiter", "brier": 0.18, "n": 24}],
        proposals=[{"kind": "threshold_change", "target": "slippage_model",
                    "expected_effect": "Reduce modeled slippage error"}],
        meta_scores=[
            {"symbol": "AAPL", "conviction": 0.72, "effective_probability": 0.61,
             "model_version": "meta-gradient_boosting-20260810"},
            {"symbol": "TSM", "conviction": 0.80, "effective_probability": 0.74,
             "model_version": "meta-gradient_boosting-20260810"},
        ],
        learning_metrics=[
            {"metric": "learning.brier", "value": 0.21},
            {"metric": "learning.success_rate", "value": 0.55},
            {"metric": "learning.abstention_rate", "value": 0.4},
            {"metric": "learning.ev_per_decision", "value": 1.25},
            {"metric": "learning.n_scored", "value": 20.0},
        ],
        open_proposals=3,
    )


def test_full_day_report_renders_every_section() -> None:
    md = render_markdown(build_daily_report(full_day_source(), DAY))
    assert md.startswith("# Dynamo Daily Report — 2026-08-04 (Tuesday)")
    for header in SECTION_HEADERS:
        assert header in md, header
    assert "None today." not in md


def test_glance_numbers_and_abstention_note() -> None:
    report = build_daily_report(full_day_source(), DAY)
    g = report.glance
    assert g.equity == pytest.approx(102_000.0)
    assert g.day_change_usd == pytest.approx(1_200.0)
    assert g.day_change_pct == pytest.approx(1.19, abs=0.01)
    assert g.cash_pct == pytest.approx(24.5, abs=0.1)
    assert (g.names_reviewed, g.names_acted_on) == (5, 3)
    md = render_markdown(report)
    assert "reviewed 5 names, acted on 3" in md
    assert "7 stocks, 2 options, 1 Kalshi event markets" in md


def test_decisions_grouped_and_ordered_by_conviction() -> None:
    report = build_daily_report(full_day_source(), DAY)
    assert [d.symbol for d in report.decisions.buys] == ["TSM", "AAPL"]  # conviction desc
    assert [d.symbol for d in report.decisions.sells] == ["MSFT"]
    assert report.decisions.holds == 2
    md = render_markdown(report)
    assert md.index("### Buys & adds") < md.index("### Trims, sells & avoids")
    assert md.index("TSM — add") < md.index("AAPL — buy")
    assert "Holds / no action: 2 names." in md
    assert "badges: conflict" in md  # joined from the watchlist row
    assert "The main risk: Hardware cycle rolls over." in md


def test_trades_merged_across_sleeves_in_time_order() -> None:
    report = build_daily_report(full_day_source(), DAY)
    assert [t.market for t in report.trades] == ["kalshi", "stocks", "options"]
    assert [t.filled_at.hour for t in report.trades] == [13, 14, 14]
    md = render_markdown(report)
    assert md.index("KXFED-26SEP-CUT") < md.index("| AAPL |") < md.index("AAPL260918C00230000")
    assert "42.00 cents" in md
    assert "+3.2 bps" in md


def test_polymarket_sim_rows_are_labeled_simulated() -> None:
    """ADR-0011: every rendering of the shadow sleeve says SIMULATED."""
    source = FakeSource(
        open_position_counts={"equities": 1, "options": 0, "kalshi": 1, "polymarket_sim": 2},
        pm_trades=[
            {"filled_at": _ts(13, 5), "venue": "kalshi", "market_id": "KXFED-26SEP-CUT",
             "side": "yes", "contracts": 40, "price_cents": 42.0},
            {"filled_at": _ts(13, 6), "venue": "polymarket", "market_id": "514729",
             "side": "yes", "contracts": 20, "price_cents": 44.0},
        ],
    )
    report = build_daily_report(source, DAY)
    assert report.glance.open_polymarket_sim == 2
    by_instrument = {t.instrument: t for t in report.trades}
    assert by_instrument["KXFED-26SEP-CUT"].market == "kalshi"
    assert by_instrument["514729"].market == "polymarket-sim"
    assert by_instrument["514729"].sleeve == "polymarket (SIMULATED)"
    md = render_markdown(report)
    assert "2 Polymarket shadow (SIMULATED — tracked internally, no real orders sent)" in md
    assert "Polymarket rows are SIMULATED — tracked internally, no real orders sent." in md


def test_kalshi_only_glance_line_omits_sim_sleeve() -> None:
    md = render_markdown(build_daily_report(full_day_source(), DAY))
    assert "7 stocks, 2 options, 1 Kalshi event markets" in md
    assert "SIMULATED" not in md  # nothing simulated happened, nothing rendered


def test_cost_math_cache_and_batch_savings() -> None:
    report = build_daily_report(full_day_source(), DAY)
    c = report.cost
    # Sonnet: 1M cache reads * $3/MTok * 0.9 saved, halved by the batch discount
    # = $1.35; Opus: 0.4M * $5/MTok * 0.9 = $1.80.
    assert c.cache_savings_usd == pytest.approx(1.35 + 1.80)
    assert c.batch_savings_usd == pytest.approx(2.4)  # batches bill at half price
    assert c.total_today_usd == pytest.approx(2.4 + 0.25 + 1.2)
    sonnet = next(t for t in c.tiers if t.model_id == "claude-sonnet-5")
    assert sonnet.label == "Sonnet (analysts)"
    assert sonnet.cache_read_share_pct == pytest.approx(100 * 1_000_000 / 1_500_000)
    assert [k.call_kind for k in c.kinds] == ["dossier", "arbiter", "triage"]  # cost desc
    md = render_markdown(report)
    assert "saved about $3.15 today" in md
    assert "$42.50 of the $90.00 budget (47%)" in md


def test_cache_savings_skips_unknown_models() -> None:
    entries = [
        {"model_id": "not-a-model", "cache_read_tokens": 1_000_000, "batch": False},
        {"model_id": "claude-haiku-4-5", "cache_read_tokens": 1_000_000, "batch": False},
    ]
    assert cache_savings_usd(entries) == pytest.approx(0.9)  # $1/MTok * 0.9


def test_watchlist_movers_show_rank_changes() -> None:
    md = render_markdown(build_daily_report(full_day_source(), DAY))
    assert "| long_term | 1 | AAPL | up 2 | 0.82 |" in md
    assert "| tactical | 1 | SMCI | new | 0.66 |" in md


def test_empty_day_is_compact_with_none_today_lines() -> None:
    md = render_markdown(build_daily_report(FakeSource(), DAY))
    for header in SECTION_HEADERS:
        assert header in md, header
    assert md.count("None today.") == 7  # every section except the glance
    assert "no equity snapshot recorded yet" in md
    assert "reviewed 0 names, acted on 0" in md


def test_deterministic_same_inputs_identical_markdown() -> None:
    first = render_markdown(build_daily_report(full_day_source(), DAY))
    second = render_markdown(build_daily_report(full_day_source(), DAY))
    assert first == second


def test_markdown_has_no_raw_json_dumps() -> None:
    for source in (full_day_source(), FakeSource()):
        md = render_markdown(build_daily_report(source, DAY))
        assert "{" not in md and "}" not in md
        assert "```" not in md
        assert "None," not in md  # no repr() of python rows


def test_report_json_round_trips() -> None:
    report = build_daily_report(full_day_source(), DAY)
    payload = report.model_dump(mode="json")
    assert payload["day"] == "2026-08-04"
    restored = type(report).model_validate(payload)
    assert render_markdown(restored) == render_markdown(report)


def test_meta_scores_render_advisory_table() -> None:
    """Plan Phase 3: meta rows appear in Learning, clearly labeled advisory."""
    report = build_daily_report(full_day_source(), DAY)
    assert len(report.learning.meta_scores) == 2
    md = render_markdown(report)
    assert "ADVISORY only, not used for sizing" in md
    assert "| AAPL | 0.72 | 0.61 | meta-gradient_boosting-20260810 |" in md
    assert "| TSM | 0.80 | 0.74 | meta-gradient_boosting-20260810 |" in md


def test_meta_scores_absent_renders_nothing_meta() -> None:
    md = render_markdown(build_daily_report(FakeSource(), DAY))
    assert "Meta-model advisory scores" not in md


def test_learning_metrics_render_headline_lines() -> None:
    """Plan Phase 6: the weekly learning.* headline metrics + open-proposal
    count appear in the Learning section, plain-language and deterministic."""
    report = build_daily_report(full_day_source(), DAY)
    assert report.learning.metrics["learning.brier"] == pytest.approx(0.21)
    assert report.learning.open_proposals == 3
    md = render_markdown(report)
    assert "Learning health — weekly metrics on resolved decisions (over 20 scored" in md
    assert "Brier score (confidence vs reality; lower is better): 0.210" in md
    assert "Success rate (scored decisions that worked out): 55%" in md
    assert "Abstention rate (committee chose to sit out): 40%" in md
    assert "EV per decision (avg realized pnl): +1.25%" in md
    assert "Improvement proposals open (awaiting your review, nothing auto-applies): 3" in md


def test_learning_metrics_absent_render_nothing() -> None:
    md = render_markdown(build_daily_report(FakeSource(), DAY))
    assert "Learning health" not in md
    assert "Improvement proposals open" not in md
