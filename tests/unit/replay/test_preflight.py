"""Replay preflight tests (backtest-campaign Phase 1): every failure mode is
LOUD, the report shape is stable, and the Phase-2/3 hooks behave both before
and after those phases land. NO network, NO database — a fake reads object
stands in for the engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from dynamo.replay import preflight as pf
from dynamo.replay.discovery import DISCOVERY_LOOKBACK_DAYS
from dynamo.replay.gateway import SURVEILLANCE_TIER
from dynamo.replay.runner import (
    REGIME_MACRO_LOOKBACK_DAYS,
    RETURNS_LOOKBACK_DAYS,
    ReplayConfig,
)

START = date(2026, 2, 2)
END = date(2026, 7, 31)
LOOKBACK = START - timedelta(days=RETURNS_LOOKBACK_DAYS)
MACRO_REACH = START - timedelta(days=REGIME_MACRO_LOOKBACK_DAYS)

SYMBOLS = [f"S{i:02d}" for i in range(20)]
# The tier-2 surveillance universe, disjoint from the traded core so a check
# reading the wrong tier is visible rather than coincidentally right.
TIER2_SYMBOLS = [f"W{i:02d}" for i in range(50)]
DISCOVERY_REACH = START - timedelta(days=DISCOVERY_LOOKBACK_DAYS)


@dataclass
class FakeReads:
    """In-memory PreflightReads: healthy by default, degradable per test."""

    universe: list[str] = field(default_factory=lambda: list(SYMBOLS))
    # Tier-2 surveillance universe, queried only when discovery is on.
    surveillance: list[str] = field(default_factory=lambda: list(TIER2_SYMBOLS))
    tier_queries: list[tuple[int, ...]] = field(default_factory=list)
    audit_symbols: list[list[str]] = field(default_factory=list)
    bars_missing_at: dict[date, set[str]] = field(default_factory=dict)
    macro_first: dict[str, date | None] = field(
        default_factory=lambda: {
            s: MACRO_REACH - timedelta(days=30) for s in pf.REQUIRED_MACRO_SERIES
        }
    )
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    bar_queries: list[tuple[date, date]] = field(default_factory=list)
    # Adjustment audit (D5). Healthy default: one flat series per symbol and
    # one recorded split that leaves no step, i.e. properly adjusted data.
    audit_bars: pd.DataFrame | None = None
    splits: set[tuple[str, date]] | None = None
    audit_queries: list[tuple[date, date]] = field(default_factory=list)

    def universe_symbols(self, as_of: date, tiers: tuple[int, ...]) -> list[str]:
        self.tier_queries.append(tuple(tiers))
        if tuple(tiers) == (SURVEILLANCE_TIER,):
            return list(self.surveillance)
        return list(self.universe)

    def symbols_with_bars(
        self, symbols: Sequence[str], start: date, end: date
    ) -> set[str]:
        self.bar_queries.append((start, end))
        missing: set[str] = set()
        for anchor, syms in self.bars_missing_at.items():
            if start <= anchor <= end:
                missing = syms
        return set(symbols) - missing

    def macro_first_observation(
        self, series_ids: Sequence[str], vintage_cutoff: date
    ) -> dict[str, date | None]:
        return {s: self.macro_first.get(s) for s in series_ids}

    def get_run(self, run_id: str) -> Mapping[str, Any] | None:
        return self.runs.get(run_id)

    def bars_for_adjustment_audit(
        self, symbols: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        self.audit_queries.append((start, end))
        self.audit_symbols.append(list(symbols))
        if self.audit_bars is not None:
            return self.audit_bars
        return adjusted_series(SYMBOLS[0])

    def split_ex_dates(
        self, symbols: Sequence[str], start: date, end: date
    ) -> set[tuple[str, date]]:
        return set(self.splits) if self.splits is not None else {(SYMBOLS[0], SPLIT_DAY)}


SPLIT_DAY = date(2026, 3, 4)


def audit_frame(symbol: str, closes: list[float], *, source: str = "alpaca",
                adj: bool = False) -> pd.DataFrame:
    """A stored-bar frame shaped like `bars_for_adjustment_audit`, whose
    third session is SPLIT_DAY."""
    stamps = [datetime.combine(SPLIT_DAY, datetime.min.time(), tzinfo=UTC)
              + timedelta(days=i - 2) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": stamps,
            "close": closes,
            "source": source,
            "adj_close": closes if adj else [None] * len(closes),
        }
    )


def adjusted_series(symbol: str) -> pd.DataFrame:
    """The same 2-for-1 split, correctly back-adjusted: no step at all."""
    return audit_frame(symbol, [50.0, 50.0, 50.0, 50.0])


def unadjusted_series(symbol: str) -> pd.DataFrame:
    """The D1 defect: the price halves on the split ex-date."""
    return audit_frame(symbol, [100.0, 100.0, 50.0, 50.0])


def config(**overrides: Any) -> ReplayConfig:
    payload: dict[str, Any] = {
        "start": START,
        "end": END,
        "mode": "quant",
        "label": "preflight-test",
    }
    payload.update(overrides)
    return ReplayConfig(**payload)


def errors_of(exc: pf.PreflightError) -> dict[str, str]:
    return {c.name: c.detail for c in exc.report.errors}


# -- happy path + report shape ----------------------------------------------


def test_healthy_data_passes_with_quant_model_warning() -> None:
    report = pf.run_preflight(FakeReads(), config())
    assert report.passed
    names = [c.name for c in report.checks]
    assert names == [
        "universe",
        "bar_coverage_start",
        "bar_coverage_lookback",
        "bar_adjustment",
        "macro_history",
        "quant_model",
        "source_run",
    ]
    # quant_model_version unset -> warning recorded, never an error.
    assert [c.name for c in report.warnings] == ["quant_model"]

    payload = report.as_dict()
    assert payload["passed"] is True
    assert len(payload["checks"]) == len(report.checks)
    assert set(payload["checks"][0]) == {"name", "status", "detail"}


def test_bar_coverage_checked_at_start_and_at_lookback() -> None:
    reads = FakeReads()
    pf.run_preflight(reads, config())
    assert len(reads.bar_queries) == 2
    (s1, e1), (s2, e2) = reads.bar_queries
    assert s1 <= START <= e1
    assert s2 <= LOOKBACK <= e2


# -- failure modes (each loud) ----------------------------------------------


def test_empty_universe_is_an_error() -> None:
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(universe=[]), config())
    errs = errors_of(exc.value)
    assert "universe" in errs
    assert "EMPTY" in errs["universe"]
    # bar coverage is not checked against an empty universe
    assert not any(n.startswith("bar_coverage") for n in errs)


def test_bar_coverage_below_95pct_at_start_is_an_error() -> None:
    reads = FakeReads(bars_missing_at={START: set(SYMBOLS[:2])})  # 18/20 = 90%
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    errs = errors_of(exc.value)
    assert "bar_coverage_start" in errs
    assert "95%" in errs["bar_coverage_start"]
    assert "bar_coverage_lookback" not in errs


def test_bar_coverage_below_95pct_at_lookback_is_an_error() -> None:
    reads = FakeReads(bars_missing_at={LOOKBACK: set(SYMBOLS[:2])})
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    assert "bar_coverage_lookback" in errors_of(exc.value)


def test_exactly_95pct_coverage_passes() -> None:
    reads = FakeReads(bars_missing_at={START: set(SYMBOLS[:1])})  # 19/20 = 95%
    report = pf.run_preflight(reads, config())
    assert report.passed


def test_macro_series_missing_or_short_is_an_error() -> None:
    reads = FakeReads(
        macro_first={
            "BAMLH0A0HYM2": None,  # never ingested
            "T10Y3M": MACRO_REACH + timedelta(days=1),  # too shallow
        }
    )
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    detail = errors_of(exc.value)["macro_history"]
    assert "BAMLH0A0HYM2" in detail and "no observations" in detail
    assert "T10Y3M" in detail


# -- quant-model hook (Phase 2) ---------------------------------------------


def test_quant_model_version_set_but_no_loader_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dynamo.research.quant.training as training

    # Pre-Phase-2 state: no loader on the module (future-proofed).
    monkeypatch.delattr(training, "bundle_from_version", raising=False)
    cfg = config()
    object.__setattr__(cfg, "quant_model_version", "campaign-v1")
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(), cfg)
    assert "quant_model" in errors_of(exc.value)


def test_quant_model_loader_failure_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import dynamo.research.quant.training as training

    def bundle_from_version(version: str) -> Any:
        raise LookupError(f"unknown model version {version!r}")

    monkeypatch.setattr(
        training, "bundle_from_version", bundle_from_version, raising=False
    )
    cfg = config()
    object.__setattr__(cfg, "quant_model_version", "campaign-v1")
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(), cfg)
    assert "not loadable" in errors_of(exc.value)["quant_model"]


def test_quant_model_real_loader_unknown_version_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Phase 2 landed: the REAL artifact loader backs the check — an
    unresolvable version (no bundle manifest on disk) fails preflight."""
    import dynamo.research.quant.training as training
    from dynamo.research.quant.bundle import bundle_from_version

    monkeypatch.setattr(
        training,
        "bundle_from_version",
        lambda version: bundle_from_version(version, tmp_path),
    )
    cfg = config()
    object.__setattr__(cfg, "quant_model_version", "never-trained-v1")
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(), cfg)
    detail = errors_of(exc.value)["quant_model"]
    assert "not loadable" in detail
    assert "unknown quant model version" in detail


def _loadable(monkeypatch: pytest.MonkeyPatch, metrics: dict[str, Any] | Exception) -> None:
    """The bundle loads; its manifest metrics are whatever the test says."""
    import dynamo.research.quant.training as training

    monkeypatch.setattr(
        training, "bundle_from_version", lambda version: object(), raising=False
    )

    def read(version: str) -> Any:
        if isinstance(metrics, Exception):
            raise metrics
        return metrics

    monkeypatch.setattr(pf, "_read_bundle_metrics", read)


def _task_report(ships: bool, metric: float = 0.05, baseline: float = 0.09) -> dict[str, Any]:
    return {
        "ships_ok": {"lgbm": ships},
        "best_baseline_pooled": baseline,
        "pooled": {
            "baseline-historic-average": {"fold_metric_mean": baseline, "win_share": 1.0},
            "lgbm": {"fold_metric_mean": metric, "win_share": 0.2},
        },
    }


def test_quant_model_loadable_and_shipping_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _loadable(monkeypatch, {"tasks": {"rank": _task_report(True, 0.12)}})
    cfg = config(quant_model_version="campaign-v9")
    report = pf.run_preflight(FakeReads(), cfg)
    check = next(c for c in report.checks if c.name == "quant_model")
    assert check.status == pf.OK
    assert "ships" in check.detail


def test_quant_model_that_failed_its_ship_gate_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The campaign-v3 shape: the bundle LOADS but lost to its own baseline
    on every task. Loading was the only thing preflight used to assert."""
    _loadable(
        monkeypatch,
        {
            "tasks": {
                "rank": _task_report(False, 0.0486, 0.0929),
                "downside": _task_report(False, 0.1179, 0.0961),
                "vol": _task_report(False, 0.4248, 0.3861),
            }
        },
    )
    cfg = config(quant_model_version="campaign-v3")
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(), cfg)
    detail = errors_of(exc.value)["quant_model"]
    assert "FAILED its own ship gate" in detail
    for task in ("rank", "downside", "vol"):
        assert f"{task}: ships_ok=False" in detail


def test_ship_gate_refusal_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``require_shipping_bundle=False`` (Settings.replay_require_shipping_bundle)
    downgrades the refusal to a loud warning — the run proceeds knowingly."""
    _loadable(monkeypatch, {"tasks": {"rank": _task_report(False)}})
    cfg = config(quant_model_version="campaign-v3")
    report = pf.run_preflight(FakeReads(), cfg, require_shipping_bundle=False)
    check = next(c for c in report.checks if c.name == "quant_model")
    assert check.status == pf.WARNING
    assert "FAILED its own ship gate" in check.detail
    assert report.passed


def test_ship_gate_unreadable_metrics_warns_but_does_not_pass_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loadable(monkeypatch, RuntimeError("no manifest on disk"))
    report = pf.run_preflight(FakeReads(), config(quant_model_version="campaign-v3"))
    check = next(c for c in report.checks if c.name == "quant_model")
    assert check.status == pf.WARNING
    assert "gate was NOT consulted" in check.detail


def test_ship_gate_pre_gate_manifest_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle trained before ships_ok was recorded: no silent OK."""
    _loadable(monkeypatch, {"registry": {"n_folds": 5}})
    report = pf.run_preflight(FakeReads(), config(quant_model_version="ancient-v0"))
    check = next(c for c in report.checks if c.name == "quant_model")
    assert check.status == pf.WARNING
    assert "no per-task ships_ok" in check.detail


# -- source-run hook (backtest-campaign Phase 3) ----------------------------


def _committee_run(
    start: date = START,
    end: date = END,
    mode: str = "committee",
    universe_tiers: tuple[int, ...] | None = (1, 3),
) -> dict[str, Any]:
    return {
        "id": "replay-src",
        "mode": mode,
        "start_as_of": start,
        "end_as_of": end,
        "config_json": (
            {"universe_tiers": list(universe_tiers)} if universe_tiers is not None else {}
        ),
    }


def test_source_run_must_exist() -> None:
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(), config(source_run_id="replay-missing"))
    assert "does not exist" in errors_of(exc.value)["source_run"]


def test_source_run_must_be_committee_mode_and_covering() -> None:
    reads = FakeReads(
        runs={"replay-src": _committee_run(start=START + timedelta(days=5), mode="quant")}
    )
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config(source_run_id="replay-src"))
    detail = errors_of(exc.value)["source_run"]
    assert "'quant'" in detail
    assert "does not cover" in detail


def test_source_run_window_not_covering_is_an_error() -> None:
    reads = FakeReads(
        runs={"replay-src": _committee_run(end=END - timedelta(days=1))}
    )
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config(source_run_id="replay-src", mode="portfolio"))
    assert "does not cover" in errors_of(exc.value)["source_run"]


def test_source_run_incompatible_universe_tiers_is_an_error() -> None:
    reads = FakeReads(runs={"replay-src": _committee_run(universe_tiers=(1,))})
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config(source_run_id="replay-src", mode="portfolio"))
    assert "universe_tiers" in errors_of(exc.value)["source_run"]


def test_source_run_without_stored_config_tolerated() -> None:
    """Runs recorded before config_json carried universe_tiers stay usable."""
    reads = FakeReads(runs={"replay-src": _committee_run(universe_tiers=None)})
    report = pf.run_preflight(reads, config(source_run_id="replay-src"))
    check = next(c for c in report.checks if c.name == "source_run")
    assert check.status == pf.OK


def test_valid_source_run_passes() -> None:
    reads = FakeReads(runs={"replay-src": _committee_run()})
    report = pf.run_preflight(reads, config(source_run_id="replay-src", mode="portfolio"))
    check = next(c for c in report.checks if c.name == "source_run")
    assert check.status == pf.OK


def test_portfolio_mode_without_source_run_is_an_error() -> None:
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(), config(mode="portfolio"))
    assert "source_run" in errors_of(exc.value)


# -- error aggregation ------------------------------------------------------


def test_all_failures_reported_together() -> None:
    reads = FakeReads(
        universe=list(SYMBOLS),
        bars_missing_at={START: set(SYMBOLS[:10]), LOOKBACK: set(SYMBOLS[:10])},
        macro_first={s: None for s in pf.REQUIRED_MACRO_SERIES},
    )
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    errs = errors_of(exc.value)
    assert set(errs) == {"bar_coverage_start", "bar_coverage_lookback", "macro_history"}
    # The exception message itself lists every error (loudness contract).
    for name in errs:
        assert name in str(exc.value)


def test_session_window_spans_center() -> None:
    lo, hi = pf.session_window(date(2026, 2, 2))
    assert lo < date(2026, 2, 2) < hi
    assert (hi - lo).days <= 30


# -- bar adjustment (D5) ----------------------------------------------------


def test_adjusted_universe_passes_the_adjustment_check() -> None:
    report = pf.run_preflight(FakeReads(), config())
    check = next(c for c in report.checks if c.name == "bar_adjustment")
    assert check.status == pf.OK
    assert "no unadjusted corporate actions" in check.detail


def test_an_unadjusted_split_fails_the_run() -> None:
    """The whole point: a split that reads as a -50% crash must stop the run,
    not quietly become the flagged set."""
    reads = FakeReads(audit_bars=unadjusted_series(SYMBOLS[0]))
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    detail = errors_of(exc.value)["bar_adjustment"]
    assert "UNADJUSTED" in detail
    assert SYMBOLS[0] in detail
    assert "re-backfill" in detail


def test_a_large_move_with_no_recorded_split_is_not_an_adjustment_failure() -> None:
    """A genuine -50% crash is data, not a defect. Blaming it on adjustment
    would make the check cry wolf on exactly the events the platform is for."""
    reads = FakeReads(audit_bars=unadjusted_series(SYMBOLS[0]), splits={("OTHER", SPLIT_DAY)})
    report = pf.run_preflight(reads, config())
    assert next(c for c in report.checks if c.name == "bar_adjustment").status == pf.OK


def test_a_split_matched_within_one_session_still_fails() -> None:
    """Vendors disagree by a session about which bar a split's step lands on."""
    reads = FakeReads(
        audit_bars=unadjusted_series(SYMBOLS[0]),
        splits={(SYMBOLS[0], SPLIT_DAY - timedelta(days=1))},
    )
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    assert "bar_adjustment" in errors_of(exc.value)


def test_no_recorded_splits_at_all_is_a_warning_not_a_pass() -> None:
    """corporate_actions held 0 rows for a year. With no witness the check
    cannot conclude anything, and must say so rather than report OK."""
    report = pf.run_preflight(FakeReads(splits=set()), config())
    check = next(c for c in report.checks if c.name == "bar_adjustment")
    assert check.status == pf.WARNING
    assert "UNVERIFIED" in check.detail


def test_a_source_declared_unadjusted_fails_without_needing_a_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared half of the check: no corporate action required."""
    monkeypatch.setattr(pf, "UNADJUSTED_BAR_SOURCES", frozenset({"rawvendor"}))
    reads = FakeReads(
        audit_bars=audit_frame(SYMBOLS[0], [50.0] * 4, source="rawvendor", adj=False)
    )
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, config())
    assert "declared unadjusted" in errors_of(exc.value)["bar_adjustment"]


def test_a_declared_raw_source_carrying_adj_close_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pf, "UNADJUSTED_BAR_SOURCES", frozenset({"rawvendor"}))
    reads = FakeReads(
        audit_bars=audit_frame(SYMBOLS[0], [50.0] * 4, source="rawvendor", adj=True)
    )
    report = pf.run_preflight(reads, config())
    assert next(c for c in report.checks if c.name == "bar_adjustment").status == pf.OK


def test_a_symbol_with_a_second_adjusted_feed_is_not_flagged_as_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Only" matters: a symbol that also has an adjusted row has an adjusted
    close available through the canonical view."""
    monkeypatch.setattr(pf, "UNADJUSTED_BAR_SOURCES", frozenset({"rawvendor"}))
    frame = pd.concat(
        [
            audit_frame(SYMBOLS[0], [50.0] * 4, source="rawvendor", adj=False),
            audit_frame(SYMBOLS[0], [50.0] * 4, source="tiingo", adj=True),
        ]
    )
    report = pf.run_preflight(FakeReads(audit_bars=frame), config())
    assert next(c for c in report.checks if c.name == "bar_adjustment").status == pf.OK


def test_adjustment_audit_spans_the_returns_lookback() -> None:
    """A split before the window still corrupts the returns the run computes,
    so the audit reaches back as far as the runner does."""
    reads = FakeReads()
    pf.run_preflight(reads, config())
    (start, end) = reads.audit_queries[0]
    assert start == LOOKBACK
    assert end == END


def test_no_bars_at_all_warns_rather_than_silently_passing() -> None:
    reads = FakeReads(audit_bars=pd.DataFrame())
    report = pf.run_preflight(reads, config())
    check = next(c for c in report.checks if c.name == "bar_adjustment")
    assert check.status == pf.WARNING
    assert "could not be checked" in check.detail


def test_a_feed_that_stores_raw_close_beside_adj_close_is_not_a_false_alarm() -> None:
    """The audit reads the EFFECTIVE close, the one the canonical view serves.

    Tiingo stores XLE's unadjusted 92.22 -> 45.92 across its real 2025-12-05
    two-for-one and puts the continuous series in adj_close. Reading `close`
    flagged both XLE and XLK as unadjusted when the runner never sees that
    column — a check that cries wolf on correct data gets switched off."""
    frame = audit_frame(SYMBOLS[0], [100.0, 100.0, 50.0, 50.0], source="tiingo")
    frame["adj_close"] = [50.0, 50.0, 50.0, 50.0]
    report = pf.run_preflight(FakeReads(audit_bars=frame), config())
    assert next(c for c in report.checks if c.name == "bar_adjustment").status == pf.OK


def test_adj_close_that_is_itself_unadjusted_still_fails() -> None:
    """Trusting the column is not the same as trusting the numbers in it."""
    frame = audit_frame(SYMBOLS[0], [100.0, 100.0, 50.0, 50.0], source="tiingo")
    frame["adj_close"] = [100.0, 100.0, 50.0, 50.0]
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(audit_bars=frame), config())
    assert "bar_adjustment" in errors_of(exc.value)


# -- discovery (wide-campaign discovery layer) ------------------------------


def _discovery_config(**overrides: Any) -> ReplayConfig:
    return config(**{"discovery_mode": "code", **overrides})


def test_discovery_off_asks_nothing_about_tier_2() -> None:
    """The control arm must not acquire new ways to fail."""
    reads = FakeReads()
    report = pf.run_preflight(reads, _discovery_config(discovery_mode="off"))
    names = [c.name for c in report.checks]
    assert "discovery_universe" not in names
    assert "discovery_bar_coverage" not in names
    assert (SURVEILLANCE_TIER,) not in reads.tier_queries


def test_discovery_on_checks_the_surveillance_universe() -> None:
    reads = FakeReads()
    report = pf.run_preflight(reads, _discovery_config())
    checks = {c.name: c for c in report.checks}
    assert checks["discovery_universe"].status == pf.OK
    assert str(len(TIER2_SYMBOLS)) in checks["discovery_universe"].detail
    assert checks["discovery_bar_coverage"].status == pf.OK
    assert (SURVEILLANCE_TIER,) in reads.tier_queries


def test_an_empty_tier_2_refuses_the_run() -> None:
    """The silent failure this prevents: 115 sessions of detectors over zero
    symbols, reporting zero promotions as though discovery found nothing."""
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(FakeReads(surveillance=[]), _discovery_config())
    detail = errors_of(exc.value)["discovery_universe"]
    assert "EMPTY" in detail
    assert "backdate" in detail


def test_tier_2_without_lookback_history_refuses_the_run() -> None:
    """Coverage is measured at start - 420d, not at start: range_break reads
    252 prior sessions and range_bound will not fire below 131, so a universe
    with bars only from the start date cannot flag for months."""
    reads = FakeReads(bars_missing_at={DISCOVERY_REACH: set(TIER2_SYMBOLS[:40])})
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight(reads, _discovery_config())
    detail = errors_of(exc.value)["discovery_bar_coverage"]
    assert "10/50" in detail
    assert f"{DISCOVERY_LOOKBACK_DAYS}d lookback" in detail


def test_a_partly_covered_tier_2_still_passes() -> None:
    """Tier 2 is a whole-market list; a real fraction of it is illiquid enough
    to have gaps, and discovery scans those names rather than trading them.
    The floor is deliberately well below the traded core's 95%."""
    thin = set(TIER2_SYMBOLS[: int(len(TIER2_SYMBOLS) * 0.3)])
    reads = FakeReads(bars_missing_at={DISCOVERY_REACH: thin})
    report = pf.run_preflight(reads, _discovery_config())
    assert report.passed


def test_the_adjustment_audit_widens_to_the_surveillance_universe() -> None:
    """An unadjusted split reads as a crash and the detectors hunt crashes.
    Auditing only the 55-name core while sweeping 11,500 would leave the
    flagged set contaminated in exactly the population promotions come from."""
    narrow = FakeReads()
    pf.run_preflight(narrow, config())
    assert set(narrow.audit_symbols[0]) == set(SYMBOLS)

    wide = FakeReads()
    pf.run_preflight(wide, _discovery_config())
    assert set(wide.audit_symbols[0]) == set(SYMBOLS) | set(TIER2_SYMBOLS)
