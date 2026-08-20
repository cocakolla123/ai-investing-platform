"""Replay discovery — panel, gates, determinism, and the off-mode parity proof.

The load-bearing assertions here, in order of how much they matter:

1. **Parity.** ``discovery_mode='off'`` must reproduce the pre-discovery
   runner exactly, and a discovery driver that promotes nothing must not
   perturb a single decision, fill or equity point. If discovery is not a
   pure addition, the A/B it exists to measure is not an A/B.
2. **Point-in-time.** The panel is loaded ONCE over the whole window, so the
   per-day bound lives entirely in ``slice`` — a bar dated after ``as_of``
   must be invisible, and a broken bound must raise, not silently leak.
3. **Determinism.** Same config, same data, same promotions — otherwise
   ``discovery_mode`` inside ``config_hash`` is a lie.
4. **The haiku gate never calls a real model.** It is exercised only through
   a fake; the assertions cover the call shape, not a network round trip.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import pytest

from dynamo.domain.models.market import SurveillanceFlag
from dynamo.replay.discovery import (
    DISCOVERY_LOOKBACK_DAYS,
    PromotionWindow,
    ReplayDiscovery,
    SurveillancePanel,
    code_gate,
    flag_rows,
    haiku_gate,
    off_gate,
    tradable_flags,
)
from dynamo.replay.gateway import ReplayLeakageError
from dynamo.replay.runner import ReplayConfig, ReplayRunner
from tests.unit.replay.test_runner import (
    END,
    HISTORY_SESSIONS,
    RUN_SESSIONS,
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    nyse_sessions,
)

# The tier-2 surveillance universe the panel scans. Deliberately disjoint
# from ``test_runner.UNIVERSE`` (the traded core) so a promotion is visibly a
# name the run would otherwise never have seen.
SPIKE = "SPIKE"  # flat, then a +12% session on 10x volume
CALM = "CALM"  # flat throughout — fires nothing
PENNY = "PENNY"  # same shape as SPIKE at 30 cents — price filter rejects it
THIN = "THIN"  # same shape as SPIKE on 200 shares/day — volume filter rejects
TIER2 = (SPIKE, CALM, PENNY, THIN)

PANEL_SESSIONS = 200


def _series(base: float, n: int, *, spike: bool) -> list[float]:
    """Flat with +/-0.1% alternating noise (so sigma > 0), optional last-day
    jump big enough to clear the z-score's 2% floor without tripping the
    2x price-ratio sanity ceiling."""
    closes = [base * (1 + (0.001 if i % 2 else -0.001)) for i in range(n)]
    if spike:
        closes[-1] = closes[-2] * 1.12
    return closes


def panel_bars(sessions: list[date]) -> pd.DataFrame:
    shape = {
        SPIKE: (50.0, True, 2_000_000),
        CALM: (60.0, False, 2_000_000),
        PENNY: (0.30, True, 2_000_000),
        THIN: (50.0, True, 200),
    }
    rows: list[dict[str, Any]] = []
    for symbol, (base, spike, volume) in shape.items():
        closes = _series(base, len(sessions), spike=spike)
        for i, day in enumerate(sessions):
            last = spike and i == len(sessions) - 1
            rows.append(
                {
                    "symbol": symbol,
                    "ts": datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC),
                    "open": closes[i] * 0.999,
                    "high": closes[i] * 1.01,
                    "low": closes[i] * 0.99,
                    "close": closes[i],
                    "adj_close": closes[i],
                    "volume": volume * (10 if last else 1),
                    "source": "tiingo",
                    "volume_source": "tiingo",
                }
            )
    return pd.DataFrame(rows)


class FakeSurveillanceGateway:
    """Duck-typed :class:`~dynamo.replay.discovery.SurveillanceReads`, plus the
    quant-runner reads, so one object drives both the loop and discovery."""

    def __init__(
        self,
        as_of: date,
        bars: pd.DataFrame,
        scores: pd.DataFrame,
        surveillance: pd.DataFrame,
        *,
        members: dict[str, date] | None = None,
    ) -> None:
        self._inner = FakeGateway(as_of, bars, scores)
        self.as_of = as_of
        self._surveillance = surveillance
        self._members = members or dict.fromkeys(TIER2, date(2000, 1, 1))
        self.surveillance_reads = 0

    # -- the surveillance slice -------------------------------------------
    def surveillance_universe(self) -> list[str]:
        return sorted(s for s, start in self._members.items() if start <= self.as_of)

    def surveillance_bars(self, symbols: Any, lookback_days: int) -> pd.DataFrame:
        self.surveillance_reads += 1
        frame = self._surveillance
        ts = pd.to_datetime(frame["ts"]).dt.date
        frame = cast(
            pd.DataFrame,
            frame[
                (ts <= self.as_of)
                & (ts >= self.as_of - timedelta(days=lookback_days))
                & frame["symbol"].isin(list(symbols))
            ],
        )
        return frame.reset_index(drop=True)

    # -- delegation (the runner's own reads) ------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class FakeFlagSink:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def insert_flags(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.rows.extend(rows)
        return len(rows)


class FakeTriageModel:
    """Records every call and answers from a fixed verdict map. The ``haiku``
    gate is NEVER exercised against a real client anywhere in this suite."""

    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        symbol = kwargs["context"].splitlines()[0].rsplit(" ", 1)[-1]
        return {
            "materiality": 0.9,
            "verdict": self.verdicts.get(symbol, "noise"),
            "reason": f"because {symbol}",
        }


@pytest.fixture(scope="module")
def sessions() -> list[date]:
    return nyse_sessions(END, HISTORY_SESSIONS)


@pytest.fixture(scope="module")
def surveillance(sessions: list[date]) -> pd.DataFrame:
    return panel_bars(sessions[-PANEL_SESSIONS:])


@pytest.fixture()
def panel(sessions: list[date], surveillance: pd.DataFrame) -> SurveillancePanel:
    gw = FakeSurveillanceGateway(
        sessions[-1], pd.DataFrame(), pd.DataFrame(), surveillance
    )
    return SurveillancePanel.load(gw, list(TIER2), span_days=RUN_SESSIONS)


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------


def _first(frame: pd.DataFrame) -> date:
    return cast(date, pd.Timestamp(frame["ts"].iloc[0]).date())


def _last(frame: pd.DataFrame) -> date:
    return cast(date, pd.Timestamp(frame["ts"].iloc[-1]).date())


def test_panel_reads_the_whole_window_in_one_query(
    sessions: list[date], surveillance: pd.DataFrame
) -> None:
    """The reason the panel exists. Measured on the real database a per-day
    read is 9s over 3.09M rows; 115 of them is 17 minutes of round trips
    re-reading the same rows."""
    gw = FakeSurveillanceGateway(
        sessions[-1], pd.DataFrame(), pd.DataFrame(), surveillance
    )
    panel = SurveillancePanel.load(gw, list(TIER2), span_days=RUN_SESSIONS)
    assert gw.surveillance_reads == 1
    assert set(panel.symbols) == set(TIER2)
    for day in sessions[-RUN_SESSIONS:]:
        panel.slice(day)
    assert gw.surveillance_reads == 1, "slicing must never hit the database"


def test_slice_is_the_point_in_time_bound(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    """The panel holds bars from the whole run window, so every later
    session's data is physically present when an early day is sliced. The
    slice is what makes it invisible."""
    early, late = sessions[-RUN_SESSIONS], sessions[-1]
    assert early < late
    sliced = panel.slice(early)
    for frame in sliced.per_symbol.values():
        assert _last(frame) <= early
    # ...and the later data really is in the panel, i.e. the test above is
    # not passing because there is nothing to leak.
    assert any(_last(f) == late for f in panel.slice(late).per_symbol.values())


def test_slice_respects_the_lookback_floor(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    day = sessions[-1]
    frames = panel.slice(day, lookback_days=30).per_symbol
    for frame in frames.values():
        assert _first(frame) >= day - timedelta(days=30)
    assert len(frames[CALM]) < len(panel.slice(day).per_symbol[CALM])


def test_a_broken_bound_raises_instead_of_leaking() -> None:
    """The slice's index arithmetic cannot return a later bar today. The
    assertion exists so that if a future edit breaks it, the run stops rather
    than quietly trading on tomorrow's close."""
    future = pd.DataFrame(
        {"ts": [pd.Timestamp("2026-07-01", tz="UTC")], "close": [1.0]}
    )
    with pytest.raises(ReplayLeakageError, match="panel slice"):
        SurveillancePanel._assert_bounded({"X": future}, date(2026, 6, 30))


def test_liquidity_and_price_filters_remove_untradable_names(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    day = sessions[-1]
    wide_open = panel.slice(day, min_dollar_volume=0.0, min_price=0.0)
    assert set(wide_open.symbols) == set(TIER2)
    assert wide_open.filtered_out == 0

    filtered = panel.slice(day, min_dollar_volume=1_000_000.0, min_price=1.0)
    assert PENNY not in filtered.symbols, "30-cent name must not survive min_price"
    assert THIN not in filtered.symbols, "200 shares/day must not survive the floor"
    assert SPIKE in filtered.symbols and CALM in filtered.symbols
    assert filtered.filtered_out == 2


def test_tradability_is_judged_on_pre_as_of_bars_only() -> None:
    """The leak ``_assert_bounded`` structurally CANNOT catch.

    That tripwire checks bar dates. The tradability filter is a second,
    independent way the future can reach into a replay day: if the liquidity
    and price test were computed over the whole panel instead of the slice,
    a name that is untradable on day D and only becomes liquid weeks later
    would enter the candidate pool on D. Every bar would still be correctly
    dated, so the tripwire would pass and the flags would look legitimate.

    So: a name that is a 40-cent, 200-share stock through D and a $50,
    million-share stock afterwards must be filtered AT D, and admitted only
    once the liquidity is genuinely in its past.
    """
    cutoff = date(2026, 6, 30)
    days = pd.bdate_range("2026-01-01", "2026-12-31")
    late_bloomer = pd.DataFrame(
        {
            "ts": days,
            "close": [0.40 if d.date() <= cutoff else 50.0 for d in days],
            "volume": [200 if d.date() <= cutoff else 1_000_000 for d in days],
            "volume_source": "consolidated",
        }
    )
    last_day = cast(date, days[-1].date())  # type: ignore[union-attr]
    panel = SurveillancePanel({"LATE": late_bloomer}, last_day)

    at_cutoff = panel.slice(cutoff, min_dollar_volume=1_000_000.0, min_price=1.0)
    assert "LATE" not in at_cutoff.symbols, (
        "a name whose liquidity arrives after as_of must be untradable at as_of"
    )
    assert at_cutoff.filtered_out == 1

    # Not vacuous: the same filter admits it once the volume is in its past.
    # LIQUIDITY_WINDOW_SESSIONS trailing sessions must clear the floor, so
    # step well past the cutoff rather than one day past it.
    later = panel.slice(
        cutoff + timedelta(days=180), min_dollar_volume=1_000_000.0, min_price=1.0
    )
    assert "LATE" in later.symbols


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def _flag(symbol: str, detector: str, severity: float) -> SurveillanceFlag:
    return SurveillanceFlag(
        symbol=symbol,
        detector=detector,
        severity=severity,
        details={"n": severity},
        ts=datetime(2026, 6, 30, tzinfo=UTC),
    )


FLAGS = [
    _flag("AAA", "volume_spike", 1.0),  # loudest raw severity, weakest detector
    _flag("BBB", "range_break", 0.6),
    _flag("BBB", "price_zscore", 0.7),  # corroborated by two detectors
    _flag("CCC", "range_bound", 0.5),
    _flag("DQ", "data_quality", 0.9),  # never a trading candidate
]


def test_data_quality_flags_are_never_promotable() -> None:
    """They mark a bar the detectors could not trust — an unadjusted
    corporate action — not a price move. Promoting on one puts a fabricated
    return into the universe."""
    assert [f.symbol for f in tradable_flags(FLAGS)] == ["AAA", "BBB", "BBB", "CCC"]
    assert "DQ" not in code_gate(FLAGS, cap=10).symbols


def test_code_gate_ranks_by_the_shared_ranking_not_raw_severity() -> None:
    """AAA carries the single highest raw severity (1.0) off the weakest,
    loudest detector. BBB has two independent detectors agreeing. Ranking by
    raw severity — the obvious implementation — puts AAA first, which is
    exactly the failure ``flag_ranking`` exists to fix."""
    result = code_gate(FLAGS, cap=1)
    assert result.symbols == ("BBB",)


def test_code_gate_caps_and_records_the_audit_trail() -> None:
    result = code_gate(FLAGS, cap=2)
    assert len(result.promoted) == 2
    assert [p.rank for p in result.promoted] == [1, 2]
    bbb = next(p for p in result.promoted if p.symbol == "BBB")
    assert bbb.detectors == ("price_zscore", "range_break")
    assert bbb.severity_total == pytest.approx(1.3)
    assert bbb.score > 0.0
    assert code_gate(FLAGS, cap=0).promoted == ()
    assert code_gate([], cap=10).promoted == ()


def test_code_gate_is_pure() -> None:
    first, second = code_gate(FLAGS, cap=3), code_gate(FLAGS, cap=3)
    assert first == second


def test_off_gate_promotes_nothing() -> None:
    assert off_gate(FLAGS) == off_gate([])
    assert off_gate(FLAGS).promoted == ()
    assert off_gate(FLAGS).llm_calls == 0


def test_haiku_gate_promotes_only_material_verdicts() -> None:
    model = FakeTriageModel({"BBB": "material", "AAA": "watch", "CCC": "noise"})
    result = haiku_gate(
        FLAGS,
        cap=10,
        as_of=date(2026, 6, 30),
        model=model,
        system_prompt="triage system",
        schema={"type": "object"},
    )
    assert result.symbols == ("BBB",)
    # Every RANKED symbol costs a call; only "material" promotes.
    assert result.llm_calls == len(result.considered) == 3
    assert len(model.calls) == 3


def test_haiku_gate_mirrors_the_live_triage_call_shape() -> None:
    """Live parity with ``orchestration/assets/research.py::_triage_flags`` —
    the whole point of the arm is to measure what THAT decision adds."""
    model = FakeTriageModel({"BBB": "material"})
    haiku_gate(
        FLAGS,
        cap=1,
        as_of=date(2026, 6, 30),
        model=model,
        system_prompt="triage system",
        schema={"type": "object"},
    )
    call = model.calls[0]
    assert call["role"] == "triage"
    assert call["model_tier"] == "screen"
    assert call["max_tokens"] == 512
    assert call["system"] == "triage system"
    assert call["context"].startswith("## Surveillance flags for BBB\n")
    assert "- price_zscore severity=0.70 details=" in call["context"]
    assert call["instruction"] == (
        "Classify the materiality of these flags for BBB as of 2026-06-30."
    )


def test_haiku_gate_does_not_promote_on_an_unusable_response() -> None:
    """A triage call that did not produce a verdict is not evidence of
    materiality."""

    class Silent:
        def complete_json(self, **kwargs: Any) -> dict[str, Any]:
            return {}

    result = haiku_gate(
        FLAGS,
        cap=10,
        as_of=date(2026, 6, 30),
        model=Silent(),
        system_prompt="",
        schema={},
    )
    assert result.promoted == ()
    assert result.llm_calls == 3


def test_haiku_mode_refuses_to_construct_without_a_model() -> None:
    with pytest.raises(ValueError, match="needs a triage model"):
        ReplayDiscovery(
            SurveillancePanel({}, date(2026, 6, 30)),
            run_id="r",
            mode="haiku",
            promotions_per_day=1,
            promotion_days=1,
        )


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        ReplayDiscovery(
            SurveillancePanel({}, date(2026, 6, 30)),
            run_id="r",
            mode="sonnet",
            promotions_per_day=1,
            promotion_days=1,
        )


# --------------------------------------------------------------------------
# the promotion window
# --------------------------------------------------------------------------


def test_promotion_window_expires_after_n_sessions() -> None:
    window = PromotionWindow(sessions=2)
    window.record(date(2026, 6, 1), ["AAA"])
    assert window.active() == ("AAA",)
    window.record(date(2026, 6, 2), ["BBB"])
    assert window.active() == ("AAA", "BBB")
    window.record(date(2026, 6, 3), [])
    assert window.active() == ("BBB",), "AAA has aged out"
    window.record(date(2026, 6, 4), [])
    assert window.active() == ()


def test_promotion_window_counts_sessions_not_calendar_days() -> None:
    """A promotion made before a long weekend must get the same number of
    TRADING days in the universe as one made on a Tuesday."""
    window = PromotionWindow(sessions=3)
    window.record(date(2026, 6, 5), ["FRI"])  # Friday
    window.record(date(2026, 6, 8), [])  # Monday, three calendar days later
    window.record(date(2026, 6, 9), [])
    assert "FRI" in window.active()


def test_zero_day_window_promotes_into_nothing() -> None:
    window = PromotionWindow(sessions=0)
    window.record(date(2026, 6, 1), ["AAA"])
    assert window.active() == ()


# --------------------------------------------------------------------------
# the driver: end to end over the fake gateway
# --------------------------------------------------------------------------


def _discovery(
    panel: SurveillancePanel, *, mode: str = "code", **kwargs: Any
) -> ReplayDiscovery:
    defaults: dict[str, Any] = {
        "run_id": "replay-test",
        "mode": mode,
        "promotions_per_day": 5,
        "promotion_days": 3,
        "min_dollar_volume": 1_000_000.0,
        "min_price": 1.0,
        "sectors": dict.fromkeys(TIER2, "tech"),
    }
    return ReplayDiscovery(panel, **{**defaults, **kwargs})


def test_discovery_widens_the_universe_with_a_name_the_core_never_had(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    from tests.unit.replay.test_runner import UNIVERSE

    day = sessions[-1]
    gw = FakeSurveillanceGateway(day, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    sink = FakeFlagSink()
    discovery = _discovery(panel, flag_sink=sink)
    record = discovery.run_day(gw, day, UNIVERSE)

    assert SPIKE in record.universe
    assert set(UNIVERSE) <= set(record.universe)
    assert SPIKE not in UNIVERSE, "the fixture must promote a name outside the core"
    assert {f.detector for f in record.flags} >= {"price_zscore", "volume_spike"}
    # The filtered names never reached the detectors at all.
    assert record.filtered_out == 2
    assert all(f.symbol not in (PENNY, THIN) for f in record.flags)
    # ...and every flag was persisted, through the ONLY write path there is.
    assert len(sink.rows) == len(record.flags)
    assert {r["run_id"] for r in sink.rows} == {"replay-test"}


def test_membership_starting_after_the_day_is_invisible(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    """Point-in-time universe: the panel holds SPIKE's bars for the whole
    window, but a tier-2 membership that begins tomorrow must not let it be
    scanned today."""
    day = sessions[-1]
    gw = FakeSurveillanceGateway(
        day,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        members={SPIKE: day + timedelta(days=1), CALM: date(2000, 1, 1)},
    )
    record = _discovery(panel).run_day(gw, day, [])
    assert SPIKE not in record.universe
    assert all(f.symbol != SPIKE for f in record.flags)


def test_promotions_expire_out_of_the_universe(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    discovery = _discovery(panel, promotion_days=1)
    window = sessions[-3:]
    universes = [
        discovery.run_day(
            FakeSurveillanceGateway(day, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
            day,
            [],
        ).universe
        for day in window
    ]
    # SPIKE only spikes on the LAST session, so it enters on that day alone.
    assert SPIKE in universes[-1]
    assert SPIKE not in universes[0]


def test_same_config_twice_promotes_identically(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    """Determinism is what makes ``discovery_mode`` meaningful inside
    ``config_hash``: a re-run of the same id must re-derive the same rows."""

    def run() -> list[list[dict[str, Any]]]:
        discovery = _discovery(panel)
        out = []
        for day in sessions[-RUN_SESSIONS:]:
            gw = FakeSurveillanceGateway(
                day, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )
            out.append(flag_rows("replay-test", discovery.run_day(gw, day, [])))
        return out

    assert run() == run()


def test_flag_rows_are_deterministically_ided_and_mark_promotions(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    day = sessions[-1]
    gw = FakeSurveillanceGateway(day, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    record = _discovery(panel).run_day(gw, day, [])
    rows = flag_rows("replay-test", record)
    assert len(rows) == len({r["id"] for r in rows}), "ids must be unique"
    for row in rows:
        assert row["id"].startswith(f"replay-test:{day.isoformat()}:{row['symbol']}:")
        assert row["as_of"] == day
    promoted = {p.symbol for p in record.promoted}
    assert {r["symbol"] for r in rows if r["promoted"]} == promoted


def test_off_mode_scans_nothing_and_promotes_nothing(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    day = sessions[-1]
    gw = FakeSurveillanceGateway(day, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    record = _discovery(panel, mode="off").run_day(gw, day, ["AAA"])
    assert record.promoted == ()
    assert record.universe == ("AAA",)


def test_summary_reports_the_campaign_acceptance_counters(
    panel: SurveillancePanel, sessions: list[date]
) -> None:
    discovery = _discovery(panel)
    for day in sessions[-RUN_SESSIONS:]:
        gw = FakeSurveillanceGateway(
            day, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        discovery.run_day(gw, day, [])
    stats = discovery.summary()
    assert stats["mode"] == "code"
    assert stats["sessions"] == RUN_SESSIONS
    assert stats["llm_calls"] == 0
    assert stats["distinct_promoted"] >= 1
    assert stats["flags_per_session"] > 0


# --------------------------------------------------------------------------
# config_hash
# --------------------------------------------------------------------------


BASE_CONFIG = {"start": date(2026, 6, 1), "end": date(2026, 6, 30), "label": "x"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discovery_mode", "code"),
        ("discovery_min_dollar_volume", 5_000_000.0),
        ("discovery_min_price", 5.0),
        ("discovery_promotions_per_day", 30),
        ("discovery_promotion_days", 10),
    ],
)
def test_every_discovery_knob_changes_the_run_id(field: str, value: Any) -> None:
    """A different discovery setting is a different experiment. Two arms of
    the campaign must never be able to share a run id and upsert over each
    other's rows."""
    base = ReplayConfig(**BASE_CONFIG)
    changed = ReplayConfig(**BASE_CONFIG, **{field: value})
    assert base.config_hash != changed.config_hash
    assert base.run_id != changed.run_id


def test_unknown_discovery_mode_is_refused_by_the_config() -> None:
    with pytest.raises(ValueError, match="discovery_mode"):
        ReplayConfig(**BASE_CONFIG, discovery_mode="gpt")


# --------------------------------------------------------------------------
# PARITY: discovery is a pure addition
# --------------------------------------------------------------------------


class PassThroughDiscovery:
    """A driver that promotes nothing. Standing in for a discovery run whose
    detectors were silent, it isolates the question this test asks: does the
    PRESENCE of discovery change anything downstream?"""

    def __init__(self) -> None:
        self.calls = 0

    def universe_for(self, gw: Any, day: date, base_universe: Any) -> list[str]:
        self.calls += 1
        return list(base_universe)


def _decisions_without_run_id(result: Any) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in row.items() if k not in ("id", "run_id")}
        for row in result.decisions
    ]


def _run(config: ReplayConfig, frames: Any, discovery: Any = None) -> Any:
    _sessions, bars, scores = frames
    return ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=FakeRunRepo(),
        config=config,
        discovery=discovery,
    ).run()


@pytest.fixture(scope="module")
def runner_frames() -> tuple[list[date], pd.DataFrame, pd.DataFrame]:
    from tests.unit.replay.test_runner import make_bars, make_scores

    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    return sessions, make_bars(sessions), make_scores(sessions)


def test_a_discovery_that_promotes_nothing_changes_nothing(
    runner_frames: Any,
) -> None:
    """THE parity test. Everything downstream of step 4 is supposed to be
    untouched; if adding the layer perturbs a single fill, the campaign's
    `code` vs `off` comparison measures the plumbing rather than discovery."""
    sessions = runner_frames[0]
    window = sessions[-RUN_SESSIONS:]
    control = ReplayConfig(
        start=window[0], end=window[-1], label="control", horizon_sessions=3
    )
    treated = control.model_copy(update={"discovery_mode": "code"})
    assert control.run_id != treated.run_id  # different experiments, by design

    baseline = _run(control, runner_frames)
    driver = PassThroughDiscovery()
    with_layer = _run(treated, runner_frames, discovery=driver)

    assert driver.calls == len(window), "the driver must run on every session"
    assert with_layer.equity_curve == baseline.equity_curve
    assert with_layer.fills == baseline.fills
    assert with_layer.session_records == baseline.session_records
    assert with_layer.turnover_total == baseline.turnover_total
    # Decisions differ ONLY in the run id they are namespaced by — the two
    # arms are deliberately distinct runs. Everything the run DECIDED is
    # identical.
    assert _decisions_without_run_id(with_layer) == _decisions_without_run_id(baseline)


def test_off_mode_refuses_an_injected_driver(runner_frames: Any) -> None:
    """Structural, not aspirational: the control arm cannot accidentally be
    handed a discovery driver and still call itself the control."""
    sessions = runner_frames[0]
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(start=window[0], end=window[-1], label="control")
    with pytest.raises(ValueError, match="discovery_mode is 'off'"):
        _run(config, runner_frames, discovery=PassThroughDiscovery())


def test_the_lookback_matches_the_detector_catalog() -> None:
    """A shorter window makes ``range_break`` report 3-month highs as 52-week
    breakouts and stops ``range_bound`` firing at all — both silently, which
    is why the number is derived rather than chosen."""
    from dynamo.data.detector_windows import TIER2_SCAN_LOOKBACK_DAYS

    assert DISCOVERY_LOOKBACK_DAYS == TIER2_SCAN_LOOKBACK_DAYS
