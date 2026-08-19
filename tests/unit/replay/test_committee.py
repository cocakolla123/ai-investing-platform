"""Committee replay with a FAKE ResearchModel (plan Phase 5).

Covers: gateway-only dossier assembly (future news invisible), the real
4-stage engine over canned batch outputs, verdicts landing ONLY in the
replay-verdict sink (firewall extension — no live-table sink even exists on
the fakes), run-namespaced custom ids, budget-cap refusal marking the run
``budget_exhausted`` with partial results preserved, and entry verdicts
resolving into ``source='replay'`` experiences. ZERO network, ZERO LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.domain.models.market import Instrument, NewsItem
from dynamo.replay.budget import ReplayBudget
from dynamo.replay.committee import (
    CommitteeReplay,
    build_replay_contexts,
    committee_shortlist,
    namespace_custom_id,
    strip_namespace,
)
from dynamo.replay.runner import ReplayConfig
from dynamo.research.committee import split_custom_id
from tests.unit.replay.test_runner import (
    END,
    HISTORY_SESSIONS,
    UNIVERSE,
    FakeGateway,
    make_bars,
    make_scores,
    nyse_sessions,
)

RUN_SESSIONS = 6
HORIZON = 3

SCHEMAS = {"dossier": {}, "advocate": {}, "arbiter": {}}
TIER_MODELS = {
    "screen": "claude-haiku-4-5",
    "analyst": "claude-sonnet-5",
    "arbiter": "claude-opus-5",
}


@dataclass(frozen=True)
class FakePrompt:
    name: str
    version: str
    system: str


PROMPTS = {
    name: FakePrompt(name=name, version=f"{name}-v1", system=f"system prompt for {name}")
    for name in ("dossier_analyst", "bull_advocate", "bear_advocate", "arbiter")
}


def canned_payload(stage: str, symbol: str) -> dict[str, Any]:
    if stage == "macro_digest":
        return {
            "narrative": "canned replay macro narrative",
            "rates_outlook": "steady",
            "inflation_outlook": "cooling",
            "regime_comment": "stable",
        }
    if stage == "dossier":
        return {"summary": f"dossier for {symbol}", "strengths": [], "weaknesses": []}
    if stage in ("bull", "bear"):
        return {
            "thesis": f"{stage} case for {symbol}",
            "key_points": [],
            "kill_cases": [],
            "expected_move_pct": 10.0 if stage == "bull" else -8.0,
            "horizon_days": 20,
            "conviction": 0.8 if stage == "bull" else 0.2,
        }
    assert stage == "arbiter"
    return {
        "action": "buy",
        "conviction": 0.75,
        "thesis": f"verdict for {symbol}",
        "expected_upside_pct": 8.0,
        "expected_downside_pct": 4.0,
        "horizon_days": 20,
    }


class FakeResearchModel:
    """Canned batch pipeline. The sync path is booby-trapped: committee
    replay's documented primary path is batch-only."""

    def __init__(self, budget: ReplayBudget | None = None, cost_per_batch: float = 0.0) -> None:
        self._batches: dict[str, list[dict[str, Any]]] = {}
        self.submitted: list[list[dict[str, Any]]] = []
        self._budget = budget
        self._cost_per_batch = cost_per_batch
        self._n = 0

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("committee replay must not use the sync path")

    def submit_batch(self, requests: Sequence[dict[str, Any]]) -> str:
        self._n += 1
        batch_id = f"batch-{self._n}"
        self._batches[batch_id] = [dict(r) for r in requests]
        self.submitted.append(self._batches[batch_id])
        # Simulate the metered actual cost flowing through the ledger tee.
        if self._budget is not None and self._cost_per_batch:
            self._budget.record_actual(self._cost_per_batch)
        return batch_id

    def batch_status(self, batch_id: str) -> str:
        return "ended"

    def batch_results(self, batch_id: str) -> list[tuple[str, dict[str, Any] | None]]:
        out: list[tuple[str, dict[str, Any] | None]] = []
        for req in self._batches[batch_id]:
            custom_id = str(req["custom_id"])
            _run_id, live_id = custom_id.split(":", 1)
            _, symbol, stage = split_custom_id(live_id)
            out.append((custom_id, canned_payload(stage, symbol)))
        return out


class CommitteeGateway(FakeGateway):
    """FakeGateway + the committee-context reads, all bounded at as_of."""

    def __init__(
        self,
        as_of: date,
        bars: pd.DataFrame,
        scores: pd.DataFrame,
        news: list[NewsItem] = (),  # type: ignore[assignment]
    ) -> None:
        super().__init__(as_of, bars, scores)
        self._news = list(news)

    def macro_digest(self) -> dict[str, Any] | None:
        return {"narrative": f"macro digest as of {self.as_of.isoformat()}"}

    def instruments(self, symbols: Any) -> dict[str, Instrument]:
        return {s: Instrument(symbol=s, name=f"{s} Corp", sector="tech") for s in symbols}

    def news(self, symbols: Any, lookback_days: int = 7) -> dict[str, list[NewsItem]]:
        # The REAL gateway bounds published_at <= as_of; the fake enforces the
        # same contract so the context test proves the builder never reaches
        # around the gateway for more.
        cutoff = datetime(self.as_of.year, self.as_of.month, self.as_of.day, 23, 59, tzinfo=UTC)
        out: dict[str, list[NewsItem]] = {s: [] for s in symbols}
        for item in self._news:
            if item.published_at > cutoff:
                continue
            for s in item.symbols:
                if s in out:
                    out[s].append(item)
        return out

    def transcripts(self, symbols: Any) -> pd.DataFrame:
        return pd.DataFrame({"symbol": [], "event_date": [], "source": [], "content": []})

    def journal_search(self, **kwargs: Any) -> list[tuple[Any, float]]:
        return []

    def similar_experiences(self, **kwargs: Any) -> list[DecisionExperience]:
        return []


class FakeExperienceRepo:
    def __init__(self) -> None:
        self.inserted: list[DecisionExperience] = []

    def insert(self, exp: DecisionExperience) -> bool:
        self.inserted.append(exp)
        return True


class FakeRunRepo:
    """Records EVERY write the driver makes — the unit-level firewall: the
    only verdict sink is insert_verdicts (replay_verdicts); there is no
    committee_verdicts write method to call."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.statuses: list[tuple[str, str]] = []
        self.verdicts: list[dict[str, Any]] = []
        self.contexts: list[dict[str, Any]] = []
        self.spends: list[tuple[str, float]] = []

    def create_run(self, **kwargs: Any) -> None:
        self.runs.append(kwargs)

    def update_status(self, run_id: str, status: str) -> None:
        self.statuses.append((run_id, status))

    def insert_verdicts(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.verdicts.extend(rows)
        return len(rows)

    def insert_contexts(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.contexts.extend(rows)
        return len(rows)

    def record_spend(self, run_id: str, delta_usd: float) -> None:
        self.spends.append((run_id, delta_usd))


@pytest.fixture(scope="module")
def frames() -> tuple[list[date], pd.DataFrame, pd.DataFrame]:
    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    return sessions, make_bars(sessions), make_scores(sessions)


def _config(frames: Any, **overrides: Any) -> ReplayConfig:
    sessions, _, _ = frames
    window = sessions[-RUN_SESSIONS:]
    defaults: dict[str, Any] = dict(
        start=window[0],
        end=window[-1],
        mode="committee",
        label="committee-test",
        horizon_sessions=HORIZON,
        committee_deep_dives_per_day=2,
    )
    defaults.update(overrides)
    return ReplayConfig(**defaults)


def _replay(
    frames: Any,
    *,
    cap: float = 100.0,
    cost_per_batch: float = 0.0,
    news: list[NewsItem] | None = None,
    **replay_kwargs: Any,
) -> tuple[CommitteeReplay, FakeResearchModel, FakeRunRepo, FakeExperienceRepo, ReplayConfig]:
    _, bars, scores = frames
    config = _config(frames)
    run_repo = FakeRunRepo()
    exp_repo = FakeExperienceRepo()
    budget = ReplayBudget(cap_usd=cap, run_repo=run_repo, run_id=config.run_id)
    model = FakeResearchModel(budget=budget, cost_per_batch=cost_per_batch)
    replay = CommitteeReplay(
        **replay_kwargs,
        gateway_factory=lambda d: CommitteeGateway(d, bars, scores, news or []),
        research_model=model,
        budget=budget,
        run_repo=run_repo,
        experience_repo=exp_repo,
        config=config,
        tier_models=TIER_MODELS,
        prompts=PROMPTS,
        schemas=SCHEMAS,
        sleep_fn=lambda s: pytest.fail("batch already ended — no sleep expected"),
        # The June-2026 test window sits inside the opus-5 documented training
        # window — acknowledged here exactly like the campaign runbook would.
        acknowledge_model_knowledge=True,
    )
    return replay, model, run_repo, exp_repo, config


# --------------------------------------------------------------------------
# mode / shortlist / context assembly
# --------------------------------------------------------------------------


def test_requires_committee_mode(frames: Any) -> None:
    _, bars, scores = frames
    with pytest.raises(ValueError, match="committee"):
        CommitteeReplay(
            gateway_factory=lambda d: CommitteeGateway(d, bars, scores),
            research_model=FakeResearchModel(),
            budget=ReplayBudget(cap_usd=1.0, run_repo=FakeRunRepo(), run_id="r"),
            run_repo=FakeRunRepo(),
            experience_repo=FakeExperienceRepo(),
            config=_config(frames, mode="quant"),
            tier_models=TIER_MODELS,
            prompts=PROMPTS,
            schemas=SCHEMAS,
        )


def test_shortlist_reuses_runner_scoring_capped(frames: Any) -> None:
    sessions, bars, scores = frames
    day = sessions[-1]
    gw = CommitteeGateway(day, bars, scores)
    all_scores, shortlist = committee_shortlist(gw, _config(frames), day)
    # Runner parity on the SCORING side; selection is the diversity selector
    # (campaign findings item 19) with an empty history: rank order, capped
    # at committee_deep_dives_per_day (=2). EER ranks make ILLQ + LIQ the two
    # best candidates (see test_runner fixtures).
    assert shortlist == ["ILLQ", "LIQ"]
    assert set(all_scores) == set(UNIVERSE)
    one = committee_shortlist(gw, _config(frames, committee_deep_dives_per_day=1), day)[1]
    assert one == ["ILLQ"]


def test_shortlist_cooldown_forces_a_new_name_the_next_session(frames: Any) -> None:
    """The fixation fix at driver level: yesterday's slate is on cooldown, so
    today's pulls the next candidate in (AAA, which the old top-k slice would
    never have reached on a frozen ranking)."""
    sessions, bars, scores = frames
    day = sessions[-1]
    gw = CommitteeGateway(day, bars, scores)
    history = ((sessions[-2], ("ILLQ", "LIQ")),)
    _, shortlist = committee_shortlist(gw, _config(frames), day, history=history)
    assert "AAA" in shortlist
    # …and the wider candidate band is what made AAA reachable at all: at the
    # old top-20% band it is not even a candidate.
    narrow = _config(frames, shortlist_candidate_percentile=0.8)
    assert "AAA" not in committee_shortlist(gw, narrow, day, history=history)[1]


def test_shortlist_is_deterministic_for_the_same_config_and_history(frames: Any) -> None:
    sessions, bars, scores = frames
    day = sessions[-1]
    history = ((sessions[-2], ("ILLQ",)),)
    slates = {
        tuple(
            committee_shortlist(
                CommitteeGateway(day, bars, scores), _config(frames), day, history=history
            )[1]
        )
        for _ in range(5)
    }
    assert len(slates) == 1


def test_dossier_context_from_gateway_only_future_news_invisible(frames: Any) -> None:
    sessions, bars, scores = frames
    day = sessions[-3]
    past_item = NewsItem(
        id="n-past",
        symbols=("LIQ",),
        headline="LIQ wins a big contract",
        source="test",
        published_at=datetime(day.year, day.month, day.day, 9, tzinfo=UTC) - timedelta(days=1),
    )
    future_item = NewsItem(
        id="n-future",
        symbols=("LIQ",),
        headline="LIQ FUTURE-EVENT headline",
        source="test",
        published_at=datetime(day.year, day.month, day.day, 9, tzinfo=UTC) + timedelta(days=2),
    )
    gw = CommitteeGateway(day, bars, scores, news=[past_item, future_item])
    all_scores, _ = committee_shortlist(gw, _config(frames), day)
    contexts, _hits = build_replay_contexts(gw, ["LIQ"], all_scores)
    ctx = contexts["LIQ"]
    # As-of data present…
    assert "LIQ wins a big contract" in ctx
    assert f"macro digest as of {day.isoformat()}" in ctx
    assert "## Price/volume stats" in ctx and "close:" in ctx
    # …future data invisible; absent sources render like thin live dossiers.
    assert "FUTURE-EVENT" not in ctx
    assert "## Filing excerpts\nnone" in ctx
    assert "## Journal hits\nnone" in ctx


def _facts_frame(symbol: str) -> pd.DataFrame:
    """xbrl-shaped rows: three quarters plus a fiscal-year row in the Q4 slot."""
    quarters = (
        ("2025-03-31", 100.0, "2025-04-20"),
        ("2025-06-30", 110.0, "2025-07-20"),
        ("2025-09-30", 120.0, "2025-10-20"),
        ("2025-12-31", 460.0, "2026-02-20"),
        ("2026-03-31", 130.0, "2026-04-20"),
    )
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "tag": "Revenues",
                "period_end": pd.Timestamp(period),
                "value": value * 1e6,
                "filed_at": datetime.fromisoformat(filed).replace(tzinfo=UTC),
            }
            for period, value, filed in quarters
        ]
    )


class FundamentalsGateway(CommitteeGateway):
    """CommitteeGateway + the dossier's dedicated fundamentals read."""

    def fundamentals(self, symbols: Any = None, lookback_days: int = 1500) -> pd.DataFrame:
        return _facts_frame("LIQ")


def test_dossier_carries_fundamentals_peers_and_filtered_news(frames: Any) -> None:
    sessions, bars, scores = frames
    day = sessions[-1]
    stamp = datetime(day.year, day.month, day.day, 9, tzinfo=UTC)
    own = NewsItem(
        id="n-own",
        symbols=("LIQ",),
        headline="LIQ raises guidance",
        source="test",
        published_at=stamp - timedelta(days=1),
    )
    unrelated = NewsItem(
        id="n-other",
        symbols=("LIQ",),
        headline="Rival chipmaker cuts prices",
        source="test",
        published_at=stamp - timedelta(days=2),
    )
    gw = FundamentalsGateway(day, bars, scores, news=[own, unrelated])
    all_scores, _ = committee_shortlist(gw, _config(frames), day)
    ctx = build_replay_contexts(gw, ["LIQ"], all_scores)[0]["LIQ"]

    # Fundamentals: the fiscal-year row is differenced into a Q4 (460 - 100 -
    # 110 - 120 = 130), so TTM is 110 + 120 + 130 + 130 = 490, not 820.
    assert "## Fundamentals" in ctx
    assert "revenue_ttm: 490.0M USD [xbrl:LIQ:Revenues:2025-06-30..2026-03-31" in ctx

    # Peers: relative strength against the benchmark in the bars fixture.
    assert "## Peer / relative strength" in ctx
    assert "excess_vs_SPY_20d_pct:" in ctx

    # Extended price stats beyond the original four.
    assert "high_52w:" in ctx and "max_drawdown_52w_pct:" in ctx

    # News: only the ticker's own item leads; the rest is labelled market news.
    digest = ctx.split("## News digest", 1)[1].split("## Market news", 1)[0]
    assert "[n-own]" in digest and "[n-other]" not in digest
    assert "[n-other]" in ctx.split("## Market news", 1)[1]


def test_journal_retrieval_is_symbol_scoped_in_the_query(frames: Any) -> None:
    """The journal is symbol-scoped evidence: the symbol must reach the query,
    not just ``rank_similar``'s boost, or a weak embedding backend serves
    other tickers' entries as this ticker's 'similar situations'."""
    from dynamo.replay.committee import replay_journal_hits

    sessions, bars, scores = frames
    seen: list[dict[str, Any]] = []

    class RecordingGateway(CommitteeGateway):
        def journal_search(self, **kwargs: Any) -> list[tuple[Any, float]]:
            seen.append(kwargs)
            return []

    class Embedder:
        def embed(self, texts: Any) -> list[list[float]]:
            return [[0.0] * 384 for _ in list(texts)]

    gw = RecordingGateway(sessions[-1], bars, scores)
    replay_journal_hits(gw, ["LIQ"], embedder=Embedder())
    assert seen and seen[0]["symbols"] == ["LIQ"]


def test_replay_section_signature_matches_the_live_builder(frames: Any) -> None:
    """Both call sites must render the same sections in the same order —
    ``tests/unit/orchestration/test_dossier_contexts.py`` asserts this list."""
    sessions, bars, scores = frames
    day = sessions[-1]
    gw = FundamentalsGateway(day, bars, scores)
    all_scores, _ = committee_shortlist(gw, _config(frames), day)
    ctx = build_replay_contexts(gw, ["LIQ"], all_scores)[0]["LIQ"]
    assert [line for line in ctx.splitlines() if line.startswith("## ")] == [
        "## Instrument profile",
        "## Fundamentals",
        "## Quant score",
        "## Price/volume stats",
        "## Peer / relative strength",
        "## Filing excerpts",
        "## News digest",
        "## Macro digest",
        "## Journal hits",
    ]


def test_dossier_falls_back_to_feature_inputs_without_a_fundamentals_read(
    frames: Any,
) -> None:
    """A gateway without the dedicated read still gets a fundamentals section."""
    sessions, bars, scores = frames
    day = sessions[-1]

    class FeatureInputsOnlyGateway(CommitteeGateway):
        def feature_inputs(self, symbols: Any) -> Any:
            from dynamo.research.quant.catalog import FeatureInputs

            return FeatureInputs(fundamentals=_facts_frame("LIQ"))

    gw = FeatureInputsOnlyGateway(day, bars, scores)
    all_scores, _ = committee_shortlist(gw, _config(frames), day)
    ctx = build_replay_contexts(gw, ["LIQ"], all_scores)[0]["LIQ"]
    assert "revenue_ttm: 490.0M USD" in ctx


# --------------------------------------------------------------------------
# custom-id namespacing
# --------------------------------------------------------------------------


def test_custom_ids_are_namespaced_by_run(frames: Any) -> None:
    replay, model, _, _, config = _replay(frames)
    replay.run()
    assert model.submitted, "stages were submitted"
    for batch in model.submitted:
        for req in batch:
            cid = str(req["custom_id"])
            assert cid.startswith(f"{config.run_id}:")
            live_id = strip_namespace(config.run_id, cid)
            _as_of, symbol, stage = split_custom_id(live_id)
            assert symbol in UNIVERSE and stage in ("dossier", "bull", "bear", "arbiter")
            assert namespace_custom_id(config.run_id, live_id) == cid
    with pytest.raises(ValueError):
        strip_namespace("other-run", f"{config.run_id}:2026-06-01-LIQ-dossier")


# --------------------------------------------------------------------------
# happy path: verdicts -> replay sink only; experiences resolve in-window
# --------------------------------------------------------------------------


def test_verdicts_go_to_replay_sink_only(frames: Any) -> None:
    replay, model, run_repo, _exp_repo, config = _replay(frames)
    result = replay.run()
    assert result.status == "completed"
    assert run_repo.statuses[-1] == (config.run_id, "completed")
    assert run_repo.runs[0]["mode"] == "committee"
    assert run_repo.runs[0]["budget_usd"] == pytest.approx(100.0)
    # Verdicts landed exclusively in the replay-verdict sink, run-tagged.
    assert result.deep_dives and len(run_repo.verdicts) == len(result.deep_dives)
    for row in run_repo.verdicts:
        assert row["run_id"] == config.run_id
        assert row["id"].startswith(f"{config.run_id}:")
        assert row["action"] == "buy"
        assert row["payload"]["prompt_version"] == "arbiter-v1"
        assert row["payload"]["model_id"] == TIER_MODELS["arbiter"]
    # Deep dives ran the full 4-stage flow per day: dossier, advocates, arbiter.
    assert len(model.submitted) % 3 == 0


def test_entry_verdicts_resolve_into_replay_experiences(frames: Any) -> None:
    replay, _, _, exp_repo, config = _replay(frames)
    result = replay.run()
    assert result.experiences, "entry verdicts inside the window must resolve"
    assert [e.id for e in exp_repo.inserted] == [e.id for e in result.experiences]
    sessions = list(result.sessions)
    for exp in result.experiences:
        assert exp.source == "replay"
        assert exp.replay_run_id == config.run_id
        assert exp.resolved and exp.label in ("win", "loss")
        assert exp.action == "buy" and exp.conviction == pytest.approx(0.75)
        # Entry at the NEXT session's open, never the decision day.
        assert exp.entry_at is not None and exp.exit_at is not None
        entry_day = exp.entry_at.date()
        assert entry_day in sessions and entry_day > exp.as_of
        assert exp.exit_at.date() <= sessions[-1]
    # A verdict on the final session has no next open — never resolves.
    final_day = sessions[-1]
    assert all(e.as_of != final_day for e in result.experiences)


# --------------------------------------------------------------------------
# budget: refusal, budget_exhausted status, partial results preserved
# --------------------------------------------------------------------------


def test_tiny_cap_refuses_first_submission_and_marks_exhausted(frames: Any) -> None:
    replay, model, run_repo, _exp_repo, config = _replay(frames, cap=0.0001)
    result = replay.run()
    assert result.status == "budget_exhausted"
    assert (config.run_id, "budget_exhausted") in run_repo.statuses
    assert ("failed" not in {s for _, s in run_repo.statuses})
    assert model.submitted == []  # refused BEFORE any submission
    assert result.deep_dives == () and run_repo.verdicts == []


def test_exhaustion_midway_keeps_partial_results(frames: Any) -> None:
    # Each batch "costs" $1.00 of metered spend. Day 1 (3 stage batches)
    # spends $3.00; day 2's projections then blow the $3.50 cap and the run
    # stops — with day 1's verdicts persisted and its experiences resolved.
    replay, _model, run_repo, _exp_repo, config = _replay(
        frames, cap=3.5, cost_per_batch=1.0
    )
    result = replay.run()
    assert result.status == "budget_exhausted"
    assert (config.run_id, "budget_exhausted") in run_repo.statuses
    day1 = result.sessions[0]
    assert result.deep_dives, "day-1 deep dives are kept"
    assert {d.as_of for d in result.deep_dives} == {day1}
    assert {r["as_of"] for r in run_repo.verdicts} == {day1}
    # Partial experiences: day-1 entries still resolve inside the window.
    assert result.experiences
    assert all(e.as_of == day1 for e in result.experiences)
    # Durable checkpoints flowed through record_spend as the money was spent.
    assert run_repo.spends and sum(d for _, d in run_repo.spends) == pytest.approx(
        result.spent_usd
    )


# --------------------------------------------------------------------------
# incremental resume: days with persisted verdicts are never re-submitted
# --------------------------------------------------------------------------


def _submitted_days(model: FakeResearchModel, run_id: str) -> set[date]:
    days: set[date] = set()
    for batch in model.submitted:
        for req in batch:
            live_id = strip_namespace(run_id, str(req["custom_id"]))
            as_of, _symbol, _stage = split_custom_id(live_id)
            days.add(as_of)
    return days


def test_resume_skip_days_are_never_resubmitted(frames: Any) -> None:
    """A resumed run (same deterministic run id) with days already holding
    persisted verdicts submits NOTHING for those days — no re-spend, no
    plan-usage burn — and still completes the remaining days."""
    sessions, _, _ = frames
    window = sessions[-RUN_SESSIONS:]
    done = frozenset(window[:2])
    replay, model, run_repo, _exp_repo, config = _replay(
        frames, resume_skip_days=done
    )
    progress: list[str] = []
    result = replay.run(progress=progress.append)
    assert result.status == "completed"
    submitted = _submitted_days(model, config.run_id)
    assert submitted and submitted.isdisjoint(done)
    assert submitted == set(window[2:]) & submitted  # only non-skipped days
    assert {r["as_of"] for r in run_repo.verdicts}.isdisjoint(done)
    assert sum("skipped (resume)" in line for line in progress) == 2


def test_run_rotates_names_across_sessions_via_the_cooldown(frames: Any) -> None:
    """Across a whole run the driver carries its own debate history, so the
    slate rotates instead of repeating the top of a frozen ranking — the
    control run debated WMT on 52 of 74 days."""
    replay, _model, run_repo, _exp_repo, _cfg = _replay(frames)
    replay.run()
    debated = {r["symbol"] for r in run_repo.verdicts}
    assert "AAA" in debated, "the cooldown must force a third name into the run"
    per_day = {}
    for row in run_repo.verdicts:
        per_day.setdefault(row["as_of"], set()).add(row["symbol"])
    assert len({frozenset(v) for v in per_day.values()}) > 1  # not one frozen slate


def test_resume_seeds_the_cooldown_from_persisted_verdicts(frames: Any) -> None:
    """A resumed run must not restart its diversity windows empty: the days
    already persisted seed the cooldown, so the first resumed session does
    not simply re-debate what the previous attempt just covered."""
    sessions, _bars, _scores = frames
    window = sessions[-RUN_SESSIONS:]
    done = window[:1]
    replay, _model, run_repo, _exp_repo, _cfg = _replay(
        frames,
        resume_skip_days=frozenset(done),
        prior_dives={done[0]: ["ILLQ", "LIQ"]},
    )
    replay.run()
    first_resumed = min(r["as_of"] for r in run_repo.verdicts)
    assert first_resumed == window[1]
    assert "AAA" in {
        r["symbol"] for r in run_repo.verdicts if r["as_of"] == first_resumed
    }


def test_no_resume_skip_by_default(frames: Any) -> None:
    replay, model, _run_repo, _exp_repo, config = _replay(frames)
    replay.run()
    assert len(_submitted_days(model, config.run_id)) == RUN_SESSIONS  # every day submitted


# --------------------------------------------------------------------------
# ClaudeCodeResearchModel through CommitteeReplay (ADR-0014): the eager
# sequential batch loop, $0 metering, plan-limit soft-fail + resume
# --------------------------------------------------------------------------

CLAUDE_CODE_TIERS = {
    "screen": "claude-haiku-4-5",
    "analyst": "claude-sonnet-5",
    "arbiter": "claude-fable-5",
}

_SYSTEM_TO_STAGE = {
    "system prompt for dossier_analyst": "dossier",
    "system prompt for bull_advocate": "bull",
    "system prompt for bear_advocate": "bear",
    "system prompt for arbiter": "arbiter",
}


class StageCliRunner:
    """Fake `claude -p` runner: reads the stage off --system-prompt and
    returns a canned JSON envelope. After ``fail_after`` calls it reports a
    plan-window usage limit (the subscription exhaustion shape)."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[str] = []
        self._fail_after = fail_after

    def __call__(self, cmd: list[str], prompt: str, timeout: float) -> Any:
        import json as _json

        from dynamo.llm.claude_code_client import CliResult

        stage = _SYSTEM_TO_STAGE[cmd[cmd.index("--system-prompt") + 1]]
        self.calls.append(stage)
        if self._fail_after is not None and len(self.calls) > self._fail_after:
            return CliResult(1, "", "5-hour limit reached - resets at 6pm")
        envelope = {
            "is_error": False,
            "subtype": "success",
            "result": _json.dumps(canned_payload(stage, "X")),
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        return CliResult(0, _json.dumps(envelope), "")


def _claude_code_replay(
    frames: Any, runner: StageCliRunner, run_repo: FakeRunRepo
) -> tuple[CommitteeReplay, ReplayBudget, ReplayConfig]:
    from dynamo.config.settings import Settings
    from dynamo.llm.claude_code_client import ClaudeCodeResearchModel
    from dynamo.llm.cost_meter import CostMeter, InMemoryLedgerWriter
    from dynamo.replay.budget import BudgetLedgerTee

    _, bars, scores = frames
    config = _config(frames)
    exp_repo = FakeExperienceRepo()
    budget = ReplayBudget(cap_usd=100.0, run_repo=run_repo, run_id=config.run_id)
    meter = CostMeter(BudgetLedgerTee(InMemoryLedgerWriter(), budget), 90.0)
    model = ClaudeCodeResearchModel(
        Settings(ANTHROPIC_API_KEY=""),
        meter,
        binary="/fake/claude",
        runner=runner,
    )
    done = frozenset(r["as_of"] for r in run_repo.verdicts)
    replay = CommitteeReplay(
        gateway_factory=lambda d: CommitteeGateway(d, bars, scores),
        research_model=model,
        budget=budget,
        run_repo=run_repo,
        experience_repo=exp_repo,
        config=config,
        tier_models=CLAUDE_CODE_TIERS,
        prompts=PROMPTS,
        schemas=SCHEMAS,
        sleep_fn=lambda s: pytest.fail("eager local batches end immediately — no sleep"),
        resume_skip_days=done,
    )
    return replay, budget, config


def test_claude_code_eager_batches_flow_through_submit_and_collect(frames: Any) -> None:
    """The CLI backend's eager sequential submit_batch (status immediately
    'ended') satisfies CommitteeReplay's submit -> poll -> collect loop with
    ZERO sleeps; verdicts/contexts persist normally and every metered entry
    is $0 (flat-rate) so the budget never accumulates spend."""
    run_repo = FakeRunRepo()
    runner = StageCliRunner()
    replay, budget, _config = _claude_code_replay(frames, runner, run_repo)
    result = replay.run()
    assert result.status == "completed"
    assert result.deep_dives and run_repo.verdicts
    # The June-2026 window is CLEAN for the claude_code tiers (fable/sonnet
    # cutoffs 2026-01-31 + 30d margin) — no acknowledgement needed.
    gate = replay._run_config_extras["model_knowledge"]
    assert gate["blocked"] is False
    assert gate["cutoffs"]["claude-fable-5"] == "2026-01-31"
    # 4-stage flow ran through the real CLI adapter for every dive.
    assert set(runner.calls) == {"dossier", "bull", "bear", "arbiter"}
    # Flat-rate: real token counts metered at $0 — no spend, no checkpoints.
    assert budget.spent_usd == 0.0
    assert run_repo.spends == []
    assert result.spent_usd == 0.0 and result.cap_usd == pytest.approx(100.0)
    # Decision-time contexts persisted before every submission.
    assert run_repo.contexts
    assert {r["model_id"] for r in run_repo.contexts} <= set(CLAUDE_CODE_TIERS.values())


def test_claude_code_plan_exhaustion_soft_fails_then_resumes(frames: Any) -> None:
    """Plan-window exhaustion mid-run: the exhausted batch marks its requests
    failed (None), the affected days yield no verdicts ($0), and the run
    still completes with partial results. A second run of the SAME config
    (resume_skip_days = days with persisted verdicts) submits ONLY the
    missing days and completes the window."""
    sessions, _, _ = frames
    window = sessions[-RUN_SESSIONS:]
    run_repo = FakeRunRepo()
    # Day 1 = 2 dossier + 2 bull + 2 bear + 2 arbiter = 8 CLI calls; the 9th
    # call (day 2's first dossier) hits the subscription limit.
    runner = StageCliRunner(fail_after=8)
    replay, budget, config = _claude_code_replay(frames, runner, run_repo)
    result = replay.run()
    assert result.status == "completed"  # soft-fail, not a crash
    assert {r["as_of"] for r in run_repo.verdicts} == {window[0]}
    assert budget.spent_usd == 0.0

    # The plan window reset — the SAME deterministic run id resumes: day 1
    # is skipped (no resubmission), days 2..N run to completion.
    healthy = StageCliRunner()
    resumed, _budget2, config2 = _claude_code_replay(frames, healthy, run_repo)
    assert config2.run_id == config.run_id  # deterministic resume identity
    result2 = resumed.run()
    assert result2.status == "completed"
    verdict_days = {r["as_of"] for r in run_repo.verdicts}
    assert verdict_days == set(window)  # the resumed run filled in days 2..N
    # Day 1 was never re-submitted on resume: the healthy runner only saw
    # the remaining days' stages.
    assert len(healthy.calls) == 8 * (RUN_SESSIONS - 1)
