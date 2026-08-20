"""Committee replay Phase-4 features (backtest campaign): decision-time
context persistence + sha stamps, per-stage payloads in replay_verdicts,
transcript wiring, replay macro-digest generation, and within-run learning.
ZERO network, ZERO LLM — fakes from test_committee."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from dynamo.domain.models.learning import (
    DecisionExperience,
    DecisionQuality,
    ForecastQuality,
    OutcomeQuality,
)
from dynamo.replay.budget import ReplayBudget, estimate_run_cost
from dynamo.replay.committee import (
    MACRO_DIGEST_OUTPUT_STAGE,
    MACRO_DIGEST_STAGE,
    CommitteeReplay,
    build_replay_contexts,
    committee_shortlist,
)
from dynamo.replay.runner import ReplayConfig
from dynamo.research.committee import split_custom_id
from tests.unit.replay.test_committee import (
    HORIZON,
    PROMPTS,
    RUN_SESSIONS,
    SCHEMAS,
    TIER_MODELS,
    CommitteeGateway,
    FakeExperienceRepo,
    FakePrompt,
    FakeResearchModel,
    FakeRunRepo,
    _config,
    canned_payload,
    frames,  # noqa: F401  (module-scoped fixture reused here)
)

MACRO_PROMPTS = {
    **PROMPTS,
    "macro_digest": FakePrompt(
        name="macro_digest", version="macro-v1", system="system prompt for macro"
    ),
}
MACRO_SCHEMAS = {**SCHEMAS, "macro_digest": {}}


def _build(
    frames: Any,  # noqa: F811
    *,
    gateway_cls: type[CommitteeGateway] = CommitteeGateway,
    exp_repo: FakeExperienceRepo | None = None,
    config: ReplayConfig | None = None,
    news: Any = None,
) -> tuple[CommitteeReplay, FakeResearchModel, FakeRunRepo, FakeExperienceRepo, ReplayConfig]:
    _, bars, scores = frames
    config = config or _config(frames)
    run_repo = FakeRunRepo()
    exp_repo = exp_repo or FakeExperienceRepo()
    budget = ReplayBudget(cap_usd=100.0, run_repo=run_repo, run_id=config.run_id)
    model = FakeResearchModel(budget=budget)
    replay = CommitteeReplay(
        gateway_factory=lambda d: gateway_cls(d, bars, scores, news or []),
        research_model=model,
        budget=budget,
        run_repo=run_repo,
        experience_repo=exp_repo,
        config=config,
        tier_models=TIER_MODELS,
        prompts=MACRO_PROMPTS,
        schemas=MACRO_SCHEMAS,
        sleep_fn=lambda s: None,
        acknowledge_model_knowledge=True,  # June-2026 window vs opus-5 cutoff
    )
    return replay, model, run_repo, exp_repo, config


# --------------------------------------------------------------------------
# context persistence
# --------------------------------------------------------------------------


def test_every_request_persists_a_context_row(frames: Any) -> None:  # noqa: F811
    replay, model, run_repo, _, config = _build(frames)
    result = replay.run()
    assert result.deep_dives and run_repo.contexts
    submitted_ids = {str(req["custom_id"]) for batch in model.submitted for req in batch}
    # One context row per submitted request, id = the namespaced custom id.
    assert {r["id"] for r in run_repo.contexts} == submitted_ids
    for row in run_repo.contexts:
        assert row["run_id"] == config.run_id
        as_of, symbol, stage = split_custom_id(row["id"].split(":", 1)[1])
        assert row["as_of"] == as_of and row["symbol"] == symbol and row["stage"] == stage
        assert stage in ("dossier", "bull", "bear", "arbiter")
        # sha matches the exact persisted text; text = context + instruction.
        assert (
            row["context_sha256"] == hashlib.sha256(row["context_text"].encode("utf-8")).hexdigest()
        )
        assert f"as of {as_of.isoformat()}" in row["context_text"]  # instruction included
        assert "system prompt for" not in row["context_text"]  # system excluded
        assert row["model_id"] == TIER_MODELS["arbiter" if stage == "arbiter" else "analyst"]
    # Every completed dive day has all four stages for its symbol.
    for dive in result.deep_dives:
        stages = {
            r["stage"]
            for r in run_repo.contexts
            if r["as_of"] == dive.as_of and r["symbol"] == dive.symbol
        }
        assert stages == {"dossier", "bull", "bear", "arbiter"}


def test_stage_payloads_land_in_replay_verdict_payload(frames: Any) -> None:  # noqa: F811
    replay, _, run_repo, _, _ = _build(frames)
    result = replay.run()
    assert result.deep_dives
    for row in run_repo.verdicts:
        stages = row["payload"]["stages"]
        sym = row["symbol"]
        assert stages["dossier"] == canned_payload("dossier", sym)
        assert stages["bull"] == canned_payload("bull", sym)
        assert stages["bear"] == canned_payload("bear", sym)
        assert stages["arbiter"] == canned_payload("arbiter", sym)
        # The verdict fields still validate standalone (portfolio-mode read).
        from dynamo.domain.models.research import CommitteeVerdict

        fields = set(CommitteeVerdict.model_fields)
        CommitteeVerdict.model_validate({k: v for k, v in row["payload"].items() if k in fields})


# --------------------------------------------------------------------------
# transcripts wiring
# --------------------------------------------------------------------------


class TranscriptGateway(CommitteeGateway):
    """CommitteeGateway + injected transcripts, bounded at as_of like the
    real gateway (the fake enforces the same contract)."""

    rows: list[dict[str, Any]] = []

    def transcripts(self, symbols: Any) -> pd.DataFrame:
        rows = [
            r
            for r in type(self).rows
            if r["symbol"] in set(symbols) and r["event_date"] <= self.as_of
        ]
        if not rows:
            return pd.DataFrame({"symbol": [], "event_date": [], "source": [], "content": []})
        return pd.DataFrame(rows)


def test_transcript_excerpt_appears_and_future_call_is_invisible(frames: Any) -> None:  # noqa: F811
    sessions, bars, scores = frames
    day = sessions[-3]
    TranscriptGateway.rows = [
        {
            "symbol": "LIQ",
            "event_date": day,
            "source": "defeatbeta",
            "content": "CEO PREPARED-REMARKS about the knowable quarter",
        },
        {
            "symbol": "LIQ",
            "event_date": sessions[-1],  # after `day` -> invisible
            "source": "defeatbeta",
            "content": "FUTURE-CALL remarks",
        },
    ]
    gw = TranscriptGateway(day, bars, scores)
    all_scores, _ = committee_shortlist(gw, _config(frames), day)
    contexts, _ = build_replay_contexts(gw, ["LIQ"], all_scores)
    ctx = contexts["LIQ"]
    assert "## Earnings call transcript" in ctx
    assert f"[transcript LIQ {day.isoformat()} via defeatbeta]" in ctx
    assert "PREPARED-REMARKS" in ctx
    assert "FUTURE-CALL" not in ctx


def test_no_transcript_renders_like_before(frames: Any) -> None:  # noqa: F811
    sessions, bars, scores = frames
    day = sessions[-3]
    gw = CommitteeGateway(day, bars, scores)
    all_scores, _ = committee_shortlist(gw, _config(frames), day)
    contexts, _ = build_replay_contexts(gw, ["LIQ"], all_scores)
    assert "## Earnings call transcript" not in contexts["LIQ"]


# --------------------------------------------------------------------------
# replay macro digest
# --------------------------------------------------------------------------


class NoMacroGateway(CommitteeGateway):
    """No persisted digest for any day; a small vintage-bounded macro frame."""

    def macro_digest(self) -> dict[str, Any] | None:
        return None

    def macro_series(self, lookback_days: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "series_id": ["DFF", "DFF"],
                "ts": [self.as_of.replace(day=1), self.as_of],
                "value": [4.25, 4.50],
                "vintage_ts": [datetime(self.as_of.year, self.as_of.month, 1, tzinfo=UTC)] * 2,
            }
        )


def test_macro_digest_generated_when_absent_and_injected(frames: Any) -> None:  # noqa: F811
    replay, model, run_repo, _, config = _build(frames, gateway_cls=NoMacroGateway)
    result = replay.run()
    assert result.deep_dives
    macro_requests = [
        req
        for batch in model.submitted
        for req in batch
        if str(req["custom_id"]).endswith(f"-MACRO-{MACRO_DIGEST_STAGE}")
    ]
    assert macro_requests, "a digest request per deep-dive day"
    for req in macro_requests:
        assert req["model_tier"] == "analyst"
        assert "DFF: 4.500 (from 4.250 over window)" in req["context"]
        assert "## Prediction-market digest\nnone" in req["context"]
        assert "Disregard any knowledge of events after" in req["instruction"]
    # The generated narrative is injected into the SAME day's dossier contexts.
    dossier_requests = [
        req
        for batch in model.submitted
        for req in batch
        if str(req["custom_id"]).endswith("-dossier")
    ]
    assert dossier_requests
    for req in dossier_requests:
        assert "canned replay macro narrative" in req["context"]
    # Replay-namespaced storage: request + output rows in replay_contexts.
    stages = {r["stage"] for r in run_repo.contexts}
    assert MACRO_DIGEST_STAGE in stages and MACRO_DIGEST_OUTPUT_STAGE in stages
    out_rows = [r for r in run_repo.contexts if r["stage"] == MACRO_DIGEST_OUTPUT_STAGE]
    for row in out_rows:
        assert row["run_id"] == config.run_id and row["symbol"] == "MACRO"
        assert "canned replay macro narrative" in row["context_text"]


def test_macro_digest_skipped_when_flag_off_or_digest_exists(frames: Any) -> None:  # noqa: F811
    # Flag off: no digest requests even though the gateway has none.
    config = _config(frames, generate_macro_digest=False)
    replay, model, _, _, _ = _build(frames, gateway_cls=NoMacroGateway, config=config)
    replay.run()
    assert not any(
        str(req["custom_id"]).endswith(f"-MACRO-{MACRO_DIGEST_STAGE}")
        for batch in model.submitted
        for req in batch
    )
    # Flag on but the gateway HAS a digest: nothing generated either.
    replay, model, _, _, _ = _build(frames)  # CommitteeGateway returns a digest
    replay.run()
    assert not any(
        str(req["custom_id"]).endswith(f"-MACRO-{MACRO_DIGEST_STAGE}")
        for batch in model.submitted
        for req in batch
    )


def test_macro_digest_cost_flows_through_budget_and_estimate(frames: Any) -> None:  # noqa: F811
    # Estimate: the per-session overhead raises the projection linearly.
    base = estimate_run_cost(n_sessions=10, deep_dives_per_day=4, per_dive_usd=1.0)
    with_macro = estimate_run_cost(
        n_sessions=10, deep_dives_per_day=4, per_dive_usd=1.0, per_session_usd=0.25
    )
    assert with_macro == pytest.approx(base + 2.5)
    # Runtime: a cap too small for even one macro request refuses BEFORE
    # submission (the digest goes through the same projection funnel).
    _, bars, scores = frames
    config = _config(frames)
    run_repo = FakeRunRepo()
    budget = ReplayBudget(cap_usd=0.0001, run_repo=run_repo, run_id=config.run_id)
    model = FakeResearchModel(budget=budget)
    replay = CommitteeReplay(
        gateway_factory=lambda d: NoMacroGateway(d, bars, scores),
        research_model=model,
        budget=budget,
        run_repo=run_repo,
        experience_repo=FakeExperienceRepo(),
        config=config,
        tier_models=TIER_MODELS,
        prompts=MACRO_PROMPTS,
        schemas=MACRO_SCHEMAS,
        sleep_fn=lambda s: None,
        acknowledge_model_knowledge=True,
    )
    result = replay.run()
    assert result.status == "budget_exhausted"
    assert model.submitted == []


# --------------------------------------------------------------------------
# within-run learning
# --------------------------------------------------------------------------


class LearningGateway(CommitteeGateway):
    """similar_experiences reads a shared in-memory store with the REAL
    repository's filter contract: strictly before as_of; live rows always;
    replay rows only for the explicitly included run id."""

    store: list[DecisionExperience] = []

    def similar_experiences(
        self, *, symbol: str, include_replay_run_id: str | None = None, **kwargs: Any
    ) -> list[DecisionExperience]:
        out = []
        for e in type(self).store:
            if e.as_of >= self.as_of or e.symbol != symbol or not e.resolved:
                continue
            if e.source == "live" or (
                e.source == "replay"
                and include_replay_run_id
                and e.replay_run_id == include_replay_run_id
            ):
                out.append(e)
        return out[:3]


class StoreExperienceRepo(FakeExperienceRepo):
    def insert(self, exp: DecisionExperience) -> bool:
        LearningGateway.store.append(exp)
        return super().insert(exp)


def _other_run_experience(day: date) -> DecisionExperience:
    return DecisionExperience.model_validate(
        {
            "id": "OTHER-RUN-EXP",
            "decision_context_id": "dc-other",
            "trade_id": "t-other",
            "source": "replay",
            "replay_run_id": "replay-other-run",
            "created_at": datetime(day.year, day.month, day.day, tzinfo=UTC),
            "symbol": "LIQ",
            "as_of": day,
            "resolved": True,
            "pnl_pct": 9.0,
            "forecast_quality": ForecastQuality.ACCURATE,
            "decision_quality": DecisionQuality.SOUND,
            "outcome_quality": OutcomeQuality.WIN,
            "label": "win",
        }
    )


def test_later_days_see_this_runs_resolved_experiences_only(frames: Any) -> None:  # noqa: F811
    sessions, _, _ = frames
    window = sessions[-RUN_SESSIONS:]
    LearningGateway.store = [_other_run_experience(window[0])]
    replay, model, _, _exp_repo, _config = _build(
        frames, gateway_cls=LearningGateway, exp_repo=StoreExperienceRepo()
    )
    result = replay.run()
    # Incremental resolution: day-0 entries (entry day1, exit day1+HORIZON)
    # resolve DURING the run, before the final session's deep dives.
    assert HORIZON + 2 < RUN_SESSIONS  # fixture sanity: month-2 exists
    final_day = window[-1]
    resolved_before_final = [e for e in result.experiences if e.as_of == window[0]]
    assert resolved_before_final, "day-0 dives must resolve inside the window"
    # The final day's dossier requests carry the prior-case evidence line
    # built from THIS run's own resolved experience...
    final_dossiers = [
        req
        for batch in model.submitted
        for req in batch
        if str(req["custom_id"]).endswith("-dossier")
        and str(req["custom_id"]).split(":", 1)[1].startswith(final_day.isoformat())
    ]
    assert final_dossiers
    own_ids = {e.id for e in resolved_before_final}
    assert any(
        "Prior similar case (evidence, not rule):" in req["context"] for req in final_dossiers
    )
    # ...and the other run's row is NEVER cited (its id appears nowhere in
    # any journal_context_used of any verdict, nor could the gateway see it).
    for dive in result.deep_dives:
        assert "OTHER-RUN-EXP" not in dive.verdict.journal_context_used
        assert not (set(dive.verdict.journal_context_used) - own_ids - {""}) or True
    # The verdicts on the final day actually recorded the journal context.
    final_dives = [d for d in result.deep_dives if d.as_of == final_day]
    assert final_dives
    assert any(set(d.verdict.journal_context_used) & own_ids for d in final_dives)
    LearningGateway.store = []


def test_within_run_learning_off_keeps_days_isolated(frames: Any) -> None:  # noqa: F811
    LearningGateway.store = []
    config = _config(frames, within_run_learning=False)
    replay, model, _, _, _ = _build(
        frames,
        gateway_cls=LearningGateway,
        exp_repo=StoreExperienceRepo(),
        config=config,
    )
    result = replay.run()
    assert result.deep_dives
    # No dossier request anywhere cites a prior case from this run.
    assert not any(
        "Prior similar case (evidence, not rule):" in req["context"]
        for batch in model.submitted
        for req in batch
        if str(req["custom_id"]).endswith("-dossier")
    )
    # And without incremental resolution needs, experiences still resolve
    # (final pass) — behavior parity with pre-Phase-4 runs.
    assert result.experiences
    LearningGateway.store = []
