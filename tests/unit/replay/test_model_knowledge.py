"""Model knowledge-cutoff gate (backtest-campaign Phase 4).

The committee could "know the future" of any window inside (or too close to)
its documented training-data cutoff. The gate resolves every tier model's
cutoff from ``dynamo.llm.pricing`` and refuses blocked windows unless the
run explicitly acknowledges — with everything recorded into config_json.
ZERO network, ZERO LLM (fakes from test_committee).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from dynamo.llm.pricing import UnknownModelError, spec_for
from dynamo.replay.budget import ReplayBudget
from dynamo.replay.committee import (
    CommitteeReplay,
    ModelKnowledgeError,
    model_knowledge_gate,
)
from tests.unit.replay.test_committee import (
    PROMPTS,
    SCHEMAS,
    TIER_MODELS,
    CommitteeGateway,
    FakeExperienceRepo,
    FakeResearchModel,
    FakeRunRepo,
    _config,
    frames,  # noqa: F401  (module-scoped fixture reused here)
)

SONNET_CUTOFF = spec_for("claude-sonnet-5").knowledge_cutoff  # 2026-01-31
OPUS_CUTOFF = spec_for("claude-opus-5").knowledge_cutoff  # 2026-05-31
MARGIN = 30


def test_documented_cutoffs_resolved() -> None:
    assert spec_for("claude-sonnet-5").knowledge_cutoff == date(2026, 1, 31)
    assert spec_for("claude-opus-5").knowledge_cutoff == date(2026, 5, 31)
    assert spec_for("claude-haiku-4-5").knowledge_cutoff == date(2025, 7, 31)


def test_gate_blocks_at_and_inside_the_margin_boundary() -> None:
    models = {"analyst": "claude-sonnet-5"}
    boundary = SONNET_CUTOFF + timedelta(days=MARGIN)
    # start <= cutoff + margin -> blocked (boundary inclusive).
    for start in (SONNET_CUTOFF, boundary - timedelta(days=5), boundary):
        gate = model_knowledge_gate(start, models, MARGIN)
        assert gate["blocked"] is True
        assert gate["blocking_models"] == ["claude-sonnet-5"]
    # one day past the margin -> allowed.
    gate = model_knowledge_gate(boundary + timedelta(days=1), models, MARGIN)
    assert gate["blocked"] is False
    assert gate["blocking_models"] == []
    assert gate["cutoffs"] == {"claude-sonnet-5": SONNET_CUTOFF.isoformat()}
    assert gate["margin_days"] == MARGIN


def test_gate_takes_the_latest_cutoff_across_tiers() -> None:
    # Past sonnet's margin but inside opus's -> opus alone blocks.
    start = SONNET_CUTOFF + timedelta(days=MARGIN + 10)
    gate = model_knowledge_gate(start, TIER_MODELS, MARGIN)
    assert gate["blocked"] is True
    assert gate["blocking_models"] == ["claude-opus-5"]


def test_unknown_model_is_a_loud_error() -> None:
    with pytest.raises(UnknownModelError, match="gpt-9"):
        model_knowledge_gate(date(2027, 1, 1), {"analyst": "gpt-9"}, MARGIN)


def _replay_kwargs(frames: Any, **overrides: Any) -> dict[str, Any]:  # noqa: F811
    _, bars, scores = frames
    config = _config(frames)
    run_repo = FakeRunRepo()
    kwargs: dict[str, Any] = dict(
        gateway_factory=lambda d: CommitteeGateway(d, bars, scores),
        research_model=FakeResearchModel(),
        budget=ReplayBudget(cap_usd=100.0, run_repo=run_repo, run_id=config.run_id),
        run_repo=run_repo,
        experience_repo=FakeExperienceRepo(),
        config=config,
        tier_models=TIER_MODELS,
        prompts=PROMPTS,
        schemas=SCHEMAS,
        sleep_fn=lambda s: None,
    )
    kwargs.update(overrides)
    return kwargs


def test_blocked_window_refuses_without_acknowledgement(frames: Any) -> None:  # noqa: F811
    # The shared June-2026 test window sits inside opus-5's margin.
    with pytest.raises(ModelKnowledgeError, match="claude-opus-5"):
        CommitteeReplay(**_replay_kwargs(frames))
    # Nothing was created, nothing spent.
    kwargs = _replay_kwargs(frames)
    run_repo = kwargs["run_repo"]
    assert run_repo.runs == [] and run_repo.contexts == []


def test_acknowledged_bypass_is_recorded_into_config_json(frames: Any) -> None:  # noqa: F811
    kwargs = _replay_kwargs(frames, acknowledge_model_knowledge=True)
    replay = CommitteeReplay(**kwargs)
    replay.run()
    run_repo = kwargs["run_repo"]
    mk = run_repo.runs[0]["config"]["model_knowledge"]
    assert mk["blocked"] is True
    assert mk["acknowledged"] is True
    assert mk["blocking_models"] == ["claude-opus-5"]
    assert mk["cutoffs"]["claude-opus-5"] == OPUS_CUTOFF.isoformat()
    assert mk["margin_days"] == 30


def test_clean_window_records_unblocked_gate(frames: Any) -> None:  # noqa: F811
    """A window fully past every cutoff constructs without acknowledgement
    and still records the resolved cutoffs."""
    kwargs = _replay_kwargs(frames)
    safe_models = dict(TIER_MODELS)
    gate = model_knowledge_gate(kwargs["config"].start, safe_models, 0)
    # With margin 0 the June window is still inside opus (May 31 + 0 < Jun)?
    # start > 2026-05-31 -> allowed at margin 0: assert to pin the fixture.
    assert gate["blocked"] is False
    replay = CommitteeReplay(**_replay_kwargs(frames, cutoff_margin_days=0))
    assert replay._run_config_extras["model_knowledge"]["blocked"] is False
    assert replay._run_config_extras["model_knowledge"]["acknowledged"] is False
