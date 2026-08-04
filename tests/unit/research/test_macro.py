"""Macro digest + prediction-market view builders (code-enforced invariants)."""

from __future__ import annotations

from datetime import date

from fakes import (
    MACRO_PAYLOAD,
    PM_INVALID_PAYLOAD,
    PM_VALID_PAYLOAD,
    FakeResearchModel,
)

from dynamo.domain.models.market import PredictionMarket
from dynamo.domain.models.research import MacroDigest
from dynamo.research.macro import build_macro_digest, build_prediction_market_views
from dynamo.research.prompt_loader import load_prompt

AS_OF = date(2026, 8, 3)
SCHEMA = {"type": "object"}

MARKET = PredictionMarket(
    venue="kalshi", market_id="FED-26DEC", question="Will the Fed cut rates by December 2026?"
)


def test_build_macro_digest_returns_domain_object() -> None:
    model = FakeResearchModel({"macro_digest": [MACRO_PAYLOAD]})
    digest = build_macro_digest(
        model,
        "CPI 2.8%; unemployment 4.1%",
        "Fed cut implied 62%",
        AS_OF,
        prompt=load_prompt("macro_digest"),
        schema=SCHEMA,
    )
    assert isinstance(digest, MacroDigest)
    assert digest.as_of == AS_OF
    assert digest.prompt_version == "1.0.0"
    assert digest.narrative == MACRO_PAYLOAD["narrative"]
    call = model.calls[0]
    assert call["model_tier"] == "analyst"
    assert "Prediction-market digest" in call["context"]


def test_pm_view_retry_then_success_is_kept() -> None:
    model = FakeResearchModel({"prediction_market": [PM_INVALID_PAYLOAD, PM_VALID_PAYLOAD]})
    views = build_prediction_market_views(
        model,
        [(MARKET, 0.62)],
        AS_OF,
        prompt=load_prompt("prediction_market"),
        schema=SCHEMA,
    )
    assert len(model.calls) == 2  # one retry with corrective instruction
    assert "rejected" in model.calls[1]["instruction"]
    assert len(views) == 1
    view = views[0]
    assert view.uncertainties  # non-empty enforced
    assert view.why_wrong
    assert view.market_implied_prob == 0.62
    assert abs(view.edge - (view.estimated_prob - 0.62)) < 1e-12
    assert view.as_of == AS_OF


def test_pm_view_dropped_after_second_invalid_output() -> None:
    model = FakeResearchModel({"prediction_market": [PM_INVALID_PAYLOAD, PM_INVALID_PAYLOAD]})
    views = build_prediction_market_views(
        model,
        [(MARKET, 0.62)],
        AS_OF,
        prompt=load_prompt("prediction_market"),
        schema=SCHEMA,
    )
    assert views == []
    assert len(model.calls) == 2
