"""Schema post-processing invariants + round-trips through the pydantic models."""

from __future__ import annotations

from typing import Any

from dynamo.domain.models.research import (
    AdvocateBrief,
    CommitteeVerdict,
    DossierAnalysis,
    MacroDigest,
    PredictionMarketView,
)
from dynamo.llm.schemas import (
    SCHEMAS,
    AdvocateBriefOut,
    ArbiterVerdictOut,
    DossierAnalysisOut,
    MacroDigestOut,
    PredictionMarketViewOut,
    TriageOut,
)

EVIDENCE = {
    "claim": "Revenue grew 21%",
    "source_type": "filing",
    "source_ref": "0001-26-000042",
    "as_of": "2026-07-31",
}


def walk(node: Any) -> Any:
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


class TestStrictness:
    def test_every_object_is_closed_and_fully_required(self) -> None:
        for name, schema in SCHEMAS.items():
            for node in walk(schema):
                if isinstance(node, dict) and node.get("type") == "object" and (
                    "properties" in node
                ):
                    assert node.get("additionalProperties") is False, name
                    assert node.get("required") == list(node["properties"].keys()), name

    def test_no_refs_or_defs_remain(self) -> None:
        for name, schema in SCHEMAS.items():
            assert "$defs" not in schema, name
            for node in walk(schema):
                if isinstance(node, dict):
                    assert "$ref" not in node, name

    def test_top_level_is_object(self) -> None:
        for schema in SCHEMAS.values():
            assert schema["type"] == "object"


class TestRoundTrip:
    def test_dossier_example_through_out_and_domain_models(self) -> None:
        example = {
            "symbol": "ACME",
            "as_of": "2026-08-03",
            "business_summary": "Anvils.",
            "fundamentals": {
                "growth": "g",
                "margins": "m",
                "cash_flow": "c",
                "balance_sheet": "b",
                "evidence": [EVIDENCE],
            },
            "filing_deltas": [EVIDENCE],
            "technical_state": {
                "trend": "up",
                "relative_strength": "strong",
                "volatility_state": "normal",
                "key_levels": "92/110",
                "evidence": [],
            },
            "news_catalysts": [],
            "data_gaps": [],
        }
        out = DossierAnalysisOut.model_validate(example)
        domain = DossierAnalysis.model_validate({**example, "prompt_version": "1.0.0"})
        assert out.symbol == domain.symbol == "ACME"

    def test_advocate_example(self) -> None:
        example = {
            "symbol": "ACME",
            "side": "bear",
            "thesis": "t",
            "key_points": [EVIDENCE],
            "kill_cases": [
                {"description": "churn", "severity": "high", "trigger": "renewal miss"}
            ],
            "expected_move_pct": -10.0,
            "horizon_days": 30,
            "conviction": 0.4,
        }
        assert AdvocateBriefOut.model_validate(example).conviction == 0.4
        assert AdvocateBrief.model_validate(example).side.value == "bear"

    def test_arbiter_example_builds_domain_verdict(self) -> None:
        example = {
            "action": "hold",
            "conviction": 0.5,
            "thesis": "t",
            "bull_case": "b",
            "bear_case": "r",
            "bear_rebuttals": [
                {"kill_case": "churn", "rebuttal": "renewed", "evidence": [EVIDENCE]}
            ],
            "risks": ["macro"],
            "exit_conditions": [
                {"description": "close < 90", "checkable": True, "predicate": "close<90"}
            ],
            "expected_upside_pct": 10.0,
            "expected_downside_pct": 5.0,
            "horizon_days": 20,
            "position_size_hint_pct": 2.0,
            "dissents": [{"agent": "bear", "stance": "sell", "reason": "churn risk"}],
            "self_critique": "sc",
        }
        out = ArbiterVerdictOut.model_validate(example)
        domain = CommitteeVerdict.model_validate(
            {**example, "id": "ACME-2026-08-03-deadbeef", "symbol": "ACME", "as_of": "2026-08-03"}
        )
        assert out.action == domain.action

    def test_macro_and_pm_and_triage_examples(self) -> None:
        macro = {
            "narrative": "n",
            "rates_outlook": "r",
            "inflation_outlook": "i",
            "regime_comment": "rc",
            "prediction_market_context": ["x"],
            "evidence": [],
        }
        assert MacroDigestOut.model_validate(macro).narrative == "n"
        assert MacroDigest.model_validate({**macro, "as_of": "2026-08-03"}).as_of.year == 2026

        pm = {
            "venue": "kalshi",
            "market_id": "FED-26DEC",
            "question": "q",
            "market_implied_prob": 0.6,
            "estimated_prob": 0.7,
            "edge": 0.1,
            "confidence": 0.5,
            "uncertainties": ["u"],
            "evidence": [],
            "historical_comparisons": [],
            "why_wrong": ["w"],
        }
        assert PredictionMarketViewOut.model_validate(pm).edge == 0.1
        assert (
            PredictionMarketView.model_validate({**pm, "as_of": "2026-08-03"}).venue == "kalshi"
        )

        triage = {"materiality": 0.8, "verdict": "material", "reason": "guidance cut"}
        assert TriageOut.model_validate(triage).verdict.value == "material"
