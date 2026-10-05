"""Test doubles for the research layer. No real API calls anywhere."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class FakeResearchModel:
    """ResearchModel double returning canned JSON per role.

    responses: role -> list of payloads consumed in order (the last one repeats).
    """

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[dict[str, Any]] = []
        self.batches: dict[str, list[dict[str, Any]]] = {}

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        model_tier: str,
        max_tokens: int = 4096,
        trace: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "trace": dict(trace or {}),
                "role": role,
                "system": system,
                "context": context,
                "instruction": instruction,
                "schema": schema,
                "model_tier": model_tier,
                "max_tokens": max_tokens,
            }
        )
        queue = self._responses[role]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def submit_batch(self, requests: Sequence[dict[str, Any]]) -> str:
        batch_id = f"batch-{len(self.batches) + 1}"
        self.batches[batch_id] = list(requests)
        return batch_id

    def batch_status(self, batch_id: str) -> str:
        return "ended"

    def batch_results(self, batch_id: str) -> Iterable[tuple[str, dict[str, Any] | None]]:
        return iter(())


EVIDENCE = {
    "claim": "Revenue grew 21% year over year",
    "source_type": "filing",
    "source_ref": "0001234567-26-000042",
    "as_of": "2026-07-31",
}

DOSSIER_PAYLOAD: dict[str, Any] = {
    "symbol": "ACME",
    "as_of": "2026-08-03",
    "business_summary": "Industrial anvil maker with a growing software segment.",
    "fundamentals": {
        "growth": "Revenue +21% YoY",
        "margins": "Gross margin stable at 44%",
        "cash_flow": "FCF positive 8 quarters",
        "balance_sheet": "Net cash $1.2B",
        "evidence": [EVIDENCE],
    },
    "filing_deltas": [EVIDENCE],
    "technical_state": {
        "trend": "uptrend above 50d",
        "relative_strength": "outperforming sector by 6%",
        "volatility_state": "normal",
        "key_levels": "support 92, resistance 110",
        "evidence": [],
    },
    "news_catalysts": [],
    "data_gaps": ["no guidance detail for FY27"],
}

BULL_PAYLOAD: dict[str, Any] = {
    "symbol": "ACME",
    "side": "bull",
    "thesis": "Software mix shift drives margin expansion the street has not modeled.",
    "key_points": [EVIDENCE],
    "kill_cases": [
        {
            "description": "Bear case breaks if software attach rate keeps compounding",
            "severity": "medium",
            "trigger": "attach rate > 40% next quarter",
        }
    ],
    "expected_move_pct": 18.0,
    "horizon_days": 60,
    "conviction": 0.8,
}

BEAR_HIGH_KILL_CASE = {
    "description": "Customer concentration collapse",
    "severity": "high",
    "trigger": "top customer (31% of revenue) churns at renewal",
}

BEAR_PAYLOAD: dict[str, Any] = {
    "symbol": "ACME",
    "side": "bear",
    "thesis": "Concentration risk and decelerating hardware mask the software story.",
    "key_points": [EVIDENCE],
    "kill_cases": [BEAR_HIGH_KILL_CASE],
    "expected_move_pct": -12.0,
    "horizon_days": 60,
    "conviction": 0.2,
}

ARBITER_PAYLOAD: dict[str, Any] = {
    "action": "buy",
    "conviction": 0.65,
    "thesis": "Bull case survives the bear's concentration attack; enter with tight risk.",
    "bull_case": "Margin expansion from software mix.",
    "bear_case": "Customer concentration and hardware deceleration.",
    "bear_rebuttals": [
        {
            "kill_case": "Customer concentration collapse",
            "rebuttal": "Top-customer contract renewed through 2028 per latest 10-Q.",
            "evidence": [EVIDENCE],
        }
    ],
    "risks": ["macro slowdown hits industrial capex"],
    "exit_conditions": [
        {"description": "daily close below 90", "checkable": True, "predicate": "close < 90"}
    ],
    "expected_upside_pct": 20.0,
    "expected_downside_pct": 8.0,
    "horizon_days": 60,
    "position_size_hint_pct": 3.0,
    "dissents": [],
    "self_critique": "Weakest link: renewal terms not fully disclosed.",
}

MACRO_PAYLOAD: dict[str, Any] = {
    "narrative": "Disinflation intact; growth cooling but not contracting.",
    "rates_outlook": "One further cut priced by December.",
    "inflation_outlook": "Core CPI trending toward 2.5%.",
    "regime_comment": "Regime stable: late-cycle expansion.",
    "prediction_market_context": ["Fed cut by Dec implied at 62% (kalshi FED-26DEC)"],
    "evidence": [
        {
            "claim": "Core CPI 2.8% YoY in June",
            "source_type": "macro",
            "source_ref": "CPILFESL",
            "as_of": "2026-07-15",
        }
    ],
}

PM_VALID_PAYLOAD: dict[str, Any] = {
    "venue": "kalshi",
    "market_id": "FED-26DEC",
    "question": "Will the Fed cut rates by December 2026?",
    "market_implied_prob": 0.62,
    "estimated_prob": 0.7,
    "edge": 0.08,
    "confidence": 0.55,
    "uncertainties": ["September CPI print", "labor market break risk"],
    "evidence": [
        {
            "claim": "Two governors signaled openness to a cut",
            "source_type": "news",
            "source_ref": "news-8842",
            "as_of": "2026-08-01",
        }
    ],
    "historical_comparisons": ["2024 easing cycle pricing ran 10pts ahead of the Fed"],
    "why_wrong": ["Sticky shelter inflation could stall the committee"],
}

PM_INVALID_PAYLOAD: dict[str, Any] = {**PM_VALID_PAYLOAD, "uncertainties": [], "why_wrong": []}
