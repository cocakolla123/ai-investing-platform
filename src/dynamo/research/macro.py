"""Macro digest + prediction-market analysis — CORE. Design: doc 03 sections 2, 6.

Research-only: PredictionMarketView.edge is a research artifact; no execution
hooks exist anywhere. Non-empty uncertainties/why_wrong are enforced in code
(retry once with a corrective instruction, else drop with a log line).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from dynamo.domain.models.market import PredictionMarket
from dynamo.domain.models.research import MacroDigest, PredictionMarketView
from dynamo.domain.ports import ResearchModel
from dynamo.research.prompt_loader import LoadedPrompt

log = logging.getLogger(__name__)

_MACRO_INSTRUCTION = "Produce the daily macro digest JSON for {as_of}."
_PM_INSTRUCTION = (
    "Estimate the true probability for this market and produce the "
    "prediction-market view JSON for {as_of}."
)
_PM_RETRY_SUFFIX = (
    " Your previous output was rejected: uncertainties and why_wrong MUST each "
    "contain at least one concrete, non-empty entry."
)


def _filtered(model_cls: type, parsed: Mapping[str, Any], overrides: Mapping[str, Any]) -> Any:
    fields: Mapping[str, Any] = model_cls.model_fields  # type: ignore[attr-defined]
    payload = {k: v for k, v in parsed.items() if k in fields}
    payload.update(overrides)
    return model_cls.model_validate(payload)  # type: ignore[attr-defined]


def build_macro_digest(
    research_model: ResearchModel,
    macro_context: str,
    pm_context: str,
    as_of: date,
    *,
    prompt: LoadedPrompt,
    schema: dict[str, Any],
    max_tokens: int = 3072,
) -> MacroDigest:
    """Once daily, whole market: rates/inflation/regime + prediction-market context."""
    context = (
        f"## Macro series and market context\n{macro_context.strip()}\n\n"
        f"## Prediction-market digest\n{pm_context.strip() or 'none'}"
    )
    parsed = research_model.complete_json(
        role="macro_digest",
        system=prompt.system,
        context=context,
        instruction=_MACRO_INSTRUCTION.format(as_of=as_of),
        schema=schema,
        model_tier="analyst",
        max_tokens=max_tokens,
    )
    digest: MacroDigest = _filtered(
        MacroDigest, parsed, {"as_of": as_of, "prompt_version": prompt.version}
    )
    return digest


def _pm_market_context(market: PredictionMarket, implied_prob: float) -> str:
    return "\n".join(
        [
            "## Tracked prediction market",
            f"venue: {market.venue}",
            f"market_id: {market.market_id}",
            f"question: {market.question}",
            f"category: {market.category or 'n/a'}",
            f"close_ts: {market.close_ts.isoformat() if market.close_ts else 'n/a'}",
            f"market_implied_prob: {implied_prob:.4f}",
        ]
    )


def build_prediction_market_views(
    research_model: ResearchModel,
    markets: Sequence[tuple[PredictionMarket, float]],
    as_of: date,
    *,
    prompt: LoadedPrompt,
    schema: dict[str, Any],
    max_tokens: int = 2048,
) -> list[PredictionMarketView]:
    """One view per (market, implied_prob). Empty uncertainties/why_wrong are
    rejected in code: one retry, then the market is dropped with a log line."""
    views: list[PredictionMarketView] = []
    for market, implied_prob in markets:
        context = _pm_market_context(market, implied_prob)
        parsed: Mapping[str, Any] | None = None
        for attempt in range(2):
            instruction = _PM_INSTRUCTION.format(as_of=as_of)
            if attempt == 1:
                instruction += _PM_RETRY_SUFFIX
            candidate = research_model.complete_json(
                role="prediction_market",
                system=prompt.system,
                context=context,
                instruction=instruction,
                schema=schema,
                model_tier="analyst",
                max_tokens=max_tokens,
            )
            if candidate.get("uncertainties") and candidate.get("why_wrong"):
                parsed = candidate
                break
        if parsed is None:
            log.warning(
                "dropping prediction-market view %s/%s: empty uncertainties/why_wrong after retry",
                market.venue,
                market.market_id,
            )
            continue
        estimated = max(0.0, min(1.0, float(parsed["estimated_prob"])))
        views.append(
            _filtered(
                PredictionMarketView,
                parsed,
                {
                    "venue": market.venue,
                    "market_id": market.market_id,
                    "question": market.question,
                    "as_of": as_of,
                    "market_implied_prob": implied_prob,
                    "estimated_prob": estimated,
                    "edge": estimated - implied_prob,
                },
            )
        )
    return views
