"""JSON Schemas for structured Claude outputs. Design: doc 03 section 2.

Schemas are derived from pydantic "generation" models — the subset of each
domain entity that the model actually writes (code fills prompt_version, ids,
disagreement scores, etc.) — then post-processed once via `strict_schema`:
additionalProperties:false everywhere, every property required, $defs inlined.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamo.domain.models.market import TriageVerdict
from dynamo.domain.models.research import (
    Action,
    Dissent,
    Evidence,
    ExitCondition,
    FundamentalsRead,
    KillCase,
    Rebuttal,
    Side,
    TechnicalRead,
)

# --- Post-processor (write once, reuse) ---


def _resolve_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            resolved = _resolve_refs(deepcopy(defs[name]), defs)
            extras = {k: v for k, v in node.items() if k != "$ref"}
            resolved.update(_resolve_refs(extras, defs))
            return resolved
        return {k: _resolve_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, defs) for v in node]
    return node


def _strictify(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _strictify(v) for k, v in node.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node


def strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """model_json_schema() -> inline $defs -> strict objects, all keys required."""
    schema = model_cls.model_json_schema()
    defs = schema.pop("$defs", {})
    schema = _resolve_refs(schema, defs)
    return _strictify(schema)


# --- Shared field descriptions ---

# `conviction` shipped as a bare bounded float with no anchors. Measured over
# 295 verdicts of run replay-e5a49317bbee: 8 distinct values, 98.3% inside
# {0.55, 0.58, 0.60, 0.62}, |r| < 0.15 against every available feature — i.e.
# anchoring noise, not calibration. These anchors are the fix.
_CONVICTION_SCALE = (
    "Calibrated probability, not a rhetorical device. Anchors: 0.50 = coin flip, no edge; "
    "0.60 = modest edge; 0.75 = strong edge with multi-source confirmation; "
    "0.90 = near-certain. Use the full range — if every verdict you produce lands in "
    "0.55-0.62 you are anchoring, not calibrating."
)

_ARBITER_CONVICTION_DESC = (
    "Your probability that this verdict is the correct call. " + _CONVICTION_SCALE
)
_ADVOCATE_CONVICTION_DESC = (
    "Your probability that your side's thesis is materially correct. " + _CONVICTION_SCALE
)


# --- Generation models (what the model writes; code fills the rest) ---


class _Gen(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DossierAnalysisOut(_Gen):
    """Call 1 — stance-free fact base (symbol/as_of echoed for anchoring)."""

    symbol: str
    as_of: date
    business_summary: str
    fundamentals: FundamentalsRead
    filing_deltas: tuple[Evidence, ...]
    technical_state: TechnicalRead
    news_catalysts: tuple[Evidence, ...]
    data_gaps: tuple[str, ...]


class AdvocateBriefOut(_Gen):
    """Calls 2 & 3 — independent adversarial advocates."""

    symbol: str
    side: Side
    thesis: str
    key_points: tuple[Evidence, ...]
    kill_cases: tuple[KillCase, ...]
    expected_move_pct: float
    horizon_days: int
    conviction: float = Field(ge=0.0, le=1.0, description=_ADVOCATE_CONVICTION_DESC)


class ArbiterVerdictOut(_Gen):
    """Call 4 — the fields the arbiter generates; guardrails/ids added in code."""

    action: Action
    conviction: float = Field(ge=0.0, le=1.0, description=_ARBITER_CONVICTION_DESC)
    thesis: str
    bull_case: str
    bear_case: str
    bear_rebuttals: tuple[Rebuttal, ...]
    risks: tuple[str, ...]
    exit_conditions: tuple[ExitCondition, ...]
    expected_upside_pct: float
    expected_downside_pct: float
    horizon_days: int
    position_size_hint_pct: float
    dissents: tuple[Dissent, ...]
    self_critique: str


class MacroDigestOut(_Gen):
    narrative: str
    rates_outlook: str
    inflation_outlook: str
    regime_comment: str
    prediction_market_context: tuple[str, ...]
    evidence: tuple[Evidence, ...]


class PredictionMarketViewOut(_Gen):
    venue: str
    market_id: str
    question: str
    market_implied_prob: float = Field(ge=0.0, le=1.0)
    estimated_prob: float = Field(ge=0.0, le=1.0)
    edge: float
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...]
    historical_comparisons: tuple[str, ...]
    why_wrong: tuple[str, ...] = Field(min_length=1)


class TriageOut(_Gen):
    materiality: float = Field(ge=0.0, le=1.0)
    verdict: TriageVerdict
    reason: str


DOSSIER_ANALYSIS_SCHEMA = strict_schema(DossierAnalysisOut)
ADVOCATE_BRIEF_SCHEMA = strict_schema(AdvocateBriefOut)
ARBITER_VERDICT_SCHEMA = strict_schema(ArbiterVerdictOut)
MACRO_DIGEST_SCHEMA = strict_schema(MacroDigestOut)
PREDICTION_MARKET_VIEW_SCHEMA = strict_schema(PredictionMarketViewOut)
TRIAGE_SCHEMA = strict_schema(TriageOut)

SCHEMAS: dict[str, dict[str, Any]] = {
    "dossier": DOSSIER_ANALYSIS_SCHEMA,
    "advocate": ADVOCATE_BRIEF_SCHEMA,
    "arbiter": ARBITER_VERDICT_SCHEMA,
    "macro_digest": MACRO_DIGEST_SCHEMA,
    "prediction_market": PREDICTION_MARKET_VIEW_SCHEMA,
    "triage": TRIAGE_SCHEMA,
}
