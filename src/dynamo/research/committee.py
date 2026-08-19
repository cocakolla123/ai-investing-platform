"""Committee engine — CORE. Design: doc 03 sections 1-4.

Receives a ResearchModel port instance plus loaded prompts and schema dicts via
constructor injection; never imports adapters. Functions return domain objects;
persistence is the caller's job. Guardrails live here in code, not in prompts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from hashlib import sha256
from typing import Any

import structlog

from dynamo.domain.models.research import (
    ACTION_SCORE,
    Action,
    AdvocateBrief,
    CommitteeVerdict,
    Dissent,
    DossierAnalysis,
    ExitCondition,
    KillCase,
    QuantAlignment,
    QuantScore,
    Rebuttal,
    Severity,
    Side,
    Stance,
)
from dynamo.domain.ports import CONTEXT_CACHE_BOUNDARY, ResearchModel
from dynamo.research.prompt_loader import LoadedPrompt

DISAGREEMENT_VETO_THRESHOLD = 0.6
GUARDRAIL_AGENT = "guardrail"



log = structlog.get_logger(__name__)


def _validated(model_cls: type, payload: Any) -> Any:
    """Validate an LLM-output payload, tolerating (and logging) unknown keys.

    Domain models are ``extra="forbid"`` — right for our own code, wrong at the
    model-output boundary: backends without server-side schema enforcement
    (the Claude Code CLI path) may add harmless extra keys, which must not
    void a whole verdict. Required fields and types stay strictly enforced.
    """
    if isinstance(payload, Mapping):
        known = set(model_cls.model_fields)
        extra = [k for k in payload if k not in known]
        if extra:
            log.warning(
                "committee_payload_extra_keys_dropped",
                model=model_cls.__name__,
                keys=sorted(str(k) for k in extra)[:8],
            )
            payload = {k: v for k, v in payload.items() if k in known}
    return model_cls.model_validate(payload)


# stage -> (prompt name, schema key, model tier)
STAGES: dict[str, tuple[str, str, str]] = {
    "dossier": ("dossier_analyst", "dossier", "analyst"),
    "bull": ("bull_advocate", "advocate", "analyst"),
    "bear": ("bear_advocate", "advocate", "analyst"),
    "arbiter": ("arbiter", "arbiter", "arbiter"),
}

# Temporal prompt discipline (backtest-campaign Phase 4): appended to EVERY
# stage instruction. The instruction block is the volatile per-call segment
# (never cache-controlled), so this is cache-safe. On the live path
# as_of == today, so the sentence is semantically a no-op there — accepted
# and deliberate: one instruction string, no live/replay fork.
TEMPORAL_DISCIPLINE = (
    " You are analyzing as of {as_of}. Use ONLY the provided context. "
    "Disregard any knowledge of events after {as_of}."
)

_STAGE_INSTRUCTIONS: dict[str, str] = {
    "dossier": "Produce the stance-free dossier analysis JSON for {symbol} as of {as_of}."
    + TEMPORAL_DISCIPLINE,
    "bull": "Produce the bull advocate brief JSON for {symbol} as of {as_of}."
    + TEMPORAL_DISCIPLINE,
    "bear": "Produce the bear advocate brief JSON for {symbol} as of {as_of}."
    + TEMPORAL_DISCIPLINE,
    "arbiter": "Produce the committee verdict JSON for {symbol} as of {as_of}."
    + TEMPORAL_DISCIPLINE,
}

_DEFAULT_MAX_TOKENS: dict[str, int] = {
    "dossier": 4096,
    "bull": 3072,
    "bear": 3072,
    "arbiter": 4096,
}


def _filtered_payload(
    model_cls: type, parsed: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    fields: Mapping[str, Any] = model_cls.model_fields  # type: ignore[attr-defined]
    payload = {k: v for k, v in parsed.items() if k in fields}
    payload.update(overrides)
    return payload


def custom_id_for(as_of: date, symbol: str, stage: str) -> str:
    return f"{as_of.isoformat()}-{symbol}-{stage}"


def split_custom_id(custom_id: str) -> tuple[date, str, str]:
    """Inverse of custom_id_for; robust to hyphens in the symbol."""
    as_of = date.fromisoformat(custom_id[:10])
    rest, stage = custom_id[11:].rsplit("-", 1)
    return as_of, rest, stage


def compute_disagreement(
    *,
    bull_conviction: float,
    bear_conviction: float,
    action: Action,
    arbiter_conviction: float,
) -> float:
    """Normalized conviction-weighted dispersion of stances, in [0, 1] (doc 03 §4)."""
    values = [bull_conviction, -bear_conviction, ACTION_SCORE[action] / 2.0]
    weights = [bull_conviction, bear_conviction, arbiter_conviction]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    mean = sum(w * v for w, v in zip(weights, values, strict=True)) / total
    variance = sum(w * (v - mean) ** 2 for w, v in zip(weights, values, strict=True)) / total
    return max(0.0, min(1.0, math.sqrt(variance)))


def quant_alignment_for(action: Action, quant_score: QuantScore | None) -> QuantAlignment:
    """sign(expected_excess_return) vs action direction; None score -> neutral."""
    if action in (Action.BUY, Action.ADD):
        direction = 1
    elif action in (Action.SELL, Action.TRIM, Action.AVOID):
        direction = -1
    else:
        direction = 0
    if direction == 0 or quant_score is None or quant_score.expected_excess_return is None:
        return QuantAlignment.NEUTRAL
    eer = quant_score.expected_excess_return
    if eer == 0.0:
        return QuantAlignment.NEUTRAL
    return QuantAlignment.AGREE if (eer > 0) == (direction > 0) else QuantAlignment.CONFLICT


def _unrebutted_high_kill_cases(
    kill_cases: Iterable[KillCase], rebuttals: Iterable[Rebuttal]
) -> list[KillCase]:
    rebuttal_keys = [r.kill_case.casefold() for r in rebuttals]
    unmatched: list[KillCase] = []
    for kc in kill_cases:
        if kc.severity is not Severity.HIGH:
            continue
        desc = kc.description.casefold()
        if not any(desc in key or key in desc for key in rebuttal_keys):
            unmatched.append(kc)
    return unmatched


def _verdict_id(symbol: str, as_of: date, arbiter_payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"symbol": symbol, "as_of": as_of.isoformat(), "payload": arbiter_payload},
        sort_keys=True,
        default=str,
    )
    return f"{symbol}-{as_of.isoformat()}-{sha256(canonical.encode()).hexdigest()[:8]}"


class CommitteeEngine:
    """4-call adversarial deep-dive: dossier -> bull & bear -> arbiter."""

    def __init__(
        self,
        research_model: ResearchModel,
        prompts: Mapping[str, LoadedPrompt],
        schemas: Mapping[str, dict[str, Any]],
        *,
        max_tokens: Mapping[str, int] | None = None,
    ) -> None:
        missing = [name for name, _, _ in STAGES.values() if name not in prompts]
        if missing:
            raise ValueError(f"missing prompts: {missing}")
        missing_schemas = [key for _, key, _ in STAGES.values() if key not in schemas]
        if missing_schemas:
            raise ValueError(f"missing schemas: {missing_schemas}")
        self._model = research_model
        self._prompts = prompts
        self._schemas = schemas
        self._max_tokens = {**_DEFAULT_MAX_TOKENS, **(max_tokens or {})}

    # --- single deep dive (sequential; batch path uses build_batch_requests) ---

    def _call_stage(self, stage: str, symbol: str, as_of: date, context: str) -> dict[str, Any]:
        prompt_name, schema_key, tier = STAGES[stage]
        prompt = self._prompts[prompt_name]
        return self._model.complete_json(
            role=stage,
            system=prompt.system,
            context=context,
            instruction=_STAGE_INSTRUCTIONS[stage].format(symbol=symbol, as_of=as_of),
            schema=self._schemas[schema_key],
            model_tier=tier,
            max_tokens=self._max_tokens[stage],
        )

    def run_deep_dive(
        self,
        symbol: str,
        as_of: date,
        context: str,
        quant_score: QuantScore | None = None,
        journal_context_used: Sequence[str] = (),
    ) -> tuple[DossierAnalysis, AdvocateBrief, AdvocateBrief, CommitteeVerdict]:
        dossier_raw = self._call_stage("dossier", symbol, as_of, context)
        dossier = _validated(DossierAnalysis,
            _filtered_payload(
                DossierAnalysis,
                dossier_raw,
                {
                    "symbol": symbol,
                    "as_of": as_of,
                    "prompt_version": self._prompts["dossier_analyst"].version,
                },
            )
        )

        advocate_context = advocate_context_for(context, dossier_raw)
        bull_raw = self._call_stage("bull", symbol, as_of, advocate_context)
        bull = _validated(AdvocateBrief,
            _filtered_payload(AdvocateBrief, bull_raw, {"symbol": symbol, "side": Side.BULL})
        )
        bear_raw = self._call_stage("bear", symbol, as_of, advocate_context)
        bear = _validated(AdvocateBrief,
            _filtered_payload(AdvocateBrief, bear_raw, {"symbol": symbol, "side": Side.BEAR})
        )

        arb_context = arbiter_context_for(context, dossier_raw, bull_raw, bear_raw, quant_score)
        arbiter_raw = self._call_stage("arbiter", symbol, as_of, arb_context)
        verdict = self.assemble_verdict(
            symbol=symbol,
            as_of=as_of,
            arbiter_payload=arbiter_raw,
            bull=bull,
            bear=bear,
            quant_score=quant_score,
            prompt_version=self._prompts["arbiter"].version,
            journal_context_used=journal_context_used,
        )
        return dossier, bull, bear, verdict

    # --- pure verdict assembly with guardrails in code (doc 03 §4) ---

    @staticmethod
    def assemble_verdict(
        *,
        symbol: str,
        as_of: date,
        arbiter_payload: Mapping[str, Any],
        bull: AdvocateBrief,
        bear: AdvocateBrief,
        quant_score: QuantScore | None = None,
        prompt_version: str = "",
        model_id: str = "",
        journal_context_used: Sequence[str] = (),
    ) -> CommitteeVerdict:
        action = Action(str(arbiter_payload["action"]))
        conviction = max(0.0, min(1.0, float(arbiter_payload["conviction"])))
        rebuttals = tuple(
            _validated(Rebuttal, r) for r in arbiter_payload.get("bear_rebuttals", [])
        )
        exit_conditions = tuple(
            _validated(ExitCondition, e) for e in arbiter_payload.get("exit_conditions", [])
        )
        dissents = tuple(_validated(Dissent, d) for d in arbiter_payload.get("dissents", []))
        self_critique = str(arbiter_payload.get("self_critique", ""))

        disagreement = compute_disagreement(
            bull_conviction=bull.conviction,
            bear_conviction=bear.conviction,
            action=action,
            arbiter_conviction=conviction,
        )

        # Guardrails: buy/add downgraded to hold on high disagreement or any
        # unrebutted high-severity bear kill case.
        if action in (Action.BUY, Action.ADD):
            reasons: list[str] = []
            if disagreement > DISAGREEMENT_VETO_THRESHOLD:
                reasons.append(
                    f"disagreement_score {disagreement:.2f} > "
                    f"{DISAGREEMENT_VETO_THRESHOLD} veto threshold"
                )
            unrebutted = _unrebutted_high_kill_cases(bear.kill_cases, rebuttals)
            reasons.extend(
                f"unrebutted high-severity bear kill case: {kc.description}"
                for kc in unrebutted
            )
            if reasons:
                reason_text = "; ".join(reasons)
                dissents = (
                    *dissents,
                    Dissent(
                        agent=GUARDRAIL_AGENT,
                        stance=Stance.HOLD,
                        reason=f"downgraded {action.value} to hold: {reason_text}",
                    ),
                )
                suffix = f"[guardrail] downgraded {action.value} to hold: {reason_text}"
                self_critique = f"{self_critique}\n{suffix}".strip()
                action = Action.HOLD

        return CommitteeVerdict(
            id=_verdict_id(symbol, as_of, arbiter_payload),
            symbol=symbol,
            as_of=as_of,
            action=action,
            conviction=conviction,
            thesis=str(arbiter_payload.get("thesis", "")),
            bull_case=str(arbiter_payload.get("bull_case", "")),
            bear_case=str(arbiter_payload.get("bear_case", "")),
            bear_rebuttals=rebuttals,
            risks=tuple(str(r) for r in arbiter_payload.get("risks", [])),
            exit_conditions=exit_conditions,
            expected_upside_pct=float(arbiter_payload.get("expected_upside_pct", 0.0)),
            expected_downside_pct=float(arbiter_payload.get("expected_downside_pct", 0.0)),
            horizon_days=int(arbiter_payload.get("horizon_days", 20)),
            position_size_hint_pct=float(arbiter_payload.get("position_size_hint_pct", 0.0)),
            disagreement_score=disagreement,
            dissents=dissents,
            self_critique=self_critique,
            quant_alignment=quant_alignment_for(action, quant_score),
            journal_context_used=tuple(journal_context_used),
            prompt_version=prompt_version,
            model_id=model_id,
        )

    # --- batch helpers (deterministic custom_ids; doc 01 §6 idempotency) ---

    def build_batch_requests(
        self, stage: str, items: Mapping[str, str], as_of: date
    ) -> list[dict[str, Any]]:
        """items: symbol -> context. Deterministic order and custom_ids."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGES)}")
        prompt_name, schema_key, tier = STAGES[stage]
        prompt = self._prompts[prompt_name]
        return [
            {
                "custom_id": custom_id_for(as_of, symbol, stage),
                "model_tier": tier,
                "system": prompt.system,
                "context": items[symbol],
                "instruction": _STAGE_INSTRUCTIONS[stage].format(symbol=symbol, as_of=as_of),
                "schema": self._schemas[schema_key],
                "max_tokens": self._max_tokens[stage],
            }
            for symbol in sorted(items)
        ]

    @staticmethod
    def parse_stage_results(
        results: Iterable[tuple[str, dict[str, Any] | None]], stage: str
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """-> (symbol -> payload, failed custom_ids needing resubmission)."""
        parsed: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        for custom_id, payload in results:
            _, symbol, result_stage = split_custom_id(custom_id)
            if result_stage != stage:
                continue
            if payload is None:
                failed.append(custom_id)
            else:
                parsed[symbol] = payload
        return parsed, failed


def advocate_context_for(context: str, dossier_payload: Mapping[str, Any]) -> str:
    # CONTEXT_CACHE_BOUNDARY keeps the shared base context a byte-identical
    # prefix segment, so bull/bear/arbiter re-read it from prompt cache.
    return (
        f"{context}{CONTEXT_CACHE_BOUNDARY}## Dossier analysis (JSON)\n"
        f"{json.dumps(dict(dossier_payload), sort_keys=True, default=str)}"
    )


def arbiter_context_for(
    context: str,
    dossier_payload: Mapping[str, Any],
    bull_payload: Mapping[str, Any],
    bear_payload: Mapping[str, Any],
    quant_score: QuantScore | None,
) -> str:
    tail = [
        "## Bull advocate brief (JSON)",
        json.dumps(dict(bull_payload), sort_keys=True, default=str),
        "## Bear advocate brief (JSON)",
        json.dumps(dict(bear_payload), sort_keys=True, default=str),
        "## Quant score (JSON)",
        json.dumps(quant_score.model_dump(mode="json"), sort_keys=True)
        if quant_score is not None
        else "null",
    ]
    prefix = advocate_context_for(context, dossier_payload)
    return prefix + CONTEXT_CACHE_BOUNDARY + "\n\n".join(tail)
