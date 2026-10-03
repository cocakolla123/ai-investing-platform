"""Committee engine — CORE. Design: doc 03 sections 1-4.

Receives a ResearchModel port instance plus loaded prompts and schema dicts via
constructor injection; never imports adapters. Functions return domain objects;
persistence is the caller's job. Guardrails live here in code, not in prompts.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
from typing import Any

import structlog

from dynamo.domain.models.research import (
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

# --- Rebuttal <-> kill-case matching (see _unrebutted_high_kill_cases) ---
# Tier-2 fallback thresholds, calibrated on the 516 high-severity kill cases of
# run replay-e5a49317bbee: 88.8% of same-verdict pairs match, 0 of 516
# cross-verdict (different-symbol) pairs match.
REBUTTAL_JACCARD_THRESHOLD = 0.4
REBUTTAL_RATIO_THRESHOLD = 0.6

# The arbiter prefixes restated objections with a severity label ("HIGH: ...").
_SEVERITY_PREFIX_RE = re.compile(r"^\s*(?:high|medium|low|critical|severe)\s*[:—-]\s*", re.I)
_WORD_RE = re.compile(r"[a-z0-9]+")
# Function words carry no topical signal; keeping them inflates Jaccard on
# unrelated pairs and deflates it on paraphrases of different length.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "without",
        "by",
        "as",
        "at",
        "from",
        "into",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "there",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "while",
        "than",
        "then",
        "so",
        "not",
        "no",
        "any",
        "all",
        "some",
        "more",
        "most",
        "other",
        "such",
        "only",
        "own",
        "same",
        "can",
        "could",
        "may",
        "might",
        "will",
        "would",
        "shall",
        "should",
        "must",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "over",
        "under",
        "above",
        "below",
        "between",
        "through",
        "during",
        "before",
        "after",
        "up",
        "down",
        "out",
        "off",
        "again",
        "further",
        "also",
        "both",
        "each",
        "few",
        "per",
        "via",
    }
)


log = structlog.get_logger(__name__)


def _normalized_keys(value: Any) -> Any:
    """Recursively strip whitespace from dict keys (LLMs emit ``"as_of "``).

    First-seen key wins on a post-strip collision. Lists/tuples recurse;
    scalars pass through.
    """
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = k.strip() if isinstance(k, str) else k
            if key not in out:
                out[key] = _normalized_keys(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_normalized_keys(v) for v in value]
    if isinstance(value, str):
        # LLMs sometimes stringify nested objects ('{"claim": ...}') — unwrap.
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return _normalized_keys(json.loads(stripped))
            except (ValueError, TypeError):
                return value
    return value


# Process-wide tally of model outputs that only validated after extra keys
# were stripped (by domain model name). Read as a before/after delta by the
# replay's per-day validation log; never reset, never consulted for logic.
VALIDATION_REPAIRS: Counter[str] = Counter()


def _validated(model_cls: type, payload: Any) -> Any:
    """Validate an LLM-output payload, tolerating (and logging) sloppy keys.

    Domain models are ``extra="forbid"`` — right for our own code, wrong at the
    model-output boundary: backends without server-side schema enforcement
    (the Claude Code CLI path) may emit extra keys or whitespace-padded key
    names at ANY nesting depth, and that must not void a whole verdict.
    Strategy: normalize keys recursively, validate, and on extra-key errors
    delete exactly the offending paths (reported by the validator) and retry.
    Required fields and value types stay strictly enforced.
    """
    from pydantic import ValidationError

    payload = _normalized_keys(payload)
    repaired = False
    for _attempt in range(3):
        try:
            result = model_cls.model_validate(payload)
            if repaired:
                VALIDATION_REPAIRS[model_cls.__name__] += 1
            return result
        except ValidationError as exc:
            extras = [e["loc"] for e in exc.errors() if e["type"] == "extra_forbidden"]
            if not extras or not isinstance(payload, dict):
                raise
            repaired = True
            log.warning(
                "committee_payload_extra_keys_dropped",
                model=model_cls.__name__,
                paths=[".".join(str(p) for p in loc) for loc in extras[:8]],
            )
            for loc in extras:
                parent: Any = payload
                for part in loc[:-1]:
                    if isinstance(parent, dict):
                        parent = parent.get(part)
                    elif isinstance(parent, list) and isinstance(part, int) and part < len(parent):
                        parent = parent[part]
                    else:
                        parent = None
                        break
                if isinstance(parent, dict):
                    parent.pop(loc[-1], None)
    result = model_cls.model_validate(payload)
    if repaired:
        VALIDATION_REPAIRS[model_cls.__name__] += 1
    return result


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


def compute_disagreement(*, bull_conviction: float, bear_conviction: float) -> float:
    """Conviction-weighted dispersion of the two ADVOCATES, in [0, 1] (doc 03 §4).

    Measures bull-vs-bear disagreement only. The arbiter's own action term
    (formerly ``ACTION_SCORE[action] / 2.0``, a third value in the dispersion)
    was removed for two reasons, both measured on run replay-e5a49317bbee:

    1. **Circularity.** The score gates ``buy``/``add``, yet the action term was
       its single largest contributor — choosing ``buy`` manufactured the
       disagreement that then vetoed ``buy``. Counterfactual mean with the term
       was 0.575 against an observed 0.273, and ``add`` (score 1.0 vs buy's 2.0)
       could never exceed 0.483, making ``add`` mechanically safer than ``buy``
       for reasons unrelated to the evidence.
    2. **Scale mismatch.** ``ACTION_SCORE[BUY] / 2.0`` is 1.0, while advocate
       convictions empirically live in 0.22-0.58; the action term sat outside
       the range of the values it was being pooled with.

    What the guardrail actually needs to know — do the two independently-run
    advocates disagree? — is a property of the evidence, knowable before the
    arbiter speaks. Range is preserved: two maximally-convicted opposed
    advocates still score 1.0, a unanimous committee still scores 0.0.
    """
    values = [bull_conviction, -bear_conviction]
    weights = [bull_conviction, bear_conviction]
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


def _normalize_kill_case_text(text: str) -> str:
    """Casefold, drop a leading severity label, collapse to bare word tokens."""
    return " ".join(_WORD_RE.findall(_SEVERITY_PREFIX_RE.sub("", text.strip()).casefold()))


def _content_words(normalized: str) -> set[str]:
    return {w for w in normalized.split() if len(w) > 2 and w not in _STOPWORDS}


def _texts_match(kill_case_text: str, rebuttal_text: str) -> bool:
    """Tier-2 similarity: containment, then Jaccard, then sequence ratio."""
    a = _normalize_kill_case_text(kill_case_text)
    b = _normalize_kill_case_text(rebuttal_text)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    words_a, words_b = _content_words(a), _content_words(b)
    if words_a and words_b:
        jaccard = len(words_a & words_b) / len(words_a | words_b)
        if jaccard >= REBUTTAL_JACCARD_THRESHOLD:
            return True
    # autojunk=False is load-bearing: difflib's "popular element" heuristic
    # activates above 200 elements and treats common letters as junk, which
    # collapses the ratio of long prose pairs (measured 0.19 vs a true 0.63).
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return ratio >= REBUTTAL_RATIO_THRESHOLD


def _unrebutted_high_kill_cases(
    kill_cases: Iterable[KillCase], rebuttals: Iterable[Rebuttal]
) -> list[KillCase]:
    """High-severity bear kill cases with no substantive rebuttal (doc 03 §4).

    A rebuttal only counts if it carries at least one Evidence item. Genuine
    rebuttals in run replay-e5a49317bbee carried 2-5, so the bar costs nothing
    and stops the tier-2 matcher being satisfied by a rebuttal that merely
    restates the kill case's name.

    Matching runs in two tiers:

    - **Tier 1 — index.** ``Rebuttal.kill_case_index`` is the join key: the
      bear's kill cases are numbered in ``arbiter_context_for`` and the arbiter
      is instructed to echo the integer back. Exact, order-stable, unambiguous.
    - **Tier 2 — text similarity.** For payloads with no index (replayed
      history, or a model that omitted the field) fall back to normalized text:
      casefold, strip a leading ``HIGH:``/``MEDIUM:`` label, reduce to word
      tokens, then accept on substring containment, on Jaccard of content words
      >= 0.4, or on ``difflib`` sequence ratio >= 0.6.

    Tier 2 exists because the original implementation was raw substring
    containment on un-normalized text, which failed 515 of 516 times (99.81%)
    on real arbiter output — the arbiter paraphrases and prefixes a severity
    label when restating an objection — making ``buy`` unreachable.
    """
    substantive = [r for r in rebuttals if len(r.evidence) >= 1]
    indexed: set[int] = {r.kill_case_index for r in substantive if r.kill_case_index is not None}
    unindexed_texts = [r.kill_case for r in substantive if r.kill_case_index is None]

    unmatched: list[KillCase] = []
    for position, kc in enumerate(kill_cases):
        if kc.severity is not Severity.HIGH:
            continue
        if position in indexed:
            continue
        if any(_texts_match(kc.description, text) for text in unindexed_texts):
            continue
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
        parallel_advocates: bool = False,
    ) -> None:
        missing = [name for name, _, _ in STAGES.values() if name not in prompts]
        if missing:
            raise ValueError(f"missing prompts: {missing}")
        missing_schemas = [key for _, key, _ in STAGES.values() if key not in schemas]
        if missing_schemas:
            raise ValueError(f"missing schemas: {missing_schemas}")
        self._model = research_model
        self._parallel_advocates = parallel_advocates
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
        dossier = _validated(
            DossierAnalysis,
            _filtered_payload(
                DossierAnalysis,
                dossier_raw,
                {
                    "symbol": symbol,
                    "as_of": as_of,
                    "prompt_version": self._prompts["dossier_analyst"].version,
                },
            ),
        )

        advocate_context = advocate_context_for(context, dossier_raw)
        bear_raw: dict[str, Any] | None = None
        if self._parallel_advocates:
            # Neither advocate reads the other's brief, so they can overlap.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="advocate") as pool:
                bull_future = pool.submit(self._call_stage, "bull", symbol, as_of, advocate_context)
                bear_future = pool.submit(self._call_stage, "bear", symbol, as_of, advocate_context)
                bull_raw, bear_raw = bull_future.result(), bear_future.result()
        else:
            bull_raw = self._call_stage("bull", symbol, as_of, advocate_context)
        bull = _validated(
            AdvocateBrief,
            _filtered_payload(AdvocateBrief, bull_raw, {"symbol": symbol, "side": Side.BULL}),
        )
        if bear_raw is None:
            bear_raw = self._call_stage("bear", symbol, as_of, advocate_context)
        bear = _validated(
            AdvocateBrief,
            _filtered_payload(AdvocateBrief, bear_raw, {"symbol": symbol, "side": Side.BEAR}),
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
            bull_conviction=bull.conviction, bear_conviction=bear.conviction
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
                f"unrebutted high-severity bear kill case: {kc.description}" for kc in unrebutted
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
                "role": stage,  # ledger trace only; never sent to the model
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


KILL_CASE_INDEX_HEADING = "## Bear kill cases (indexed)"


def render_indexed_kill_cases(bear_payload: Mapping[str, Any]) -> str:
    """Number the bear's kill cases so rebuttals can cite them by integer.

    The rendered position IS the join key `_unrebutted_high_kill_cases` uses,
    so it must enumerate `bear_payload["kill_cases"]` in payload order with no
    filtering — the bear brief is validated from the same list.
    """
    raw = bear_payload.get("kill_cases")
    entries = list(raw) if isinstance(raw, (list, tuple)) else []
    lines = [
        KILL_CASE_INDEX_HEADING,
        "Each rebuttal in bear_rebuttals MUST set kill_case_index to the integer i "
        "of the kill case it answers, exactly as numbered here.",
    ]
    if not entries:
        lines.append("(none)")
        return "\n".join(lines)
    for i, kc in enumerate(entries):
        severity = str(kc.get("severity", "?")) if isinstance(kc, Mapping) else "?"
        description = str(kc.get("description", "")) if isinstance(kc, Mapping) else str(kc)
        lines.append(f"kill_cases[{i}] (severity={severity}): {description}")
    return "\n".join(lines)


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
        render_indexed_kill_cases(bear_payload),
        "## Quant score (JSON)",
        json.dumps(quant_score.model_dump(mode="json"), sort_keys=True)
        if quant_score is not None
        else "null",
    ]
    prefix = advocate_context_for(context, dossier_payload)
    return prefix + CONTEXT_CACHE_BOUNDARY + "\n\n".join(tail)
