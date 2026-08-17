"""Per-run committee-replay budget: hard cap, projection, durable spend.

Plan Phase 5. Two layers, both mandatory:

1. **This module** — the RUN-LOCAL cap. ``ReplayBudget`` refuses any
   submission whose *projected* cost would push accumulated run spend past
   ``--max-usd`` (``ReplayBudgetExhausted``), and records every ACTUAL dollar
   of usage to ``replay_runs.spent_usd`` as it happens, so a crashed or
   restarted run resumes with accurate accounting (the driver seeds
   ``prior_spent_usd`` from the run row).
2. **The live calendar-month ``CostMeter``** — the OUTER bound. The replay
   CLI wires the real ``AnthropicClient`` with the real monthly meter on REAL
   wall clock (historical time is never injected into it): even a
   mis-specified ``--max-usd`` can never blow through the monthly
   kill-switch. :class:`BudgetLedgerTee` wraps the meter's ledger writer so
   every metered entry ALSO accumulates into the per-run budget — actuals
   come from the client's own usage records, never re-estimated.

Projection is a CONSERVATIVE OVERESTIMATE by construction (documented in
:func:`estimate_request_cost`): input tokens at ~3 chars/token (typical
English prose is ~4), output at the full ``max_tokens`` ceiling, no
prompt-cache discount and no 50% batch discount applied. Real spend is
therefore expected to come in well under projection; the cap can refuse
early, never late.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

from dynamo.llm.cost_meter import LedgerEntry, LedgerWriter
from dynamo.llm.pricing import calculate_cost

# Conservative token estimate: ~3 characters per token (typical English prose
# runs ~4 chars/token, JSON/code a bit lower) so the projection overestimates.
CHARS_PER_TOKEN = 3.0

# Assumed context size (chars) for one replay dossier context in the dry-run
# estimator. Replay contexts are THIN (no filings/transcripts, sparse news);
# live thin dossiers measure ~2-4k chars, so 6k chars is the conservative
# ceiling used when no real context exists yet.
ASSUMED_CONTEXT_CHARS = 6_000

# Assumed context size (chars) for one replay macro-digest request (series
# lines + "none" PM section) — generous vs the live _macro_context output
# (a dozen short lines), so the projection errs high.
ASSUMED_MACRO_CONTEXT_CHARS = 2_000


class ReplayBudgetExhausted(RuntimeError):
    """Projected spend would exceed the run's --max-usd cap; nothing submitted."""


class SpendSink(Protocol):
    """The slice of ``ReplayRepository`` the budget writes through."""

    def record_spend(self, run_id: str, delta_usd: float) -> None: ...


class ReplayBudget:
    """Run-local spend accounting + the hard per-run dollar cap."""

    def __init__(
        self,
        *,
        cap_usd: float,
        run_repo: SpendSink,
        run_id: str,
        prior_spent_usd: float = 0.0,
    ) -> None:
        if cap_usd <= 0:
            raise ValueError(f"cap_usd must be positive, got {cap_usd}")
        self.cap_usd = float(cap_usd)
        self.run_id = run_id
        self._runs = run_repo
        # Resume accounting: a restarted run starts from the durable
        # replay_runs.spent_usd, not from zero.
        self._spent = max(float(prior_spent_usd), 0.0)

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def remaining_usd(self) -> float:
        return max(self.cap_usd - self._spent, 0.0)

    def check_projection(self, projected_usd: float) -> None:
        """Refuse (raise) when spent + projection would exceed the cap.

        Boundary: a projection EXACTLY consuming the remainder is allowed;
        one dollar-cent of overshoot refuses. The projection itself is a
        conservative overestimate (module docstring), so acceptance here
        implies the actual cost fits.
        """
        projected = max(float(projected_usd), 0.0)
        if self._spent + projected > self.cap_usd:
            raise ReplayBudgetExhausted(
                f"run {self.run_id}: projected ${projected:.2f} on top of "
                f"${self._spent:.2f} spent would exceed the ${self.cap_usd:.2f} cap"
            )

    def record_actual(self, delta_usd: float) -> None:
        """Accumulate ACTUAL metered cost and checkpoint it durably.

        Called per ledger entry (finer-grained than the per-stage minimum the
        plan requires) so a crash between stages — or mid-collection — never
        loses spend."""
        delta = float(delta_usd)
        if delta <= 0.0:
            return
        self._spent += delta
        self._runs.record_spend(self.run_id, delta)


class BudgetLedgerTee:
    """LedgerWriter wrapper: forwards to the real (monthly) ledger AND feeds
    every entry's actual cost into the per-run :class:`ReplayBudget`.

    The monthly CostMeter keeps REAL wall clock and the real cost_ledger
    table; this tee only observes.
    """

    def __init__(self, inner: LedgerWriter, budget: ReplayBudget) -> None:
        self._inner = inner
        self._budget = budget

    def write(self, entry: LedgerEntry) -> None:
        self._inner.write(entry)
        self._budget.record_actual(entry.cost_usd)

    def month_to_date_usd(self, month_start: datetime) -> float:
        return self._inner.month_to_date_usd(month_start)


# --------------------------------------------------------------------------
# projection / dry-run estimation (pure math, ZERO API calls)
# --------------------------------------------------------------------------


def estimate_tokens(chars: int) -> int:
    """Conservative char->token estimate (~3 chars/token, ceil)."""
    return math.ceil(max(chars, 0) / CHARS_PER_TOKEN)


def estimate_request_cost(
    model_id: str,
    *,
    prompt_chars: int,
    max_tokens: int,
) -> float:
    """Conservative dollar projection for ONE request.

    Assumptions (all err high): every prompt char bills as fresh input at
    ~3 chars/token — no prompt-cache reads; the response consumes the FULL
    ``max_tokens``; no batch discount even though committee replay submits
    via the Message Batches API (50% off in reality).
    """
    return calculate_cost(
        model_id,
        input_tokens=estimate_tokens(prompt_chars),
        output_tokens=max(int(max_tokens), 0),
        batch=False,
    )


def estimate_batch_cost(requests: Any, tier_models: Mapping[str, str]) -> float:
    """Projected cost of a stage batch built by
    ``CommitteeEngine.build_batch_requests`` (system + context + instruction
    chars per request, priced per its tier's model)."""
    total = 0.0
    for req in requests:
        model_id = tier_models[str(req["model_tier"])]
        chars = len(str(req["system"])) + len(str(req["context"])) + len(str(req["instruction"]))
        total += estimate_request_cost(
            model_id, prompt_chars=chars, max_tokens=int(req.get("max_tokens", 4096))
        )
    return total


def estimate_deep_dive_cost(
    tier_models: Mapping[str, str],
    stage_specs: Mapping[str, tuple[str, int, int]],
    *,
    context_chars: int = ASSUMED_CONTEXT_CHARS,
) -> float:
    """Dry-run projection for ONE full 4-stage deep dive (one symbol).

    ``stage_specs``: stage -> (model_tier, system_prompt_chars, max_tokens).
    Later stages' contexts grow by earlier stages' JSON output, approximated
    conservatively as each feeding stage's full ``max_tokens`` re-billed as
    input characters (tokens * CHARS_PER_TOKEN): the dossier JSON feeds
    bull/bear; dossier + bull + bear JSON all feed the arbiter.
    """

    def out_chars(stage: str) -> int:
        return int(stage_specs[stage][2] * CHARS_PER_TOKEN)

    base = max(int(context_chars), 0)
    context_for = {
        "dossier": base,
        "bull": base + out_chars("dossier"),
        "bear": base + out_chars("dossier"),
        "arbiter": base + out_chars("dossier") + out_chars("bull") + out_chars("bear"),
    }
    total = 0.0
    for stage in ("dossier", "bull", "bear", "arbiter"):
        tier, system_chars, max_tokens = stage_specs[stage]
        total += estimate_request_cost(
            tier_models[tier],
            prompt_chars=system_chars + context_for[stage],
            max_tokens=max_tokens,
        )
    return total


def estimate_run_cost(
    *,
    n_sessions: int,
    deep_dives_per_day: int,
    per_dive_usd: float,
    per_session_usd: float = 0.0,
) -> float:
    """sessions x shortlist-cap x per-deep-dive projection, plus an optional
    flat per-session overhead (the replay macro digest when enabled).
    Conservative: it assumes the shortlist is FULL every session (real days
    often select fewer, or none when no BUY candidates exist) AND that every
    session generates a digest (days with a persisted one skip it)."""
    sessions = max(n_sessions, 0)
    return sessions * max(deep_dives_per_day, 0) * max(per_dive_usd, 0.0) + sessions * max(
        per_session_usd, 0.0
    )
