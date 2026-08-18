# ADR-0015: Alpha-research rails — pre-registered, trial-counted auditions

Status: accepted
Date: 2026-08-18

## Context

The platform has honest evaluation machinery for the features it already
ships: purged/embargoed walk-forward (doc 08 §4), a persistent trial counter
and deflated Sharpe (doc 08 §6), the alphalens signal audit
(`backtest/signal_audit.py`), and a promotion pipeline no research code can
write to. What it lacks is a disciplined path for *new* signal ideas.

The danger is specific to our setting. The universe is small (tens of
symbols, not thousands), so cross-sectional statistics are noisy and the
false-discovery rate under casual testing is enormous: with ~40 names, a
zero-skill candidate's mean daily rank IC over a year has a wide null
distribution, and anyone who tries ten ideas and keeps the best will
"discover" alpha almost surely. A future weekly research agent will generate
candidate ideas in volume, which makes uncontrolled testing a guaranteed
self-deception machine unless the rails are built first and the agent is
forced through them.

## Decision

Build **rails, not an agent**: `research/discovery/` (pure core),
`persistence/repositories/discovery.py`, and `dynamo.cli.alpha`.

1. **Pre-registration** (`hypotheses` table, migration 0018). A hypothesis —
   title, rationale (WHY it should work), prediction (expected
   sign/magnitude), data requirements, trial family — must be written down
   in a *prior* CLI invocation before any test runs. The audition refuses
   any candidate whose `registered_at` is not strictly before the audition's
   start instant. This is the anti-data-snooping rule in code: you cannot
   peek at results and then "predict" them.

2. **One audition harness**, reusing the existing machinery, never a
   parallel one:
   - alphalens signal audit on the candidate alone (optional extra, guarded
     degradation like every existing call site);
   - an **incremental walk-forward value test**: pooled OOS rank IC of the
     existing feature panel with vs without the candidate, both arms on
     bit-identical purged/embargoed folds — a candidate must add value the
     51 features do not already carry;
   - a **cross-sectional permutation null** (N=200): within each session the
     candidate's values are shuffled across symbols and the mean per-session
     rank IC recomputed; we report where the real IC falls in that null.
     This was chosen over mapping IC into deflated-Sharpe space because it
     is cleaner to defend: no return-distribution assumptions, and the null
     width is *measured* on our actual small universe rather than assumed.

3. **Trial counting.** Every audition increments the persistent
   `trial_counter` row for the hypothesis's family — before the verdict is
   computed, because looking at OOS results is a spent trial whatever the
   outcome. The family's count feeds the third verdict bar: the real IC must
   exceed `expected_max_sharpe(family_trials, var(null ICs))` — the Bailey &
   López de Prado expected-maximum formula fed with the empirically
   estimated permutation-null variance, i.e. "the best IC a zero-skill
   family of this size would be expected to show". The bar rises with every
   trial the family burns.

4. **Verdict thresholds** (constants in `research/discovery/audition.py`,
   with inline rationale): IC delta > 0, null percentile >= 95, and the
   trials-adjusted expected-max bar. Every audition — pass or fail — appends
   a row to `docs/alpha_research_log.md` and writes its full evidence to the
   hypothesis row (`audit_results` jsonb). Failed trials stay on the record;
   hiding them is how false discovery happens.

5. **Human-approved proposals only.** A passed audition may be turned into a
   `NEW_FEATURE` `ImprovementProposal` (status OPEN) via
   `dynamo.cli.alpha propose`. Proposals are never applied automatically —
   accepted ones go through the normal PR + promotion pipeline (doc 04 §2:
   catalog additions are code changes with golden tests).

## Explicit non-goals (hard lines)

- The rails **never run scheduled** — every command is a manual invocation.
- **Nothing auto-promotes**: the rails can reject ideas and open proposals;
  they cannot change the catalog, the registry, or production behavior.
- **Zero LLM involvement** in this layer. The future weekly research agent
  will *use* these rails (register hypotheses, run auditions, read
  verdicts); it gets no other path to the evaluation machinery, and its
  proposals land in the same human-approval queue.

## Consequences

- Candidate ideas cost a trial from their family's budget every time they
  are looked at, so idea generation is pushed toward fewer, better-reasoned
  hypotheses — the intended incentive.
- The permutation null is O(permutations × sessions) and runs in seconds on
  a laptop for our panel sizes; no new dependencies were added.
- The `hypotheses` table is append-mostly; pre-commitment fields are never
  updated after registration (only `status` and `audit_results` change).
