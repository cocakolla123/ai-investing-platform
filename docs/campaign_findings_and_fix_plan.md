# Committee campaign: findings and fix plan

**Source run:** `replay-e5a49317bbee` (`campaign-committee-v1`), 2026-03-03 → 2026-06-16,
74 sessions, 295 verdicts, 4 deep dives/day, Fable arbiter via the Claude Code backend.
**Portfolio result:** +5.14% vs SPY +10.29% over the same window (max drawdown −6.34%).
**Verdict distribution:** 275 `no_action`, 14 `avoid`, 6 `hold`, **0 `buy`** (stored).

The run was stopped at 64% completion once the diagnosis was clear. It stands as the
**control experiment**: "committee with a corrupted feature pipeline, a fundamentals-free
dossier, and a broken guardrail produces zero conviction." Everything below is measured
evidence from three independent read-only investigations, each verified by hand.

---

## Four independent defects, in dependency order

### 1. The feature pipeline is corrupted by calendar mixing (deepest)

`catalog.py::_pivot` builds ONE price panel indexed by the **union** of every symbol's bar
timestamps. Crypto trades 7 days/week, equities 5. The panel therefore has 421 calendar rows
where equities have only 288 real ones, and `_roll` requires `min_periods == window`.

**Verified by hand:** at 2026-03-03 the 50-day moving average is computable for **5 of 55
symbols — exactly the 5 crypto pairs.** All 50 equities are NaN.

- **16 features are NaN for every equity**: `dist_ma50`, `dist_ma200`, `ma50_slope`,
  `ret_10d_z`, `drawdown_20d_high`, `vol_cc_20d`, `vol_parkinson_20d`, `vol_of_vol_60d`,
  `vol_pctile_2y`, `atr14_over_price`, `dollar_vol_20d`, `dollar_vol_z_60d`,
  `volume_trend_20_60`, `amihud_20d`, `days_to_exit`, `dist_20d_mean_sigma`.
- Those features carry **28.1% of the trained model's importance mass**.
- **Train/serve skew, precisely dated**: crypto bars begin 2024-01-01; the bundle trained on
  2020-2025. Equity coverage was 0.669 at 2023-12-15 and **0.287 at 2024-01-15** — the day
  crypto entered the panel.
- Only **7 of 51 features vary day-to-day** across equities → day-over-day rank correlation
  **0.88**, still **0.55 at lag 40**. The ranking is frozen; hence WMT debated 52 of 74 days
  and AAPL/GOOGL/AMZN/MSFT never once, despite complete data.
- Crypto (9% of the universe) took **29% of debate slots** because they are the only names
  whose momentum/volatility features survive.

**Collateral damage — silent constant fills that hide the outage:**
- `_rsi_14` fills exactly `50.0` whenever the denominator is NaN → all 50 equities get a
  fabricated RSI of 50 that counts as "covered".
- `_breadth_above_200dma` computes breadth from whichever names have a valid 200-day MA —
  i.e. the 5 crypto only — and broadcasts it. Read 0.0 every sampled day.
- `_ret(days)` / `_mom_12_1` index positionally into a calendar panel, so an equity
  "20-day return" is really 21 calendar days and NaNs whenever the offset lands on a weekend.
- Consequently `feature_coverage` **overstates** true coverage and `confidence` is clamped to
  **0.2000 on all 4,070 scored rows** (and is never read by any selection code anyway).

**Consequence for the model:** `campaign-v3` loses to its own historic-average baseline on all
three tasks (rank IC 0.049 vs 0.093; downside AUC 0.519 vs 0.721; qlike 0.501 vs 0.342),
`ships_ok=False` on every task. It was never a modelling failure — it was fed a broken matrix.
Preflight only asserted the bundle *loads*; it never consulted the ship gate.

### 2. `buy` is mechanically unreachable (proven, decisive)

To emit `buy` the arbiter must rebut every high-severity bear kill case. The code
(`committee.py:213`) tests this with **raw substring containment** between the rebuttal's
`kill_case` string and the bear's `description`. The arbiter paraphrases and prefixes
`"HIGH: "` when restating each objection, which breaks containment.

- **Match failure rate: 515 of 516 high-severity kill cases (99.81%).**
- 293 of 295 verdicts carried ≥1 phantom "unrebutted" kill case.
- Counterfactual: had every arbiter emitted `buy`, **99.7% would have been blocked** (99.3%
  by this matcher, 26.4% by the disagreement veto).
- **It fired for real:** the model chose `buy` on ABBV on 2026-05-08, 05-13 and 05-27 — all
  three silently downgraded to `hold`. Those verdicts contained cited earnings beats, five
  evidence items per rebuttal, and six machine-checkable exit conditions.

**Second, latent blocker:** `compute_disagreement` mixes `ACTION_SCORE[BUY]/2.0 = 1.0` with
advocate convictions that empirically live in 0.22–0.58. **Choosing `buy` is itself what
manufactures the disagreement that vetoes it** (counterfactual mean 0.575 vs observed 0.273;
`add` never exceeds 0.483, so `add` is mechanically safer than `buy` — an artifact).

### 3. The committee is starved of evidence

`assemble_dossier_context` has **no parameter for fundamentals** — a design gap, not a wiring
gap. 944,859 `xbrl_facts` rows (revenue, margins, balance sheet, cash flow, EPS for all 40
operating companies, current through 2026-06-30) are compressed into a single
`expected_excess_return` scalar before the LLM ever sees them.

Population rates across 300 dossiers: filings **0%** (table empty), journal hits **0%** (table
empty), transcripts 28%, news 43% — and **only 18.9% of news items name the subject company**
(NVDA's digest contained items on Disney, Micron, Buffett, AMD and OpenAI). Median dossier:
4,130 characters; thinnest 1,153. Every one of the 295 verdicts declared a filings gap; 90.8%
declared a fundamentals gap.

**Measured effect:** context length → bull conviction **r = +0.80**; → bear conviction −0.43.
Stratified by whether a transcript happened to be present:

| Dossier | Bull conv | Bear conv |
|---|---|---|
| No transcript, no news | 0.308 | 0.394 (bear leads) |
| Transcript present | **0.435** | **0.349** (bull leads) |

**Max bull conviction in 74 days: 0.58.** Never once cleared 0.6.

**Prompt asymmetry compounds it.** `bear_advocate.md` tells the bear its high-severity kill
cases are a veto ("so make them sharp"); the bull has no equivalent lever. Result: 516 bear
high-severity kill cases vs 117 bull (4.4×), with 222 verdicts carrying exactly 2 — a quota.
`arbiter.md` states abstention is "explicitly rewarded" and "never penalized", making the loss
function one-sided. 99.7% of verdicts echo that framing back. The bear then weaponizes the
data gap directly: 177 briefs argue "no valuation anchor exists" and 63 kill cases cite
*missing data* as their evidence — a move the bull cannot mirror, since unreferenced points
are discarded.

Also unanchored: `conviction` has no description, default or scale in the schema, producing
8 distinct values across 295 verdicts with 98.3% inside {0.55, 0.58, 0.60, 0.62} and
|r| < 0.15 against every available feature. It is noise, not signal.

### 4. The drawdown ladder whipsaws the index ballast

**Verified from the equity curve:** on 2026-04-01 the ladder hit the `dd5` rung and cut
invested exposure from 94.9% to **47.7%**, then restored it to 95.2% by 2026-04-15 — selling
half the index near the local bottom (equity 953) and rebuying higher (985). This accounts for
most of the gap between the portfolio's +5.14% and SPY's +10.29%.

The ladder is correct for *active* positions (cut risk when a thesis is going wrong). Applied
to a *passive index floor* it is market timing, and it lost money.

---

## What was ruled out

- **Reasoning quality**: the arbiter's output is substantive, symbol-specific, evidence-cited
  research with named biases and machine-checkable exit conditions. Output quality tracks
  input quality.
- **The subscription backend**: 295 of 296 dives completed, zero quarantines, no truncation,
  all verdicts stamped `claude-fable-5` / prompt `1.0.0`.
- **Temporal-discipline suppression**: the "disregard knowledge after {as_of}" instruction is
  not the cause; the context-length correlation shows genuine evidence starvation.
- **Stale persisted scores**: `quant_scores` is empty; every day was scored fresh.

---

## Fix plan

Sequenced so each phase is independently verifiable and the cheap unblocking work lands first.
Phases 0–3 cost **zero Fable** (Opus agents + local compute).

### Phase 0 — Make `buy` reachable (hours)
1. Replace substring matching in `_unrebutted_high_kill_cases` with index-based matching: add
   `kill_case_index: int` to `Rebuttal`, number the bear's kill cases in `arbiter_context_for`.
   Additionally require `len(rebuttal.evidence) >= 1` (currently unenforced; genuine rebuttals
   carry 2–5 items, so this is free).
2. Re-scale `compute_disagreement` so the action term sits on the advocates' empirical scale
   (or drop it and measure bull-vs-bear dispersion directly). Update doc 03 §4.
3. Regression test: replay the three stored ABBV payloads and assert the buys survive.

### Phase 1 — Repair the feature pipeline, then retrain (days)
4. Roll every window over each symbol's **own** observation sequence rather than the shared
   calendar (`_roll_per_symbol`), fixing all 16 dead features at once. Same for positional
   lookups in `_ret` / `_mom_12_1`.
5. Stop fabricating: `_rsi_14` fills 50.0 only when the denominator is genuinely 0, never on
   NaN. `_breadth_above_200dma` uses a fixed equity reference set.
6. Make `feature_coverage` asset-class aware (an ETF lacking `fcf_yield` is expected, not a
   defect), and have `synthesize_verdicts` gate on a minimum coverage so structurally
   handicapped names abstain instead of being ranked.
7. **Regression test that would have caught this**: equity feature coverage at a fixed date
   must be invariant to adding/removing crypto from the universe.
8. Retrain → `campaign-v4` (panel cache invalidates; ~3h, free). Gate: must beat the
   historic-average baseline. Make preflight consult `ships_ok`, not just "the bundle loads".
9. **With Tiingo Power**: expand the training panel to ~500 names (the single biggest quality
   lever; `--symbols-file` plumbing already exists).

### Phase 2 — Feed the committee (days)
10. Add a `fundamentals` parameter and `## Fundamentals` section to the dossier, rendered from
    `xbrl_facts` via the existing `pit_facts` / `ttm_sum` / `quarterly_series` helpers:
    TTM revenue + growth, margins and their trend, EPS, balance sheet (assets, liabilities,
    equity, LT debt, cash), cash flow (CFO, capex, FCF), and valuation multiples computed from
    `close` × shares outstanding — each with a citable `source_ref`.
    Handle: period-type inference, restatement dedup (latest `filed_at` ≤ as_of), tag
    normalization (test CAT, which has zero `StockholdersEquity` rows).
11. Symbol-filter the news digest (81% of current items are noise).
12. Add a peer/relative-strength section — sector ETFs and `instruments.sector` are already
    populated; needs computation only, no ingestion.
13. Extend price stats: 52-week high/low and distance, 60/120/250-day returns, drawdown, beta.
14. Fix the journal: real embedder backend, seed from the 114 existing `decision_experience`
    rows, so prior-case retrieval stops being a no-op.
15. Backfill the empty high-signal tables: `filings` (also unlocks the never-fed
    `filing_excerpts` parameter), `insider_tx`, `earnings_events`.

### Phase 3 — Rebalance the debate and diversify the slate (hours)
16. `arbiter.md`: make abstention symmetric ("declining when the evidence supported action is
    the same class of error as acting on weak evidence"), and soften the absolute kill-case bar
    to "unrebutted **and** materially more probable than the price already reflects."
17. `bear_advocate.md`: remove the veto disclosure; anchor severity ("reserve `high` for
    >20% probability of a >20% drawdown; most kill cases are medium"); forbid kill cases whose
    only evidence is the absence of data.
18. Anchor `conviction` in the schema with an explicit scale (0.50 = no edge, 0.60 = modest,
    0.75 = strong multi-source, 0.90 = near-certain).
19. Shortlist selection: sector-bucketed round-robin, crypto capped at 1 slot/day, a cooldown
    excluding names debated in the last K sessions, one slot reserved for the highest-ranked
    name not seen in 20 sessions, and widen the candidate band from top-20% to top-40%.
20. Ballast should not follow the active-risk ladder: give it its own much deeper de-risking
    schedule (a −5% portfolio dip is noise for a passive floor).

### Phase 4 — Validate cheaply (zero Fable)
21. Re-run a **short window (2–3 weeks)** with all fixes, arbiter on **Sonnet-5** — whose
    training cutoff (2026-01-31) is identical to Fable's, so the contamination gate still
    passes for a March-onward window. Compare verdict distribution, bull-conviction ceiling and
    shortlist diversity against this control run.

### Phase 5 — Full campaign (Fable, when the weekly budget resets)
22. Only once Phase 4 shows justified conviction: full clean-window campaign with Fable
    arbitrating, then the three bankroll passes, reports, compare and leakage audit.

---

## Budget note

A committee day costs 17 model calls: 13 on Sonnet (dossier/bull/bear/macro) and **only 4 on
Fable** (the arbiter). Fable is under a quarter of the burn, and Sonnet-5 shares Fable's exact
knowledge cutoff — so the entire validation loop can run at zero Fable cost. Reserve Fable for
the final campaign whose numbers you intend to quote.
