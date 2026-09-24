# Wide-Campaign Results — 2026-03-03 → 2026-08-14 (115 sessions)

The first full execution of `runbook_wide_campaign.md`: four replay arms over the same
115-session window, all under the anti-leakage package (runtime tripwires, model-cutoff
gate, persisted contexts, post-run audit). Committee run `replay-63220a3a5e9a`, Sonnet-5
arbiter + analysts, Haiku-4.5 triage, claude_code backend — **$0.00 metered spend**.

**Standing caveats (do not quote numbers without them):** the quant model (campaign-v8)
is a NO-GO bundle run deliberately as a mechanics control; the panel is survivorship-biased
(99.87% survivors — optimistic, concentrated in flagged/promoted names); volume is IEX-only
(never quote capacity/slippage); SPY's +14.7% over this window is a strong-market baseline.

## CORRECTION 2026-09-23 — the committee arm was index exposure, not skill

Applying the methodology from paperswithbacktest/awesome-systematic-trading
(report beta-adjusted skill beside every headline) changes this page's main
conclusion. The beta and alpha below were in each run's `report.json` all
along; they were never printed or quoted. Re-verified independently,
date-matched against SPY:

| arm | return | beta | alpha / yr | alpha t-stat |
|---|---|---|---|---|
| quant, wide-traded | +7.80% | 1.07 | −14.84% | −0.76 |
| quant, core control | +14.57% | 0.87 | +3.49% | +0.58 |
| quant, true discovery | +10.65% | 0.97 | −6.73% | −0.62 |
| **committee** | +12.80% | **0.91** | **−1.01%** | **−0.41** |

**No arm has alpha distinguishable from zero.** The committee's +12.80% is
what 0.91 x SPY earned (+13.36%), and its 1.66% tracking error is the tell:
reconstructed from its fills, the committee portfolio was on average **86%
SPY ballast, 1% committee picks, 13% cash**, and held two symbols in 115
sessions (SPY and MCD). How 72 buy/add verdicts became one position:

1. The portfolio pass ran tier-1 only with discovery off, so verdicts on 251
   of the 267 debated symbols (56 of the 72 buys) never reached sizing.
2. Of the 16 that did, 36 decisions were crypto buys, all zeroed by the
   understated-crypto-volume bug fixed before launch.
3. What remained traded MCD; the idle-cash ballast held the rest in SPY.

So "committee beat both quant arms on every metric" is true only in the
sense that holding SPY beat the quant picks. **The committee's stock
selection was never tested by this campaign.** The "Sharpe 2.08" is largely
SPY's own Sharpe in a strong window.

Also from that repo: a strategy needs roughly (1.96 / IR)^2 years to prove
itself. At IR 0.5 that is ~15 years; a 115-session window can only
demonstrate skill above IR ~2.9. The paper run will validate mechanics for
months before it can say anything statistical about edge.

`replay compare` and `replay report` now print beta, alpha, IR, alpha
t-stat and the proof horizon (commit 2d6fb8c). The original text below is
left as written.

## Headline table

| arm | universe | return | sharpe (defl.) | maxDD | turnover | costs | vs SPY |
|---|---|---|---|---|---|---|---|
| quant, wide-traded (`replay-699b62040d45`) | 10.4k traded | +7.80% | 0.92 (0.73) | −10.83% | 6.9× | $1,981 | −6.92% |
| quant, core control (`replay-ad0c913a7f4c`) | 55 core | +14.57% | 2.33 | −7.19% | 3.9× | $316 | ≈0% |
| quant, true discovery (`replay-0239ea6a1d37`) | core + promoted | +10.65% | 1.50 | −7.60% | 5.9× | $930 | −3.92% |
| **committee** (3 bankrolls, `replay-c41988c44988` et al.) | core + haiku-promoted | **+12.80%** | **2.08 (0.92)** | **−7.19%** | **2.1×** | **$115** | **−1.92%** |

Bankroll invariance held: $1k +12.82% / $10k +12.81% / $100k +12.80% (fractional shares,
identical 63 fills each).

## What the campaign proved (mechanics — its actual purpose)

1. **`buy` is reachable.** 66 buys + 6 adds in 687 verdicts (9.6% buy rate) vs **0 buys in
   295 verdicts** in the 74-day control run, where the kill-case substring matcher made
   buying mechanically impossible. Buy conviction spans 0.42–0.62 across 11 distinct values
   (control: 98.3% of mass on 4 values, ceiling 0.58, bar never cleared).
2. **Breadth fixed.** 229 distinct symbols debated (control: 20, with WMT alone taking 70%
   of slots). Largest debate share now: the crypto pairs (see findings).
3. **The committee changes its mind.** Re-reviews of previously-bought names: 44 re-buys,
   33 no_action, **11 avoids**, 3 adds, 1 hold. It never used the literal `trim`/`sell`
   actions — exits are the risk layer's job (stops, 20-session horizon, drawdown ladder;
   all 63 portfolio fills' exits were rule-driven) — but the 11 avoids show re-evaluation
   is genuine, not rubber-stamping.
4. **Crash-safe and resumable in practice, not just design:** the run survived one CLI-hang
   crash (fixed with a retry, `f2947dd`) and one full host/Docker outage at day 104,
   resuming both times with zero lost days.
5. **Full pipeline exercised end-to-end:** discovery (haiku gate: 2,504 flags, 165 triage
   calls, 56 promotions), deep dives (66), verdict persistence, three portfolio passes,
   reporting, audit. Total metered cost of the entire committee arm: **$0.00**.

## Leakage audit — PASS after review

Config gate PASS (Sonnet-5 cutoff 2026-01-31 < window start; nothing acknowledged).
Context hashes verified. The date-scanner initially flagged 954 items; 672 were the
scanner reading decimal tails as years (`5.2089%` → "2089") — fixed in the scanner, with
a regression test. All 282 remaining findings are forward-looking references inside
properly-bounded news (announced dividend dates, "Where Will Nvidia Be in 2030?"-class
headlines published before as_of). No instance of post-as_of knowledge was found.

## Findings for the next iteration

- **Crypto over-representation:** the 5 crypto pairs took 87 debate slots (13%) — the
  reserved `--crypto-slots 1` against a 5-name crypto universe with a 5-session cooldown
  recycles them constantly. Consider widening the crypto set or shrinking the slot.
- **July buy-drought:** buys 59→61 across sim-July (≈20 sessions). Not investigated; could
  be honest caution in a toppy tape or conviction-threshold friction. The per-day dossiers
  are persisted for whoever digs.
- **Committee beat both quant arms on every metric that matters** (return, Sharpe, DD,
  turnover, costs) *with the same NO-GO model feeding its shortlist* — judgment recovered
  most of what the weak ranker gave up (−1.92% vs SPY against the wide arm's −6.92%).
  That is evidence the committee adds value over the screen, not proof of alpha: one
  strong-market window, survivorship-biased panel.
- **Discovery-as-alpha is still negative** with this model (true A/B: −3.92%); the
  mechanism (scan → flag → triage → promote → decide) is fully functional. Better model
  first, then re-measure.

## Reproduction

Runbook: `docs/runbook_wide_campaign.md` (corrected Stage C, `cfae52e`). Committee arm:
Stage D command with `--quant-model-version campaign-v8 --no-require-shipping-bundle`,
`CLAUDE_CODE_MODEL_ARBITER=claude-sonnet-5`. Artifacts under
`artifacts/backtests/replay-63220a3a5e9a/` (leakage audit) and the three portfolio run
dirs (reports). All runs resumable by re-issuing the identical command.
