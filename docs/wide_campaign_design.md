# Wide campaign design — discovery-enabled backtest

**Status:** design, measured 2026-08-19 against the live DB. Supersedes the shape of
`replay-e5a49317bbee` (74 sessions, 55 mega-caps, **no discovery**).

---

## The pilot result — first evidence that discovery carries signal

Run read-only over 18 sampled sessions (every 5th, 2026-03-03 → 2026-07-06) on a
1,248-name liquid operating-company universe. Forward 20-session return in excess of SPY:

| Group | Mean | Median | Hit rate |
|---|---|---|---|
| **Flagged** (68/day) | **+0.75%** | −0.64% | 48.4% |
| Unflagged | **−0.47%** | −1.33% | 45.1% |
| Paired difference | **+1.22%** (sd 2.76, n=18, t=1.88) | | |

Per detector:

| Detector | Names/day | Mean fwd 20d excess |
|---|---|---|
| `range_bound` | 36.7 | **+1.35%** |
| `range_break` | 23.8 | +0.41% |
| `price_zscore` | 6.7 | −0.28% |
| `volume_spike` | 7.3 | **−3.03%** |

**Read this carefully — it is encouraging, not a pass.** (a) 18 overlapping 20-session
windows are not 18 independent observations; effective n ≈ 4–5, so t=1.88 is optimistic.
(b) Survivorship inflates every number and inflates the flagged group most. (c) Both
medians are negative — the mean is carried by a right tail, exactly what survivorship
manufactures.

**Actionable now:** `volume_spike` is *contra-indicative* (−3.03%) and is simultaneously
the highest-volume detector (772 of 1,155 flags in the real sweep). Consistent with IEX
routing noise rather than real interest. Consider dropping or inverting it before any
paid run, then re-test after the data repairs below.

---

## Four blockers, three of them the same class of bug that killed the last campaign

### B1 — Every tier-2 bar is unadjusted (BLOCKER)
`alpaca_data.py:107` requests `Adjustment.RAW`; `:122` sets `adj_close=None` ("Tiingo owns
adjustment") — but Tiingo never ran on the wide universe, and `corporate_actions` has **0
rows**. Measured: **1,146 of 11,247 tier-2 symbols (10.2%)** show a single-day move below
−30% or above +45%; **190 of those are near-exact 2:1/3:1/4:1/1:2/1:10 ratios across 172
symbols**. Those are splits, indistinguishable from crashes — and they land
*disproportionately in the flagged set*, because that is what the detectors look for.
**Fix:** `Adjustment.ALL` + populate `adj_close`, re-backfill (free, 1–2 h).

### B2 — A 12× volume scale break runs through the panel
Tier-1 carries Tiingo consolidated volume; tier-2 carries Alpaca IEX-only. Measured on
comparable mega-caps: 17.4M shares/day (Tiingo) vs 1.46M (Alpaca IEX). That discontinuity
runs straight through `dollar_vol_20d`, `dollar_vol_z_60d`, `amihud_20d`,
`volume_trend_20_60` — **the identical failure mode as the crypto-calendar bug**: a data-source
discontinuity silently corrupting a cross-sectional ranking.
**Fix:** single-source the volume (pull tier-1 on Alpaca too; prefer Alpaca volume +
Tiingo adjusted close).

### B3 — Tier-2 membership is dated 2026-08-17
`gw.universe((2,))` at `as_of=2026-03-03` returns **zero** names. But a blanket backdate
injects look-ahead (a 2025-06 IPO must not be in the 2024 surveillance universe).
**Fix:** `valid_from = min(bars_1d.ts)` per symbol — a genuine listing-date proxy.

### B4 — Replay has no discovery step at all
Confirmed by exhaustive grep: `src/dynamo/replay/` contains **zero** references to
`detector`, `surveillance`, `tier2`, `triage`, `promot`, `discovery`.
`ReplayConfig.universe_tiers` defaults to `(1,3)`; `gateway.universe()` merely *reads*
whatever membership already exists — so live-written tier-3 rows would be inherited as if
known then. The promotion decision is never re-simulated.
**Fix:** new `src/dynamo/replay/discovery.py` (below).

---

---

## Repair status — measured 2026-08-19, after the fixes

| Blocker | State | Evidence |
|---|---|---|
| B1 adjustment | **mostly fixed, residual named** | see below |
| B2 volume scale | **fixed** | canonical view single-sources volume |
| B3 membership dates | **fixed** | 11,251 rows backdated, idempotent |
| B4 discovery in replay | **not started** | still the core deliverable |
| corporate_actions | **fixed** | 104,835 rows |

**B1.** Alpaca daily bars are now requested `Adjustment.SPLIT` and the whole
wide universe was re-backfilled in place (6,080,787 rows, 11,322 symbols,
763 s). Split artifacts, same symbol set, before vs after:

| Metric | RAW (before) | SPLIT-adjusted (after) |
|---|---|---|
| Extreme moves on a **recorded split ex-date** | **1,166** | **257** (−78%) |
| …within ±1 session | 1,307 | 408 |
| …distinct symbols | 876 | 242 |
| Near-exact split-ratio events (whole panel) | 219 / 213 syms | 180 / 172 syms |
| …of which land on a real split | 76 | **38** |
| Symbols fixed / regressed | — | 298 fixed, 3 regressed |

Two honest caveats. (a) **The ratio test alone is mostly noise.** Of 180
ratio-matched candidates only 38 are real splits — a −50% crash sits within 1%
of a 2:1 ratio, so the `0.5`/`2.0` buckets are dominated by genuine price
moves. Only the `corporate_actions` cross-check separates them; that is why
B1 and the corporate-actions load belong together. (b) **The residual 257 is
Alpaca's own split-coverage gap, not a code bug** — for KIDZ, IXHL and HIMZ
the RAW, `SPLIT` and `ALL` series are byte-identical, so the feed has no
split on record. `Adjustment.ALL` was measured against `SPLIT` on the 2,397
split-carrying symbols and scores identically (336 / 526 either way), so
`SPLIT` stands. Only **10** of the 264 affected names clear the liquidity
screen and only **4** move inside the campaign window.

**B3.** Tier-2 `valid_from` distribution after backdating (11,455 rows):
2018: 4, 2024: 8,642, 2025: 1,449, 2026: 1,360. Visible membership goes from
**zero at every date before 2026-08-17** to **7,727 at 2024-01-02** and
**10,374 at 2026-03-03**. The 55 Tier-1 CORE rows are unchanged.

**Preflight now refuses** a run whose universe still carries unadjusted
actions (`bar_adjustment` check). Measured: tiers (1,3) 55 names → OK; tiers
(1,2) 10,429 names → ERROR naming 264 symbols.

---

## Compute — measured, and it inverts the intuition

| Operation | Measured | Over 115 days |
|---|---|---|
| Load 15 months of the full wide panel (3.09M rows) | 9.0 s once | 9 s |
| `run_detectors`, 10,412 symbols | 7.59 s/day | ~15 min |
| `run_detectors`, **1,248-name filtered panel** | **0.93 s/day** | **1.8 min** |
| `build_features` | ≈ 4 s + 0.135 s·n | — |
| `build_features` at 1,269 symbols | ≈ **175 s/day** | **≈ 5.6 h** |
| `build_features` at 11,500 symbols | ≈ 26 min/day | **≈ 50 h** |

**Detectors are cheap; features are not.** This is the whole architectural argument for
the tier design: scan 1,250 names with 1-second detectors, give the 420-day feature
treatment only to the ~60 that promote.

---

## The universe

Screened over 2025-09-01 → 2026-03-02 (strictly before the window, so the screen is PIT-clean):

| Filter | Names |
|---|---|
| `instruments` total | 11,505 |
| Fund-like by name | 6,134 |
| ≥100 bars AND median close ≥ $5 AND median dollar volume ≥ $2M | 1,683 |
| …**operating companies** | **1,259** |

Unfiltered sweeps flag **505–533 symbols/day**; the filtered universe flags **68/day (5.5%)**.
The live triage cap is 60/night — so an unfiltered sweep saturates the gate with microcap
noise before a real name is ever seen. There is **no minimum-price floor anywhere in the
codebase**; sub-$5 names dominate percentage-move detectors and are where unadjusted
reverse splits live.

---

## How discovery participates (the missing layer)

New `src/dynamo/replay/discovery.py`. Per replay day D:

```
1. panel  = SurveillancePanel.slice(as_of=D, lookback_days=420)   # loaded once, PIT-bounded
2. flags  = run_detectors(...)                                     # detectors.py, unforked
3. promoted_D = promotion_gate(flags, cap=60)                      # `code` | `haiku` | `off`
4. universe_D = tier1_at(D) ∪ {promoted within last 30 sessions}
5-7. → quant_scores_for → shortlist → committee → risk → constructor → next-open fill
```

Steps 5–7 are **untouched**; discovery is a pure prepend that changes what `universe_D` is.

**Promotion gate, two variants** (run both — the delta measures what the triage LLM adds):
- `code`: deterministic severity ranking, cap 60/day, 30-day window. $0, hashable.
- `haiku`: the real `_triage_flags` against `claude-haiku-4-5` (cutoff 2025-07-31, passes
  the gate with 7 months' margin). ~7,800 calls ≈ **$10** on the API, $0 on the subscription.

**Firewall (non-negotiable):** replay discovery must **never** write `universe_membership`
or `surveillance_flags` — those are live tables, and a replay write would contaminate
production *and every future replay* through `gateway.universe()`. Promotions persist to a
new run-scoped `replay_flags` table via `ReplayRepository`.

**Point-in-time:** two new gateway methods (`surveillance_universe`, `surveillance_bars`)
built on `_read_df` so the existing runtime tripwire covers them automatically, plus a new
leakage test seeding tier-2 bars/membership/flags dated after `as_of` and asserting all
three are invisible.

---

## Window and models

`model_knowledge_gate` blocks when `start ≤ cutoff + 30d` for any tier model:

| Model | Cutoff | First clean start |
|---|---|---|
| haiku-4-5 (screen/triage) | 2025-07-31 | 2025-08-31 |
| sonnet-5 (analyst) | 2026-01-31 | **2026-03-03** |
| fable-5 | 2026-01-31 | **2026-03-03** |
| opus-5 | 2026-05-31 | 2026-07-01 |

**Window: 2026-03-03 → 2026-08-14 = 115 NYSE sessions.** Arbiter must be **Sonnet-5**
(zero Fable, zero Opus). Do not use `--acknowledge-model-knowledge`.

---

## Cost

| Variant | LLM calls | Wall clock | Metered |
|---|---|---|---|
| Quant + discovery `code` | 0 | ~2–3 h | **$0** |
| Quant ×3 bankrolls + no-discovery control | 0 | ~8–10 h | **$0** |
| Committee, `haiku` discovery, 6 dives/day, arbiter=Sonnet | ≈10,600 | ~8–14 h, resumable | **$0** on subscription |
| Same on the metered API | ≈10,600 | ~4–6 h | ≈**$120–160** |

A committee day at 6 dives = 25 calls, not 17 (that figure assumed 4 dives).

---

## What this run will NOT test (the honest list)

1. **Performance on a realistic universe — survivorship.** Measured: **11,304 of 11,319
   symbols (99.87%) still have bars at the last date; only 15 stop early.** Monthly distinct
   counts rise monotonically 7,856 (2024-01) → 11,304 (2026-08) and **never fall**. A real
   US universe loses 4–6%/yr — **1,000–1,400 names should have vanished.** ~99% of expected
   delistings are absent. Bias is optimistic and *concentrated in the flagged set*, because
   delisted names are disproportionately what detectors flag. **Rule of thumb: a
   flagged-vs-unflagged spread under ~1%/20 sessions is inside the survivorship band and
   proves nothing.**
2. **True liquidity** — IEX is ~2–3% of the tape; every ADV number is on the wrong scale.
   Never quote a capacity or slippage result from this run.
3. **Intraday anything** — no stops, no PDT interaction, no same-day entries.
4. **Options, Kalshi, Polymarket, social** — no historical source for any of them.
5. **Web research** — you cannot replay the internet as of a past date. Any fetch returns
   today's page, including the outcome. This is permanent.
6. **4 of 9 detectors** — `filings`, `insider_tx`, `earnings_events` are 0 rows;
   `news` covers 50 core symbols. **Discovery is tested at ~5/9 strength, and the missing
   four are the fundamental/event detectors.** A weak result falsifies the price/volume half
   of the design, not the design.
7. **Fundamentals for tier-2 names** — 40 of 11,505 have a CIK. Promoted names get
   fundamentals-free dossiers, i.e. exactly the condition that produced the control run's
   bear-leading conviction. **Report core-name and promoted-name verdict distributions
   separately or the aggregate is meaningless.**

---

## Interpretation discipline

1. **Report the discovery A/B, not the absolute return.** `wide-quant` vs
   `core-quant-control` is the only comparison where both arms share the survivorship bias,
   the IEX bias and the missing sleeves. That difference is the closest thing to an unbiased
   estimate the free data permits.
2. Positions opened within ~20 sessions of the window end get synthetic exits (~17% of the window).
3. Never quote capacity, slippage or ADV-participation numbers.

---

## Acceptance criteria for the discovery layer

| Check | Pass |
|---|---|
| Flags/day (filtered universe) | 30–120 |
| Distinct promoted names over 115 days | 250–800 |
| Flagged vs liquidity-matched random, paired | positive AND above the 95th percentile of a shuffled-date null |
| Any single detector | not significantly negative — else drop it |
| Promoted share of deep-dive slots | 30–70% |
| Split artifacts among flagged names | **0** after B1 — hand-audit the top 20 flagged moves |

## Committee targets (vs the measured control run)

| Metric | Control | Target |
|---|---|---|
| Stored `buy` verdicts | 0 of 295 | > 0 |
| Kill-case match failure | 515/516 (99.81%) | ~0% |
| Max bull conviction | 0.58 | clears 0.6 |
| Distinct symbols debated | 20 of 55 | 400+ of ~1,300 |
| Most-debated symbol share | 70% (WMT) | < 10% |
| Day-over-day rank correlation | 0.88 | < 0.5 |

---

## Code to write

| Item | Size |
|---|---|
| `Adjustment.ALL` + `adj_close` (B1) | trivial |
| `backfill corporate-actions` | small |
| `universe-wide --backdate-from-bars` (B3) | small |
| `backfill sectors` (+ `is_etf`) | small |
| Single-source volume rule (B2) | small |
| **`src/dynamo/replay/discovery.py`** — `SurveillancePanel`, `run_discovery`, `promotion_gate` | **medium — core deliverable** |
| Gateway `surveillance_universe()` / `surveillance_bars()` via `_read_df` | small |
| `_per_symbol` hoisted out of the 5 bar detectors (3–4× speedup) | small + golden test |
| `replay_flags` table + migration + repository writer | small |
| `ReplayConfig.discovery_*` + CLI flags | small |
| `tests/unit/replay/test_discovery.py` | medium |
| Replay-side feature cache (**separate** from `PanelFeatureCache` — see its docstring) | small |

Roughly one session for the data repairs, one for `discovery.py` and its tests.
Nothing requires new external data or a paid API.
