# Re-runs after the price and replay fixes (2026-10-04)

Results made before these fixes were re-run on code that includes all of
them. The user asked for this to update their summary bullets.

| fix | commit | what was wrong |
|---|---|---|
| bar scale | 1c6c04d | Tiingo open/high/low stayed raw while close was split-adjusted. Pre-split NFLX filled at 971 against a mark of 97. |
| live-parity volume | 8cc854a | Replay read raw feed dollar volume, so BTC looked like it traded ~$4k/day. Every crypto buy was liquidity-vetoed. |
| debated names tradable | 8cc854a | Portfolio replays dropped every verdict on a discovery-promoted name. |
| ballast-exempt turnover | 5dd26dd | Daily SPY top-ups spent the monthly turnover budget and deferred committee buys. |
| next-open fills | 5dd26dd | A debated name that left the universe had no bars the next day, so its order never filled. |

The committee campaign's portfolio traded **only SPY and MCD** before these
fixes. 64 of its 72 buys never executed. It now trades 12 symbols.

Quant-only and portfolio replays only: $0, no model calls. Runs are on
`--code-version rerun-5dd26dd` unless noted.

## Before → after

| result | before | after |
|---|---|---|
| **v10 quant model** (Phase 5 LightGBM row; panel rebuilt on fixed bars) | IC +0.0295, fold t 1.58, win share 0.53 → **NO-GO** | IC **+0.0351**, fold t **2.15**, win share 0.53 → **still NO-GO**: loses to the historic-average baseline (+0.0401) and wins only 53% of folds |
| **Phase 5: momentum alone** | IC +0.0399, t 3.14 | unchanged (close-only feature) |
| **Phase 5: LightGBM vs any baseline** | beats none | **beats none** (best paired t +1.81, vs linear) |
| **Phase 5: feature groups that matter** | none (max \|t\| 1.79) | **none** (no \|t\| ≥ 2) |
| **Committee campaign, $100k** | +12.80%, Sharpe 2.08, DD −7.19%, −1.92% vs SPY, alpha −1.0%/yr (t −0.41), 2 symbols | **+11.12%**, Sharpe **1.87**, DD −7.07%, **−3.60%** vs SPY, alpha **−2.35%/yr (t −0.37)**, 12 symbols, costs $378 |
| Committee campaign, $1k / $10k | +12.82% / +12.81% | **+11.23% / +11.20%** |
| **Core control** (v8 only, 55 names) | +14.57%, Sharpe 2.33, −0.15% vs SPY, alpha +3.5%/yr (t +0.58) | **+8.02%**, Sharpe **1.42**, **−6.70%** vs SPY, alpha **−8.2%/yr (t −1.42)** |
| **True discovery** (v8, core + promoted) | +10.65%, Sharpe 1.50, −4.07% vs SPY | **+13.45%**, Sharpe **1.83**, **−1.27%** vs SPY, alpha −2.3%/yr (t −0.23) |
| Discovery − core control (paired) | −7.3%/yr (t −0.55): "discovery subtracted" | **+11.3%/yr (t +0.96)**: sign flipped, **not significant** |
| Committee − core control (paired) | — | +6.3%/yr (t +0.83), not significant |
| **Wide-traded** (v8, ~10.4k names) | +7.80%, −6.92% vs SPY | **not re-run** (user decision: ~14h, and a no-edge model can't answer "does going wide help?"). Treat the old number as pre-fix. |

SPY returned about +14.7% over the window (2026-03-03 → 2026-08-14).

## What still holds, and what changed

- **Still true:** no strategy shows skill distinguishable from zero. Every
  alpha t-stat is between −1.42 and +0.58. No model passes its pre-registered
  gate.
- **Changed:** the old ranking of the replays was an artifact of the bugs.
  - The core control's +14.57% shrank to +8.02%. Its inputs changed: the
    bar-scale fix, and the ASC 606 revenue-tag fix (732abaf), which changes
    v8's revenue features.
  - With discovery and crypto actually trading, the discovery arm went from
    worst to best of the three. None of these gaps is significant.
- **The committee's headline number was mostly SPY.** It still is, with
  SPY ballast averaging about 77% of the book. The committee holds about 3–4
  positions at a time.

## Run ids

| arm | before | after |
|---|---|---|
| committee $1k / $10k / $100k | replay-aeb6fd0b86ad / replay-29953abd0e28 / replay-c41988c44988 | replay-a1fe88311907 / replay-6176e7439103 / replay-1bccf6b04259 |
| core control | replay-ad0c913a7f4c | replay-aa1194237525 |
| true discovery | replay-0239ea6a1d37 | replay-ea9a27a900dd |
| Phase 5 / v10 | `evals/ml/results_phase5.json` | `evals/ml/results_phase5_barfix.json` |

## Operational note

Postgres (2.5 GiB container) OOM-restarted twice on 2026-10-04. Both times,
a discovery-mode preflight that queries ~11k symbols overlapped other heavy
jobs. It recovered on its own both times (WAL recovery) with no data loss.
Heavy replays now run strictly one at a time.
