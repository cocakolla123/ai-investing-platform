---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 06 — Risk Engine

**The highest-priority component.** Everything else in Dynamo is advisory; the risk engine's
output is a hard constraint. It is deliberately boring: pure, deterministic, property-tested
functions over portfolio state. No LLM participates in any risk decision.

## 1. The veto contract

```
TargetPortfolio = optimize(advisory_inputs) subject to RiskVerdict.constraints
```

- The committee, quant models, and portfolio optimizer **propose**; the risk engine **disposes**.
- `RiskVerdict` is produced independently of any proposal (from market + portfolio state alone)
  and applied as constraints/clamps — the optimizer cannot negotiate with it.
- Every veto/clamp is logged with the rule, inputs, and would-have-been value: risk decisions
  are as explainable as research decisions.
- If the risk engine cannot compute (data gap, stale prices), the answer is **"no new risk"** —
  fail-closed, never fail-open.

Interface (illustrative):

```python
@dataclass(frozen=True)
class RiskVerdict:
    as_of: datetime
    regime: RegimeState
    gross_exposure_cap: float          # fraction of equity
    net_exposure_cap: float
    per_position_cap: dict[str, float] # symbol → max weight
    sector_caps: dict[str, float]
    new_risk_allowed: bool             # drawdown ladder / kill-switch
    cash_floor: float
    stop_updates: list[StopOrderSpec]  # dynamic + trailing stop levels
    vetoes: list[Veto]                 # explicit rejections with reasons
    metrics: RiskMetrics               # VaR/CVaR/vol/correlation snapshot for reporting
```

## 2. Limit taxonomy (defaults; all config per risk bucket)

Risk buckets Low / Medium / High are **parameter presets, not different code paths**. Values
below are the Medium preset; Low ≈ 0.6×, High ≈ 1.4× (caps never exceed hard bounds).

| Limit | Medium default | Hard bound |
|---|---|---|
| Per-position weight | 8% | 15% |
| Sector weight (GICS) | 25% | 35% |
| Gross exposure | 100% (no leverage) | 100% — leverage is out of scope |
| Single-name tactical sleeve position | 4% | 8% |
| Cash floor | 5% | ≥ 0% |
| Correlation-cluster weight (see §7) | 30% | 40% |
| Illiquid position (est. days-to-exit > 3 at 10% ADV) | 0 for new buys | — |
| Portfolio 95% 1-day VaR | 2.5% of equity | 4% |
| Annualized target volatility | 12% | 18% |

## 3. Drawdown ladder (capital preservation backbone)

Measured on portfolio equity vs trailing 12-month high-water mark, evaluated daily:

| Drawdown | Response (cumulative) |
|---|---|
| −5% | New-position risk budget halved; tactical sleeve max halved |
| −8% | No new tactical positions; long-term adds require arbiter conviction ≥ 0.75 **and** human ack |
| −10% | **No new positions.** Only risk-reducing trades allowed |
| −15% | De-risk to defensive floor: reduce gross to ≤ 50%, raise cash; human review required to resume |
| −20% | Full stop: liquidate tactical sleeve, freeze platform trading; post-mortem before restart |

Ladder state changes fire alerts and are sticky: recovery requires equity back above the
trigger −1% band for 5 consecutive sessions (hysteresis prevents flapping).

## 4. Position sizing

Sizing is computed by the risk engine, taking advisory conviction only as a bounded tilt:

1. **Base size** — volatility-normalized risk parity within sleeve:
   `w_i ∝ (target_vol_contribution / σ_i)`, σ from EWMA(60d) of adjusted returns.
2. **Conviction tilt** — multiply by `0.5 + committee_conviction`, clamped to [0.5, 1.5].
   Until calibration is proven (M7), conviction may tilt, never drive.
3. **Fractional Kelly cap** — compute Kelly fraction from quant expected return/variance,
   apply **0.25× Kelly** as a *ceiling*, not a target. Negative-edge Kelly ⇒ size 0.
4. **Clamp cascade** — per-position cap → sector cap → correlation-cluster cap → liquidity cap
   (position ≤ 10% of 20d median dollar volume × 3 days) → gross/cash constraints.
5. **Round & floor** — positions below 0.5% weight are not opened (noise trades).

Property-tested invariants (hypothesis): no input can produce a weight exceeding any cap; total
weights + cash = 1; Kelly ceiling respected; monotonicity (higher σ never yields larger base
size); zero/NaN inputs ⇒ size 0, never an exception escaping.

## 5. Stops and exits

- **Initial stop:** ATR-based (default 2.5× ATR(14)) below entry for tactical; long-term sleeve
  uses thesis-invalidation levels from `CommitteeVerdict.exit_conditions` plus a disaster stop
  (−25%).
- **Trailing stop:** ratchets with price at same ATR multiple once position is > 1 ATR in profit.
- **Time stop:** tactical positions auto-flagged for exit review at strategy-configured horizon.
- **Thesis stop:** committee `exit_conditions` are stored as *checkable predicates* where
  possible (e.g., "gross margin < 30% next report", "loses key customer"); the learning loop
  audits whether exits honored them.
- Stops are recomputed daily and on 15-minute intraday checks for held positions; execution
  layer (`09`) maintains them as broker stop orders where supported, else synthetic.

## 6. VaR / CVaR / stress / Monte Carlo

- **Historical VaR/CVaR:** 95% and 99% 1-day, from 2y of portfolio-weighted historical returns
  (full revaluation, not variance-covariance normality).
- **Monte Carlo:** weekly, 10k paths, Student-t marginals fitted per position with EWMA vol +
  Gaussian copula from the correlation matrix (§7); reports P(> x% loss over 20d), expected
  shortfall paths, and the drawdown-ladder trigger probabilities.
- **Deterministic stress suite:** replay worst historical windows across current holdings
  (2008 Q4, 2010 flash crash week, 2011 US downgrade, 2015 Aug, 2018 Q4, 2020 Mar, 2022 rate
  shock) plus synthetic shocks (−10% market gap with correlation → 1; +200 bps rate shock with
  duration proxies; single-position −40% overnight). Results in equity terms with ladder
  implications; any scenario breaching −15% floor → sizing warning + dashboard flag.
- VaR model quality is itself monitored: daily VaR exceptions counted (Kupiec-style traffic
  light); too many exceptions → risk model marked degraded → caps tighten automatically by 25%.

## 7. Correlation & concentration

- EWMA correlation matrix (120d halflife) over Tier-1 + holdings; shrinkage (Ledoit-Wolf) for
  stability.
- **Correlation clusters** via hierarchical clustering (threshold ρ ≈ 0.7); cluster weights are
  a first-class limit — ten "different" AI names are one bet and get capped as one.
- Factor proxies (beta to SPY, rate sensitivity via TLT beta, sector tilts) reported; net beta
  target band per bucket (Medium: 0.4–0.9).

## 8. Market regime detection

Regime informs caps and the macro digest; it never generates buy signals.

- Inputs: SPY 50/200d trend state, realized vol percentile (VIX proxy from returns if no VIX
  feed), credit spreads (FRED HY OAS), yield-curve slope, % of Tier-1 above 200d MA (breadth).
- Model: transparent **rule-based composite** scored −2…+2 → {`risk_on`, `neutral`,
  `risk_off`, `crisis`} with hysteresis. (A HMM was considered and rejected for
  explainability + data-scale reasons; revisit criteria in ADR list.)
- Effects (Medium preset): `risk_off` → gross cap 80%, tactical sleeve halved, no new
  promotions traded; `crisis` → new_risk_allowed = false, tighten all stops one notch.
- Backfill 15+ years for eyeball validation against known regimes (M3 acceptance).

## 9. Volatility targeting

Portfolio-level: realized 20d portfolio vol vs bucket target (Medium 12% ann.). Outside the
±25% band → scale exposure toward target (subject to turnover budget in `07`). This is the
primary "how much risk overall" dial and the main mechanism by which "not investing" happens
mechanically: when vol spikes, sizing shrinks toward cash without anyone deciding anything.

## 10. Liquidity

- Entry constraint: position ≤ 3 days × 10% of 20d median dollar volume.
- Portfolio exit test (weekly): estimated days to liquidate 100% at 20% participation; > 5 days
  for > 10% of equity → concentration alert.
- ETF look-through: underlying liquidity assumed adequate for the curated ETF set; single-name
  rules apply to everything else.

## 11. Implementation notes

- `risk/` is pure: no I/O, no clock reads — state in, verdict out; adapters feed it. This makes
  it trivially property-testable and reusable identically in backtest/paper/live (the
  same-code-path requirement).
- Pyright strict; hypothesis suites are part of `make check`; mutation-style tests for the
  clamp cascade (each clamp disabled must fail a test).
- 15-minute intraday risk check (positions marked to Alpaca quotes): stop breaches, gap moves,
  ladder evaluation. Nightly full recompute with EOD data.
- All thresholds live in `RiskConfig` (pydantic), versioned; changes to risk config follow the
  same promotion pipeline as models (`08`) — risk parameters are production logic.

## 12. Open questions (carried to 14-design-review)

1. Should the disaster stop for the long-term sleeve be wider (−30%) to avoid shaking out
   high-conviction compounders in normal drawdowns? → resolve with backtest evidence in M2.
2. Netting of promoted (Tier-3) tactical exposure vs same-sector core holdings — cluster caps
   may suffice; verify in M4 scenario tests.
