---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 05 — Strategies & Scoring

This doc defines how research outputs become a **ranked watchlist and sleeve-level candidate
sets**: the two portfolio sleeves, the long-term quality score, the tactical strategy
registry, and the deterministic formula that blends `CommitteeVerdict`, `QuantScore`, and
`QualityScore`. Everything here is advisory; the risk engine (`06`) has final authority over
whether and how much of any of it becomes a position.

## 1. Two sleeves

Per `00 §3`: a **long-term quality sleeve** (months–years) and a **tactical sleeve**
(days–weeks). The split is capital allocation, not code duplication — both sleeves flow
through the same portfolio construction and the same risk clamps.

| Parameter | Default | Bounds | Where |
|---|---|---|---|
| Long-term sleeve allocation | 70% of invested equity | 50–90% | `SleeveConfig` (pydantic, in config — **never code**) |
| Tactical sleeve allocation | 30% | 10–50% (complement) | same |
| Rebalance toward split | Monthly, ±5pp drift band | — | same |

Changes to `SleeveConfig` are risk-relevant production logic and follow the promotion
pipeline (`06 §11`, `08`) — the allocation dial is not a live tuning knob. Sleeve-specific
risk parameters (tactical per-position cap 4%, etc.) remain in `RiskConfig` (`06 §2`);
`SleeveConfig` holds only allocation and candidate-selection settings.

## 2. Long-term quality score

A composite **`QualityScore` ∈ 0–100** per Tier-1 name, built from seven sub-scores, each
0–100. Sub-scores are **sector-relative percentiles** (GICS sector, within point-in-time
Tier 1 membership) so "good" always means "good versus comparable businesses now" — same
normalization philosophy as the quant features (`04 §3`).

### 2.1 Sub-scores and weights

| Sub-score | Weight | Quantitative inputs (point-in-time XBRL tags, `filed_at` ≤ as_of per `02 §3`) | Qualitative inputs |
|---|---|---|---|
| Business quality / moat | 20% | Gross margin stability (σ of `GrossProfit`/`Revenues` over 12 filed quarters, inverted), ROIC persistence (quarters of ROIC above sector median, of trailing 12) | Committee dossier moat assessment: arbiter-extracted moat rating from the persistent dossier store (`03 §1`), mapped {none:0, narrow:50, wide:100}, evidence-referenced or discarded |
| Revenue growth | 15% | `Revenues` YoY and 3y CAGR from filed quarters; acceleration term (Δ of YoY) | — |
| Profitability | 15% | Operating margin (`OperatingIncomeLoss`/`Revenues`), net margin, ROIC = NOPAT / (`StockholdersEquity` + `LongTermDebt` − `CashAndCashEquivalentsAtCarryingValue`) | — |
| Cash flow | 15% | FCF conversion (FCF / `NetIncomeLoss`), FCF margin, FCF yield (FCF / EV); FCF = `NetCashProvidedByUsedInOperatingActivities` − `PaymentsToAcquirePropertyPlantAndEquipment` | — |
| Valuation | 15% | EV/EBIT and FCF yield vs **own 3y history** (50%) and vs **sector percentile** (50%) — cheap vs itself and vs peers both count | — |
| Balance sheet | 10% | Net debt / EBITDA, interest coverage (EBIT / `InterestExpense`); net cash ⇒ scores 100 | — |
| Management quality | 10% | Capital-allocation record: buybacks vs valuation history, dilution rate (share count trend), ROIC trend across capex cycles; insider ownership + net insider buying from Form 4 (`insider_tx`, shares with the `insider_cluster` detector inputs, `02 §2`) | Committee dossier read on capital-allocation discipline, evidence-referenced |

`QualityScore = Σ weight_i × subscore_i`, weights in `QualityConfig` (config, promotion-gated
like everything else). Sub-scores with insufficient data (< 8 filed quarters, e.g. recent
IPOs) are marked `null` and the remaining weights renormalize; a name with > 2 null
sub-scores gets no composite and is ineligible for the long-term sleeve — flagged, never
silently defaulted.

### 2.2 Storage and cadence

```sql
CREATE TABLE quality_scores (
    symbol       text NOT NULL,
    as_of        date NOT NULL,
    composite    double precision,            -- null if ineligible
    subscores    jsonb NOT NULL,              -- name → {value, inputs_ref, null_reason?}
    config_hash  text NOT NULL,               -- QualityConfig version pin
    PRIMARY KEY (symbol, as_of)
);
```

- **On new filings:** the EDGAR poller (`02 §4`) triggers recompute for the affected symbol
  the same night — scores update continuously as new information lands, not on a calendar.
- **Quarterly full recompute:** entire Tier 1, catching sector-percentile drift (a name's
  score can move because *peers* reported).
- Dossier-derived qualitative inputs refresh whenever the committee produces a new dossier
  for the name; between deep-dives the last evidence-referenced read persists with its
  `as_of` shown. Qualitative staleness > 90d renders a staleness badge on the dashboard.

The composite's history per symbol is charted on the dashboard; a quality score falling
through a configurable floor (default: composite < 40 for 2 consecutive quarters) emits a
review flag for held long-term positions and feeds the Tier-1 demotion path (`02 §2`).

## 3. Tactical strategies — the `Strategy` registry

Tactical strategies are **parameterized rule objects, not code forks**. Each is an entry in a
`Strategy` registry (`research/strategies/registry.py`) declaring:

```python
class Strategy(BaseModel):
    strategy_id: str                     # e.g. "momentum-rs-v1"
    universe_filter: UniverseFilter      # tiers, liquidity floor, exclusions
    entry_rules: list[Rule]              # typed predicates over features/detector flags
    exit_rules: list[Rule]
    holding_period_days: int             # time-stop default (06 §5)
    risk_params: StrategyRiskParams      # stop multiple, max concurrent signals, per-signal size hint
    status: Literal["candidate","paper","production","retired"]
```

Rules evaluate over the nightly feature frame (`04 §2`) and `surveillance_flags` (`02 §2`) —
strategies add **no new data paths**. Signal emission is deterministic: same inputs, same
`Signal` rows. Each strategy declares its own stop and horizon defaults; **all are
subordinate to the risk engine** — a strategy's stop can be tighter than the ATR default
(`06 §5`), never wider, and its size hint is advisory input to the sizing cascade (`06 §4`).

### 3.1 Initial strategy set (production candidates for M4)

| Strategy | Universe filter | Entry rules | Exit rules | Horizon | Stop default |
|---|---|---|---|---|---|
| **`momentum-rs-v1`** | Tier 1; dollar-vol ≥ $10M | 12-1 momentum in top quintile **and** 20d RS vs sector ETF positive **and** price > 50d MA | RS turns negative for 5 sessions, or momentum exits top two quintiles | 40 trading days | 2.5× ATR(14) trailing |
| **`breakout-v1`** | Tier 1 + promoted Tier 3 | New 52-week high with volume ≥ 2× 60d median (consumes `range_break` + `volume_spike` detector flags, `02 §2`); base of ≥ 30 sessions below prior high | Close below breakout level, or trailing stop | 30 trading days | 2.0× ATR(14) below breakout level |
| **`meanrev-v1`** | Tier 1; **uptrend only** (price > 200d MA, 12-1 momentum positive) | RSI(14) < 30 **and** 5d return z-score < −2 (bounds in config) | RSI > 55, or z-score > 0, or 10-session time stop | 10 trading days | 1.5× ATR(14), no trailing |
| **`earnings-react-v1`** | Tier 1 + promoted; report within last 3 sessions | Post-earnings-announcement drift: EPS surprise > +5% **and** revenue beat **and** positive guidance flag, from `earnings_events` (`02 §3`) + the earnings-call reader's structured output (`03 §1`); day-1 gap held (no full fade) | Drift horizon lapse, or close below post-earnings low | 20 trading days | Post-earnings low − 0.5× ATR(14) |

All thresholds shown are defaults in each strategy's parameter block — config, not code.
Overlapping signals on one symbol are deduplicated: the highest-blend-score (§4) signal wins;
the others are recorded as suppressed with reasons.

### 3.2 Rule representation

`Rule` is a small typed predicate language, not arbitrary Python — declarative enough to
render on the dashboard and diff in PRs:

```python
class Rule(BaseModel):
    feature: str                          # name from features/catalog.py or detector name
    op: Literal["gt","lt","gte","lte","eq","in_top_quantile","in_bottom_quantile","crossed_above","crossed_below"]
    value: float | str
    window_sessions: int = 1              # rule must hold for N consecutive sessions
```

Entry fires when **all** entry rules hold; exit fires when **any** exit rule holds (or the
risk engine's stop fires first — `06` always wins). This keeps every strategy explainable:
the dashboard renders each live signal's rule evaluations with the underlying feature
values, the same evidence standard applied to committee claims (`00 §8`).

### 3.3 Signal output

Strategies emit `Signal` rows (domain entity, `01 §2`):

```sql
CREATE TABLE signals (
    id              bigserial PRIMARY KEY,
    strategy_id     text NOT NULL,
    symbol          text NOT NULL,
    as_of           date NOT NULL,
    direction       text NOT NULL,        -- long only in scope (00 §3)
    entry_ref_price double precision,
    stop_hint       double precision,     -- strategy default; risk engine may tighten
    horizon_days    integer NOT NULL,
    strength        double precision,     -- 0-1
    rule_snapshot   jsonb NOT NULL,       -- evaluated rules + feature values at emission
    status          text NOT NULL,        -- active | suppressed | expired | consumed
    UNIQUE (strategy_id, symbol, as_of)
);
```

Signals are inputs to the tactical candidate set (§4.3), never orders.

## 4. Combining committee, quant, and quality into the watchlist

### 4.1 The deterministic ranking formula

Nightly, after committee batches land (`02 §4`), the watchlist assembler computes a **blend
score ∈ 0–100** per eligible symbol:

```
blend = 100 × ( w_c × committee_component
              + w_q × quant_component
              + w_s × quality_component )        # weights in BlendConfig, Σ = 1
```

| Component | Definition | Default weight |
|---|---|---|
| `committee_component` | Arbiter action mapped {buy:1.0, add:0.8, hold:0.5, trim/sell:0.1, avoid/no_action:0.0} × `conviction`, from the freshest `CommitteeVerdict`; decayed ×0.9 per week of verdict age past 7d | 0.40 |
| `quant_component` | Percentile of `expected_excess_return` in the night's cross-section × `confidence`, minus a penalty `0.5 × downside_prob` floor-clamped at 0 — all from `QuantScore` (`04 §8`) | 0.35 |
| `quality_component` | `QualityScore`/100 (long-term list); for the tactical list this weight redistributes to the other two (quality is slow-moving context, not a timing signal) | 0.25 |

Same formula, two parameter sets (`BlendConfig.long_term`, `BlendConfig.tactical`). The
formula is pure and versioned; every watchlist row stores `blend_config_hash` and the three
component values, so any rank is reproducible and explainable to the same standard as a risk
veto (`06 §1`).

### 4.2 Disagreement handling

- **`quant_alignment = "conflict"`** on the verdict (`03 §2`) **caps the blend rank**: the
  symbol cannot rank above the 40th percentile of the night's list regardless of score, and
  the row carries a visible `conflict` badge. Humans-and-models disagreeing is signal to
  investigate, not to average away.
- `disagreement_score > 0.6` already downgrades verdicts in the committee assembler
  (`03 §4`); the watchlist additionally renders the dissent count on every row.
- A symbol with a committee verdict but **no fresh QuantScore** (quality-quarantined, `02 §6`)
  gets `quant_component = 0` and a `no-quant` badge — never an imputed score.

### 4.3 What each sleeve consumes

| Sleeve | Candidate set | Selection logic |
|---|---|---|
| **Long-term** | Symbols with `QualityScore ≥ 60` **and** committee action ∈ {buy, add} **and** conviction ≥ 0.6, ranked by long-term blend | Top-N (default 25) become the long-term candidate list for portfolio construction (`07`); position changes only on verdict/quality changes — low turnover by construction |
| **Tactical** | Live `Signal` rows joined to the tactical blend score | Signals ranked by `strength × tactical_blend`; top-K (default 10 concurrent across strategies, per `StrategyRiskParams`) proposed; each carries its strategy's stop/horizon into the risk engine's sizing pass |

The tactical sleeve is signal-driven (a strategy must fire); the long-term sleeve is
state-driven (quality + conviction persist). Both lists go to portfolio construction as
proposals — the risk engine's clamp cascade and drawdown ladder (`06 §3–4`) decide what, if
anything, actually changes. "Nothing qualifies tonight" is a normal, logged outcome (`00 §1`).

### 4.4 Watchlist storage and rendering

```sql
CREATE TABLE watchlist (
    as_of                date NOT NULL,
    sleeve               text NOT NULL,       -- long_term | tactical
    symbol               text NOT NULL,
    rank                 integer NOT NULL,
    blend_score          double precision NOT NULL,
    committee_component  double precision,
    quant_component      double precision,
    quality_component    double precision,
    badges               text[] NOT NULL,     -- conflict | no-quant | stale-verdict | promoted
    blend_config_hash    text NOT NULL,
    verdict_ref          text,                -- CommitteeVerdict id used
    signal_ref           bigint,              -- signals.id (tactical rows)
    PRIMARY KEY (as_of, sleeve, symbol)
);
```

The nightly report renders both lists with component breakdowns, badges, dissent counts, and
links to the underlying dossier, verdict, quant score, and signal rows — every rank is
traceable to evidence in ≤ 2 clicks (the traceability metric in `00 §4` applies here too).
Watchlist rows are immutable once written; a rerun writes a new `as_of` snapshot, preserving
the exact inputs portfolio construction saw (`DecisionContext` discipline, `01 §2`).

## 5. Sleeve interaction rules

- **Same symbol in both sleeves: allowed.** A long-term holding can also carry a tactical
  breakout position. The sleeves express different theses on different horizons with
  different exits, and merging them would blur exit discipline.
- **But caps aggregate across sleeves.** Per-position, sector, and correlation-cluster caps
  (`06 §2, §7`) apply to the **combined** exposure per symbol — two sleeve positions in one
  name are one name to the risk engine, exactly as ten correlated names are one bet
  (`06 §7`). The tactical single-name cap (4%) additionally applies to the tactical portion
  alone.
- Exits are per-sleeve: a tactical stop-out never touches the long-term position, and a
  long-term thesis-stop exit (`06 §5`) does not auto-close a live tactical signal — though
  the coincidence is flagged for the journal (`10`).
- Sleeve books are tagged per lot in the portfolio store (`07`), so attribution, turnover,
  and per-sleeve outcome metrics (`00 §4`) stay clean.

## 6. Strategy and scoring lifecycle

| Stage | Mechanism |
|---|---|
| **Entry** | New strategies and any change to `BlendConfig`/`QualityConfig` weights enter as `candidate` via the promotion pipeline (`08`): backtest with purged walk-forward evidence (`04 §5` harness), then `paper` shadow period, then human-approved `production`. The registry `status` field is written only by the promotion tool — same contract as `model_versions` (`04 §7`) |
| **Tracking** | Per-strategy performance — hit rate, profit factor, average excess return per signal, slippage vs model, decay of edge — is tracked continuously in the learning loop (`10`), attributed via the per-lot sleeve/strategy tags (§5) |
| **Decay & retirement** | A production strategy whose rolling OOS edge decays below its paper-period baseline (thresholds in `10`) triggers a **retirement recommendation**: an evidence-backed report proposing `retired` status, queued for human approval on the dashboard — mirroring the universe-change queue (`02 §2`). No strategy is ever silently reweighted or auto-killed (`00 §7`: no auto-modifying production logic); retirement of a strategy with open positions lets existing positions run to their exits while blocking new signals |

## 7. Testing

- **Determinism:** golden fixtures per strategy — frozen feature/flag inputs → exact expected
  `Signal` rows; blend formula golden tests with pinned `BlendConfig`.
- **Property tests (hypothesis):** blend score always ∈ [0,100]; conflict cap never exceeded;
  weight renormalization on null sub-scores always sums to 1; no strategy emits a signal for
  a symbol outside its declared `universe_filter`.
- **Point-in-time quality test:** quality sub-scores computed for a historical date must be
  identical whether computed then or replayed today (guards the `filed_at` discipline,
  `02 §3`).
- **Cross-sleeve cap test:** synthetic portfolios with the same symbol in both sleeves must
  never pass risk checks exceeding combined caps (integration test with `06`).

## 8. Open questions (carried to 14-design-review)

1. Should `earnings-react-v1` require the Haiku triage severity signal in addition to the
   numeric surprise, or is the structured earnings-call read (`03 §1`) sufficient gating?
   → decide with M4 paper evidence.
2. Blend weight `w_c = 0.40` presumes committee calibration that is only proven at M7
   (`00 §4`) — should the committee weight start lower and earn its way up via the promotion
   pipeline? Default position: start at 0.40, revisit at first calibration report.
