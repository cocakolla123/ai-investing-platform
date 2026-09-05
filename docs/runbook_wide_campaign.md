# Runbook — wide, discovery-enabled campaign

**Supersedes** `runbook_backtest_campaign.md` (that one describes the 55-name, no-discovery
campaign whose control run is `replay-e5a49317bbee`). Design and measurements:
`wide_campaign_design.md`. Findings from the control run: `campaign_findings_and_fix_plan.md`.

**What changed since the old runbook**
- Universe: 55 → **11,505 instruments** (6.2M adjusted bars), tier-2 membership dated by each
  symbol's first bar so a 2025 IPO is invisible in 2024.
- Discovery exists in replay (`src/dynamo/replay/discovery.py`) — detectors run per replay
  day, flags promote names into the candidate pool, downstream machinery untouched.
- Arbiter is **Sonnet-5, not Opus-5**. Both Sonnet-5 and Fable-5 cut off 2026-01-31, so the
  clean window starts **2026-03-03** rather than 2026-07-01. Opus-5 (2026-05-31) must not
  arbitrate this window.
- Catalog 51 → 69 features (weekly/multi-timeframe, accumulation, momentum quality, short
  interest, attention).

---

## Window and models

**2026-03-03 → 2026-08-14 = 115 NYSE sessions.**

| Model | Role | Cutoff | Clean from |
|---|---|---|---|
| claude-haiku-4-5 | triage / screen | 2025-07-31 | 2025-08-31 |
| claude-sonnet-5 | analyst **and arbiter** | 2026-01-31 | **2026-03-03** |
| claude-fable-5 | (unused this campaign) | 2026-01-31 | 2026-03-03 |
| claude-opus-5 | **must not arbitrate** | 2026-05-31 | 2026-07-01 |

Never pass `--acknowledge-model-knowledge`. If the gate refuses, the window is wrong.

---

## Prerequisites (all already done — verify, don't redo)

```bash
cd ~/Documents/ai-investing-platform
export DATABASE_URL="$(grep '^DATABASE_URL=' .env | cut -d= -f2-)"
docker compose up -d postgres
uv run alembic upgrade head
uv run pytest -q && uv run pyright && uv run ruff check . && uv run lint-imports
```

Verify the data (expected values as of 2026-08-20):

```sql
SELECT count(*) FROM bars_1d;                                   -- ~6.2M
SELECT count(*) FROM corporate_actions;                         -- ~104,835
SELECT count(*) FROM short_interest;                            -- ~3.9M
SELECT count(cik), count(sector), count(*) FROM instruments;    -- ~6,086 / ~5,300 / 11,505
SELECT count(*) FROM universe_membership WHERE tier=2;          -- ~11,455
SELECT count(*) FROM universe_membership WHERE tier=1;          -- 55 (must be unchanged)
```

---

## Stage A — quant model

The catalog changed (51 → 69 features), so **every previously registered bundle is stale** —
`bundle_from_version` refuses on feature drift, by design. A retrain is mandatory.

```bash
uv run python -m dynamo.cli.quant_train \
  --train-start 2024-06-01 --train-end 2026-02-28 \
  --symbols-file config/training_universe_wide.txt \
  --label campaign-v5 --session-stride 4 \
  --lgbm-max-depth 3 --lgbm-min-data-in-leaf 120 --lgbm-lambda-l2 10 \
  --progress-file /tmp/v5_progress.jsonl
```

**On `--session-stride 4`:** with a 20-session forward label, consecutive sessions share
19/20 of their label window, so adjacent rows are ~95% redundant. Striding cuts panel-build
time ~4× (21.8h → ~5.5h) *and* yields more nearly-independent samples, which makes the fold
statistics honest rather than optimistic. Purge is counted in strided rows, so a stride > 1
over-purges in calendar terms — conservative, never leaky.

**GO/NO-GO #1.** Read the printed scorecard. Required: pooled rank IC > 0, stable across
folds, downside AUC > 0.5, and — the bar v3 and v4 both failed — **it must beat its own
historic-average baseline**, i.e. `rank_ships_ok: true`. If it fails, you may still proceed,
but **only** with `--no-require-shipping-bundle` and every downstream result labelled a
mechanics control, never quoted as performance.

---

## Stage B — verification gates

```bash
# leakage, firewall, network isolation, cutoff gate, discovery PIT
uv run pytest tests/unit/replay/ -q

# against a scratch Postgres (never dynamo_dev)
docker run -d --name dynamo-scratch -e POSTGRES_PASSWORD=scratch -p 5433:5432 postgres:16
export INTEGRATION_DB_URL="postgresql+psycopg://postgres:scratch@localhost:5433/postgres"
DATABASE_URL="$INTEGRATION_DB_URL" uv run alembic upgrade head
uv run pytest tests/integration -q
docker rm -f dynamo-scratch
```

**GO/NO-GO #2:** all green, including the discovery leakage tests (a tier-2 bar, a membership
row and a surveillance flag each dated after `as_of` must all be invisible).

---

## Stage C — free quant campaign, discovery ON, plus the control

The A/B is the point. Both arms share the survivorship bias, the IEX volume bias and the
missing sleeves, so **the difference between them is the only near-unbiased number this
campaign can produce.** The absolute return vs SPY is not.

> **CORRECTED 2026-09-05 — the original treatment arm was vacuous.** It passed
> `--universe-tier 2`, which makes the entire tier-2 universe DIRECTLY tradable, so
> discovery's promotions promoted names into a pool they were already in. Measured on runs
> `replay-699b62040d45` and `replay-54f0e13875fd`: 99.6% of promoted names were already
> being decided before their promotion, and both runs produced byte-identical equity,
> fills and decisions to each other despite a 4.5x difference in promotions. Those two
> runs measure *universe widening* (wide-traded −6.8% vs core, with the NO-GO model),
> NOT discovery. The treatment arm below trades the CORE universe and lets discovery
> promote from tier 2 — the only difference from the control is promotion itself.

```bash
# discovery ON: trade tier 1+3; discovery SCANS tier 2 and promotes into the pool
uv run python -m dynamo.cli.replay run --mode quant \
  --start 2026-03-03 --end 2026-08-14 \
  --universe-tier 1 --universe-tier 3 \
  --discovery code --discovery-min-dollar-volume 2000000 --discovery-min-price 5 \
  --discovery-promotions-per-day 15 --discovery-promotion-days 30 \
  --quant-model-version campaign-v8 --min-feature-coverage 0.35 \
  --initial-equity 100000 --ballast SPY \
  --code-version wide-2026-08b --run-label disc-quant-100k-true

# CONTROL: identical traded universe, discovery OFF
uv run python -m dynamo.cli.replay run --mode quant \
  --start 2026-03-03 --end 2026-08-14 \
  --universe-tier 1 --universe-tier 3 --discovery off \
  --quant-model-version campaign-v8 \
  --initial-equity 100000 --ballast SPY \
  --code-version wide-2026-08b --run-label core-quant-100k-control
```

**GO/NO-GO #3** — acceptance criteria for the discovery layer. Computed by:

```bash
uv run python -m dynamo.cli.replay discovery-report --run-id <wide-quant-100k>
```

| Check | Pass | Where |
|---|---|---|
| Flags/day on the filtered universe | 30–120 (below 30 = gate too tight; above 200 = floor too loose) | `flags:` line |
| Distinct promoted names over 115 days | 250–800 | `distinct promoted:` |
| Promoted share of deep-dive slots | 30–70% (0% = promotion never reaches the shortlist; ~100% = core is starved) | `promoted share:` |
| Any single detector | not `CONTRA-INDICATIVE` — else drop it and re-run | per-detector table |
| Split artifacts among flagged names | 0 — hand-audit the top 20 flagged moves | manual |

Read the per-detector table the way it is built, not as a league table. The `spread` column
is flagged-minus-unflagged **on the same session**, so it is already free of "the market went
up"; `eff n` is days ÷ horizon because overlapping 20-session windows make 115 days about 6
independent observations; and `inconclusive` means the spread is inside the ±1% survivorship
band or fails |t| > 2 on that effective sample. Most detectors will read inconclusive on 115
days, and that is the correct answer, not a tooling failure.

**Known from the pilot:** `volume_spike` measured **−3.03%** forward 20-day excess and is the
loudest detector. The a-priori weights in `research/flag_ranking.py` already rank it last;
it is deliberately not inverted or zeroed on pilot evidence alone (effective n ≈ 4–5). This
run is what settles it — if `discovery-report` returns `CONTRA-INDICATIVE` for it here, drop
it from the weights and re-run Stage C.

---

## Stage D — committee (subscription-metered, zero Fable)

```bash
uv run python -m dynamo.cli.replay estimate \
  --start 2026-03-03 --end 2026-08-14 --deep-dives-per-day 6 --backend claude_code
```
Confirm the gate verdict is OK with the arbiter on Sonnet-5 before spending anything.

```bash
export CLAUDE_CODE_MODEL_ARBITER=claude-sonnet-5     # zero Fable, zero Opus
uv run python -m dynamo.cli.replay run --mode committee \
  --start 2026-03-03 --end 2026-08-14 \
  --universe-tier 1 --universe-tier 2 --universe-tier 3 \
  --discovery haiku --discovery-min-dollar-volume 2000000 --discovery-min-price 5 \
  --discovery-promotions-per-day 60 --discovery-promotion-days 30 \
  --deep-dives-per-day 6 --candidate-percentile 0.6 --crypto-slots 1 \
  --cooldown-sessions 5 --rotation-slots 1 --min-feature-coverage 0.35 \
  --quant-model-version campaign-v5 \
  --backend claude_code --max-usd 60 \
  --code-version wide-2026-09 --run-label wide-committee-mar-aug-2026
```

~10,600 calls (≈7,800 Haiku triage, ≈2,300 Sonnet analyst, ≈690 Sonnet arbiter, 115 macro
digests). $0 metered on the subscription; ~$120–160 on the API backend. Resumable — re-run
the identical command and days with persisted verdicts are skipped.

**Checkpoint after ~2 simulated weeks:** `replay show --run-id <id>`. Are promoted names
plausible? Are dossiers grounded? Is `buy` reachable (it was mechanically impossible in the
control run — 515/516 kill-case match failures, 0 buys in 295 verdicts)? Is bull conviction
clearing 0.6 (it never did in 74 control days, max 0.58)? If the slates are garbage, Ctrl-C.

---

## Stage E — portfolio passes, reports, audit

```bash
for EQ in 1000 10000 100000; do
  uv run python -m dynamo.cli.replay run --mode portfolio \
    --source-run-id <committee-run-id> --start 2026-03-03 --end 2026-08-14 \
    --initial-equity $EQ --ballast SPY \
    --code-version wide-2026-09 --run-label wide-committee-portfolio-$EQ
done

uv run python -m dynamo.cli.replay report --run-id <each>
uv run python -m dynamo.cli.replay compare --run-ids <wide-quant-100k> <core-quant-100k-control>
uv run python -m dynamo.cli.replay audit --run-id <committee-run-id>
```

---

## Committee targets vs the measured control run

| Metric | Control (`replay-e5a49317bbee`) | Target |
|---|---|---|
| Stored `buy` verdicts | **0 of 295** | > 0 |
| Kill-case match failure | 515/516 (99.81%) | ~0% |
| Max bull conviction | 0.58, never cleared 0.6 | clears 0.6 |
| Bear:bull high-severity kill cases | 516 : 117 (4.4×) | < 2× |
| Distinct symbols debated | 20 of 55 | 400+ of ~1,300 |
| Most-debated symbol's share | 70% (WMT, 52/74 days) | < 10% |
| Day-over-day rank correlation | 0.88 (0.55 at lag 40) | < 0.5 |
| Conviction distinct values | 8, 98.3% in {0.55,0.58,0.60,0.62} | broad |

---

## Interpretation discipline — read before quoting any number

1. **Report the discovery A/B, not the absolute return.** Only `wide-quant` vs
   `core-quant-control` shares all biases across both arms.
2. **Survivorship is measured and severe.** 99.87% of symbols survive to the last bar; ~99%
   of expected 2024-2026 delistings are absent. Bias is optimistic and *concentrated in the
   flagged set*, because delisted names are disproportionately what detectors flag.
   **A flagged-vs-unflagged spread under ~1%/20 sessions is inside the bias band and proves
   nothing.** Tiingo's delisted history is what would fix this.
3. **Never quote capacity, slippage or ADV-participation numbers.** Volume is IEX-only,
   understated by a factor that itself drifts (3.2% of consolidated over full history, 8.1%
   over the last 60 sessions).
4. **Discovery runs at ~5/9 detector strength.** `filings`, `insider_tx`, `earnings_events`
   coverage and wide news are still thin, so `filing_event`, `insider_cluster`,
   `earnings_surprise` and `news_burst` are dark or partial. A weak result falsifies the
   price/volume half of the design, not the design.
5. **Promoted tier-2 names get thinner dossiers than core names** (fundamentals need a CIK;
   ~6,086 of 11,505 have one). Report core-name and promoted-name verdict distributions
   **separately** or the aggregate is meaningless.
6. Positions opened within ~20 sessions of the window end get synthetic exits (~17%).
