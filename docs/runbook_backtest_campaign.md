# Backtest Campaign Runbook — 6-Month Quant Campaign + Clean-Window Committee Replay

**Status:** Phase 6 deliverable (final). Phases 0–5 are merged (crypto sleeve, backfill CLI,
preflight, quant training, portfolio mode + realism, anti-leakage, persistence/reporting).
Free backfills in step 2 were **executed 2026-08-17** — results recorded inline.

**Windows (final user decisions):**

| Campaign | Window | Mode | Cost |
|---|---|---|---|
| Quant-only campaign | 2026-02-02 → 2026-07-31 (126 sessions) | quant, 3 bankrolls + optional whole-shares run | $0 |
| Committee replay | **2026-07-01 → latest completed session** (2026-08-14 as of writing) | committee once, then 3 portfolio runs | ~$45–60, capped $80 |

### Why the committee window is July 1, 2026 onward (and NOT Feb–Jul)

The resolved model knowledge cutoffs (recorded in `llm/pricing.py` `ModelSpec.knowledge_cutoff`,
taken from Anthropic's published docs — never guessed) are:

| Model | Training-data cutoff | + 30d safety margin |
|---|---|---|
| claude-haiku-4-5 | 2025-07-31 | 2025-08-30 |
| claude-sonnet-5 | 2026-01-31 | 2026-03-02 |
| claude-opus-5 | 2026-05-31 | **2026-06-30** |

The knowledge-cutoff gate refuses any committee start ≤ 2026-06-30. **July 1, 2026 onward is
the first fully clean window** — no committee model has training-data knowledge of any event
in it.

A contaminated-window (Feb–Jul) committee replay via `--acknowledge-model-knowledge` was
**explicitly considered and rejected**, and no such run is planned. The reason is worse than
"the returns would flatter us": every committee decision produces a resolved
`decision_experience` that feeds prior-case retrieval and (optionally) generational
meta-model training. Verdicts made with hindsight knowledge would resolve as artificially
well-calibrated wins, and those experiences would **mistrain calibration toward
overconfidence** — the system would learn that aggressive conviction "works" because the
model already knew the outcome. A clean small window teaches true lessons; a contaminated
big window teaches false ones.

**Clean-window growth:** the clean window extends by ~21 sessions every month that passes.
Each later re-run of steps 8–10 (new end date → new deterministic run id) buys a longer
committee campaign at ~$1.50–2.00/session. The Feb–Jul mechanics (sizing, costs, ladder,
crypto sleeve, bankroll divergence) are already covered for free by the quant campaign.

---

## 0. Prerequisites

```bash
cd ~/Documents/ai-investing-platform
docker compose up -d postgres          # container: ai-investing-platform-postgres-1
export DATABASE_URL="$(grep '^DATABASE_URL=' .env | cut -d= -f2-)"
uv sync --extra transcripts            # transcripts extra needed for step 2f
uv run alembic upgrade head            # must land on 0017
uv run pytest -q                       # full suite green before anything else
uv run pyright && uv run ruff check . && uv run lint-imports
```

Keys required in `.env`: `FRED_API_KEY`, `FINNHUB_API_KEY`, `EDGAR_USER_AGENT`,
`ALPACA_API_KEY`/`ALPACA_API_SECRET` (free feeds), and — for step 3 only —
`TIINGO_API_KEY` (pending). `ANTHROPIC_API_KEY` is used **only** in step 9.

Token discipline: **every step here is $0 except step 9**, which is user-triggered and
hard-capped.

---

## 1. Universe backdate (free — DONE 2026-08-17)

```bash
uv run python -m dynamo.cli.backfill universe --as-of 2024-12-01
```

Backdates the CORE seed (50 equities/ETFs + 5 crypto pairs) to 2024-12-01 in one
transaction. Idempotent; safe to re-run.

**Executed result:** deleted 50 seed rows, wrote 55 CORE rows; verified 55/55 starters
active at both 2024-12-01 and today.

## 2. Free backfills (no Tiingo, no tokens — DONE 2026-08-17)

Run in this order (order only matters in that `universe` must precede `--universe` selectors):

```bash
# a. macro — FRED with vintages, back to start-900d of the earliest window
uv run python -m dynamo.cli.backfill macro --start 2023-06-01

# b. fundamentals — EDGAR companyfacts, filed_at-vintaged; seeds instruments.cik first
uv run python -m dynamo.cli.backfill fundamentals --universe --seed-ciks

# c. news — Finnhub company news covering BOTH campaign windows
uv run python -m dynamo.cli.backfill news --start 2026-01-01 --end 2026-08-16 --universe

# d. crypto bars — Alpaca crypto daily bars (free feed)
uv run python -m dynamo.cli.backfill crypto-bars --start 2024-01-01 --end 2026-08-16

# e. transcripts — bulk history via defeatbeta (requires the transcripts extra)
uv run python -m dynamo.cli.backfill transcripts --universe
```

**Executed results (2026-08-17):**

| Backfill | Outcome |
|---|---|
| macro | 19,791 vintaged points across 12 series (all series reach back to 2023-06) |
| fundamentals | 944,859 facts across 40 symbols; DIA/QQQ/SPY skipped (ETF trusts have no us-gaap companyfacts, 404); 7 ETFs (IWM/TLT/XL*) have no SEC ticker mapping — expected, ETFs carry no fundamentals features |
| news | 55,912 distinct items across all 50 equities/ETFs; crypto pairs skipped (finnhub company-news is equities-only — documented). Actual archive coverage: earliest `published_at` per symbol ranges 2026-01-01 (most symbols) to 2026-01-29 (worst: MSFT/AAPL/META/NVDA — Finnhub truncates high-volume symbols' January chunk), latest 2026-08-16. **Both campaign windows are fully covered** (quant starts 2026-02-02, committee 2026-07-01). CLI per-symbol totals sum higher (77,175) because one article tagging N symbols upserts N times to the same row |
| crypto-bars | 4,557 rows (BTC/ETH/AVAX/LINK 959 bars each = full range; **SOL/USD 721 bars — Alpaca's SOL history restarts 2024-08-26** after its 2023 delisting; acceptable: 550+ bars precede the Feb window) |
| transcripts | 1,040 transcripts across 50 symbols since 2020-01-01 |

Notes recorded for the audit report: Finnhub free tier 429s under sustained load — the CLI
now honors `Retry-After` and retries per symbol; news is provider-revisionist (completeness
untestable) — stamped caveat; XOM's SEC-mapped CIK (0002115436) is the post-2025
reorganization entity with only 202 facts — its fundamentals features will be thin.

All backfills are idempotent natural-key upserts — re-running any of them is safe and is
the documented repair path after a partial failure.

**Post-backfill preflight snapshot (2026-08-17):** `run_preflight` against a hypothetical
quant config (start 2026-02-02) now reports — universe OK (55 symbols), macro_history OK
(reaches 2023-08-17), source_run OK, quant_model WARNING (version unset — expected until
step 4), bar_coverage ERROR at 9.1% (only the 5 crypto pairs have bars — **expected until
step 3**). Exactly the intended state: bars are the sole blocker.

## 3. WAIT — Tiingo equity bars (blocked on `TIINGO_API_KEY`)

When the key lands, add it to `.env`, then:

```bash
uv run python -m dynamo.cli.backfill bars --start 2024-01-01 --end 2026-08-16 --universe
```

(missing key is a loud nonzero exit by design — no silent skip.)

**Spot-check split/dividend adjustment before proceeding** (2–3 symbols):

- NVDA around its 2024-06-10 10:1 split — `adj_close` continuous, raw `close` gaps 10×.
- AAPL across a dividend ex-date (e.g. 2026-02 ex-div) — `adj_close` slightly below `close`
  before the ex-date, equal after.

```bash
uv run python -c "
from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import MarketDataRepository
from datetime import date
repo = MarketDataRepository(make_engine(Settings().database_url))
df = repo.bars_frame(['NVDA'], date(2024,6,3), date(2024,6,17))
print(df)"
```

## 4. Quant training (free) — GO/NO-GO #1

Train on data strictly before the earliest replay window (2026-02-02):

```bash
uv run python -m dynamo.cli.quant_train \
  --train-start 2024-01-01 --train-end 2025-12-31 --label campaign-v1 --universe
```

Persists boosters + feature list, registers model version `campaign-v1` (candidate stage —
production stays on baseline until promoted; candidate use for replay is the documented
split). Prints out-of-fold walk-forward metrics.

**GO/NO-GO #1 — never proceed to any paid step with a degenerate model:**

| Check | GO | NO-GO |
|---|---|---|
| Out-of-fold rank IC | > 0 and stable across folds | ≤ 0, or sign flips fold-to-fold |
| Score dispersion | non-degenerate (scores spread, not near-constant) | all scores ~equal or ~None |
| Downside AUC | > 0.5 | ≤ 0.5 |
| Shortlist eyeball | top-of-book names are plausible momentum/quality picks for their dates | nonsense ordering |

On NO-GO: re-examine features/labels, retrain with a new `--label` (labels are
single-use); do not delete registered versions.

## 5. Leakage / isolation suites vs scratch Postgres — GO/NO-GO #2

```bash
# scratch DB (never the dev DB), migrated to head:
docker run -d --name dynamo-scratch -e POSTGRES_PASSWORD=scratch -p 5433:5432 postgres:16
export INTEGRATION_DB_URL="postgresql+psycopg://postgres:scratch@localhost:5433/postgres"
DATABASE_URL="$INTEGRATION_DB_URL" uv run alembic upgrade head

# integration suite + the anti-leakage belt:
uv run pytest tests/integration -q
uv run pytest tests/unit/replay/test_gateway_leakage.py \
              tests/unit/replay/test_gateway_tripwires.py \
              tests/unit/replay/test_firewall.py \
              tests/unit/replay/test_network_isolation.py \
              tests/unit/replay/test_model_knowledge.py -q
uv run pytest -q          # and the full suite once more

docker rm -f dynamo-scratch   # cleanup
```

**GO/NO-GO #2:** every test passes (socket-block, no-tools `_build_params` regression,
tripwires, cutoff gate, firewall). Any failure is a hard stop before spending.

## 6. Free quant dress rehearsal + 3-bankroll quant campaign — GO/NO-GO #3

Dress rehearsal over the full 6-month window (also *is* the first campaign run):

```bash
uv run python -m dynamo.cli.replay run --mode quant \
  --start 2026-02-02 --end 2026-07-31 \
  --initial-equity 100000 --quant-model-version campaign-v1 \
  --code-version campaign-2026-08 --run-label quant-campaign-100k
uv run python -m dynamo.cli.replay report --run-id <run-id-printed>
uv run python -m dynamo.cli.replay audit  --run-id <run-id-printed>
```

**GO/NO-GO #3:**

| Check | GO |
|---|---|
| Preflight | clean (no `--skip-preflight` needed) |
| Decisions | > 0 (an empty universe / all-abstain run is a NO-GO even though it "succeeds") |
| Equity curve | non-flat |
| Audit | clean `leakage_audit.{json,md}` |

Then the remaining bankrolls (+ optional whole-shares sensitivity):

```bash
uv run python -m dynamo.cli.replay run --mode quant \
  --start 2026-02-02 --end 2026-07-31 \
  --initial-equity 10000 --quant-model-version campaign-v1 \
  --code-version campaign-2026-08 --run-label quant-campaign-10k

uv run python -m dynamo.cli.replay run --mode quant \
  --start 2026-02-02 --end 2026-07-31 \
  --initial-equity 1000 --quant-model-version campaign-v1 \
  --code-version campaign-2026-08 --run-label quant-campaign-1k

# optional small-account realism sensitivity:
uv run python -m dynamo.cli.replay run --mode quant \
  --start 2026-02-02 --end 2026-07-31 \
  --initial-equity 1000 --whole-shares --min-order-notional 1 \
  --quant-model-version campaign-v1 \
  --code-version campaign-2026-08 --run-label quant-campaign-1k-whole-shares
```

Freeze `--code-version campaign-2026-08` across every run of this campaign (determinism
anchor recorded in `config_json`).

## 7. Committee cost estimate (free — ZERO API calls)

```bash
uv run python -m dynamo.cli.replay estimate \
  --start 2026-07-01 --end 2026-08-14 --deep-dives-per-day 4
```

Expected shape: ~32 NYSE sessions × 4 deep dives ≈ 128 dives at ~$0.37 each ≈ **$47**,
plus the per-day macro digest (`generate_macro_digest` defaults on: one analyst-tier call
per session, ~$0.05–0.10/day ≈ $2–4). **Expected total ≈ $45–60 conservative.**
Recommend `--max-usd 80` — headroom without meaningful downside risk (the cap is hard,
enforced by `ReplayBudget`; the run stops cleanly when reached).

The knowledge-cutoff gate resolves here too: a 2026-07-01 start passes cleanly
(> 2026-06-30). If you move the start earlier it will refuse — that refusal is correct;
see the window rationale at the top. Do not reach for `--acknowledge-model-knowledge`.

## 8. USER-TRIGGERED PAID STEP — the committee run

Run only when steps 4–7 are all GO. Extend `--end` to the latest **completed** session on
the day you run (2026-08-14 as of writing; each passing month adds ~21 sessions).

```bash
uv run python -m dynamo.cli.replay run --mode committee \
  --start 2026-07-01 --end 2026-08-14 \
  --max-usd 80 --deep-dives-per-day 4 \
  --quant-model-version campaign-v1 \
  --code-version campaign-2026-08 --run-label committee-jul-aug-2026
```

Committee mode is sizing-free: it produces verdicts once; the three bankrolls replay them
for free in step 9. Within-run learning is on by default (live parity: month 2 benefits
from month 1's resolved experiences, firewalled to this run).

### Claude Code subscription backend (ADR-0014) — optional

With `RESEARCH_BACKEND=claude_code` (or `--backend claude_code` on the command above)
the committee routes through the local `claude` CLI on the flat-rate subscription:
real token counts are recorded but **metered cost is $0**, so `--max-usd` is still
required and mechanically enforced yet effectively inert — the run consumes Claude plan
usage windows instead of dollars. Tier models come from `CLAUDE_CODE_MODEL_*`
(defaults: screen=haiku-4-5, analyst=sonnet-5, arbiter=**fable-5**), and the
knowledge-cutoff gate resolves THOSE models' documented cutoffs (fable-5/sonnet-5:
2026-01-31) — at the default 30-day margin any start from 2026-03-03 onward is clean,
so the July-1 window above passes with margin to spare. Run
`replay estimate --backend claude_code` first: it names the backend, prints the
resolved cutoffs and gate verdict, and projects no dollars.

**Resumability.** If the subscription plan window (5-hour/weekly cap) exhausts mid-run,
the remaining days soft-fail exactly like the quarantine/partial path — no verdicts,
$0 — and the run still finishes with partial results. **Re-run the exact same command
after the window resets**: the run id is deterministic, `replay_verdicts` upserts are
idempotent, and days that already have persisted verdicts are skipped (not
re-submitted), so each re-run fills in only the missing days. Repeat until `replay show
--run-id <run-id>` lists verdicts across the whole window.

**Mid-run checkpoint at ~$15–20 spent** (roughly the end of the first two weeks): in a
second terminal —

```bash
uv run python -m dynamo.cli.replay show --run-id <run-id>
```

Inspect the persisted verdicts/contexts: are the shortlisted symbols sane, are theses
grounded in the provided dossiers (news/transcripts/quant/macro), are confidences spread?
If the shortlists are garbage, kill the run (Ctrl-C is safe — everything is persisted per
session and the budget state is durable) rather than burning to the cap.

## 9. Three portfolio runs (free — verdict reuse)

```bash
for EQ in 1000 10000 100000; do
  uv run python -m dynamo.cli.replay run --mode portfolio \
    --source-run-id <committee-run-id> \
    --start 2026-07-01 --end 2026-08-14 \
    --initial-equity $EQ \
    --code-version campaign-2026-08 --run-label committee-portfolio-$EQ
done

# optional whole-shares sensitivity at $1k (near-free, same verdicts):
uv run python -m dynamo.cli.replay run --mode portfolio \
  --source-run-id <committee-run-id> \
  --start 2026-07-01 --end 2026-08-14 \
  --initial-equity 1000 --whole-shares --min-order-notional 1 \
  --code-version campaign-2026-08 --run-label committee-portfolio-1k-whole-shares
```

Portfolio mode can never construct an LLM client (token-firewall, tested). Each config is
a distinct deterministic run id.

## 10. Reports, compare, audit

```bash
# per-run tearsheets:
uv run python -m dynamo.cli.replay report --run-id <each-run-id>

# cross-run table + overlaid equity curves (quant campaign and committee campaign):
uv run python -m dynamo.cli.replay compare --run-ids <quant-100k> <quant-10k> <quant-1k>
uv run python -m dynamo.cli.replay compare --run-ids <pf-100k> <pf-10k> <pf-1k> <committee-run-id>

# the leakage-audit deliverable (on the PAID run):
uv run python -m dynamo.cli.replay audit --run-id <committee-run-id>
```

Artifacts land in `artifacts/backtests/<run_id>/` (`report.json`, tearsheet,
`leakage_audit.{json,md}`). The audit stamps: tripwire status, resolved cutoffs + gate
decision, embedder backend, context scan results, and the standing caveats
(instruments/sectors unvintaged; news provider revisionism; no social history).

## 11. Optional — generational training with replay experiences

```bash
uv run python -m dynamo.cli.generations --include-replay <committee-run-id>
```

Replay experiences are excluded by default everywhere (`source='replay'` firewall);
`--include-replay` is the sole, explicit opt-in. Bankroll-scale features
(`log10_equity_at_decision`) are live, so the meta-model can learn scale-dependent trust.
Only feed it the **clean-window** committee run — never a contaminated one (see top).

## 12. Interpretation caveats (read before quoting numbers)

1. **Horizon-tail synthetic exits:** positions opened within ~20 sessions of the window
   end get synthetic horizon exits at window close. For the quant campaign that means
   entries after ~2026-07-03; for a Jul-1→Aug-14 committee window the effect covers a
   *large fraction* of the short run — treat committee-window return metrics as
   directional, and weight decision-quality/calibration evidence (audit, verdict review,
   experience outcomes) over headline return.
2. **Short committee window:** ~32 sessions is too few for statistically strong return
   claims. The campaign's primary yield is (a) proof the full pipeline runs leak-free
   end-to-end on real history, and (b) a growing set of clean experiences. Re-run with a
   longer window as the clean period grows.
3. **Quant campaign covers the mechanics:** Feb–Jul bankroll divergence (whole shares,
   min notional, commissions, ladder behavior, crypto sleeve caps) is established by the
   free quant runs; the committee window doesn't need to re-prove it.
4. **Fidelity gaps (stamped in the audit):** no social-signal history, no options sleeve,
   no prediction-market sleeves, no intraday entries, no tier-2 surveillance sweep,
   next-open cost-model fills, today's sector labels. Everything else is the real code
   path.
5. **Crypto in replay** trades on equity sessions only, marked to weekend closes
   (documented v1 simplification). SOL/USD history begins 2024-08-26 (Alpaca relisting).
6. **News completeness is provider-limited:** Finnhub's archive for the window was
   fetched wholesale (76,970 items, 2026-01→2026-08); provider-side revisionism cannot be
   ruled out and is a stamped caveat, not a fixable defect.
