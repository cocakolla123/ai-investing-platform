# Runbook — Claude budget exhaustion

**Use when:** `claude_budget_utilization` reaches 0.7 (warn) or 1.0 (page).

**Severity:** warn at 70%, page at 100%. **Expected time:** 15 minutes.

---

## 0. The single most important fact

**This is a research-only stop. Orders, stops, reconciliation and the risk
engine are untouched.**

`CostMeter.check_budget()` raises `BudgetExceededError` once month-to-date
spend reaches `CLAUDE_MONTHLY_BUDGET_USD`, and that only prevents *new Claude
calls*. Nothing in the execution path calls Claude. An exhausted budget means
you stop generating new theses; it does not mean your book is unmanaged, and
it is **not** a reason to engage the kill switch.

## 1. Confirm where you are

```console
$ curl -s http://127.0.0.1:8000/ops/summary | jq '.data.claude_spend'
{
  "month_to_date_usd": 63.10,
  "monthly_budget_usd": 90.0,
  "budget_utilization": 0.701
}
```

`cost_ledger` has the per-call detail: which model, which `call_kind`, batch
or not.

```sql
SELECT model_id, call_kind, batch, count(*), round(sum(cost_usd)::numeric, 2)
FROM cost_ledger
WHERE ts >= date_trunc('month', now())
GROUP BY 1, 2, 3
ORDER BY 5 DESC;
```

## 2. Ask whether the spend is normal

Expected shape in the paper phase: ~35 committee calls/night, most of the
dollars in the **arbiter** (Opus) tier, most of the calls in the **screen**
(Haiku) tier, and batch calls at half price.

Red flags:

- A `call_kind` you did not expect to be large.
- `batch=false` on calls that should be batched — batching is 50% off, so a
  fallback to sequential doubles the bill.
- A day whose spend is several times the median: usually a retry loop, or a
  replay campaign run against the API backend by accident.
- Cache-read tokens near zero on the analyst tier: prompt caching is not
  working, and prompts are being re-billed in full.

## 3. Reduce the burn (in order of preference)

**3a. Turn down the funnel.** These five settings are ONE budget, and
`dynamo.research.funnel` enforces the relationship
(`triage_cap * material_rate <= deep_dive_promotion_slots`). `test_funnel.py`
fails if you break it, which is the point — change them together:

```
deep_dive_cap_per_night     8   -> 4       # the big lever: fewest, priciest calls
tier2_triage_cap_per_night  16  -> 8
deep_dive_promotion_slots   5   -> 3
```

**3b. Switch the backend to the flat-rate subscription** (ADR-0014):

```
RESEARCH_BACKEND=claude_code
```

Marginal cost goes to $0 for the nightly committee. Costs: no batch discount,
no prompt caching, and the plan's own 5-hourly/weekly windows can exhaust
mid-run (`PlanLimitExhausted`). Fine for nightly paper research; wrong for a
replay campaign.

**3c. Raise the budget** — only if the spend is *correct* and the value is
there. Editing `CLAUDE_MONTHLY_BUDGET_USD` to make an alert stop firing is
how a solo operator discovers a retry loop three months and several hundred
dollars later.

**3d. Accept degradation.** Note in the ops log the date research stops for
the month. Quant scores, risk and execution carry on; the platform is
designed to run without the committee.

## 4. What NOT to do

- Do **not** engage the kill switch. Wrong tool: it stops trading, and the
  budget has nothing to do with trading.
- Do **not** delete `cost_ledger` rows to reset utilization. The ledger is the
  audit trail, and month-to-date is computed from it.
- Do **not** disable the budget check. It is the only thing between a retry
  loop and your card.

## 5. Log it

`docs/ops-log.md`: date, month-to-date, what drove it, which lever you pulled,
and — if you raised the budget — the reason it was the right call.

---

## Known gaps

- `claude_budget_utilization` is a live Prometheus gauge set by `CostMeter`,
  so it exists only in a process that has made a call. There is no Grafana
  alert rule shipped in `infra/monitoring/`; today the 70%/100% thresholds are
  visible on `/ops/summary` and in the daily report, and the 70% warning is a
  log line, not a page. Check `/ops/summary` weekly until a rule exists.
