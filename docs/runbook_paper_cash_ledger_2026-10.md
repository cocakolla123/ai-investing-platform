# Runbook: paper cash ledger fix and release (2026-10)

Branch `fix/paper-cash-ledger`. Nothing on this branch has been merged,
applied, or sent to the broker. Every numbered step below needs the owner's
go, one step at a time.

## What broke

- The first paper fill was 2026-09-24: 123.7092 SPY at $764.62.
- Internal cash came from the latest `equity_curve` row, and the nightly
  snapshot wrote that same cash back. Cash stayed at $100,000 while the
  broker showed $5,409.29.
- Internal equity counted the SPY position on top of cash that had already
  paid for it, so it showed about $194.5k against about $100.6k at the broker.
- Reconciliation logged a cash and an equity divergence on every run: 176
  runs, ids 123 to 647, none acknowledged. The sticky halt has blocked paper
  execution and stop maintenance since then. The SPY disaster stop has never
  been placed.

## What the branch changes

| commit | change | safe to merge before the data repair? |
|---|---|---|
| e386fcb | Cash comes from a ledger: opening cash plus every fill. Reconciliation compares equity within 1% (daily close vs intraday marks); cash and positions stay strict. The nightly snapshot never rewrites the opening row. | yes |
| 199419a | The portfolio state gets a high-water mark (it was never set, so the drawdown ladder could never move). | **no**: before the repair the stored curve peaks near $195.4k, which reads as a 48% drawdown (DD20: kill switch, full exit) |
| (repair commit) | `scripts/paper_repair_ledger.py` and `dynamo.execution.ledger_repair` | n/a (script only) |

Tests: 2,717 unit tests pass (17 new) and 69 integration tests pass on the
scratch database (9 new). The regression test fails on the old code (cash
stays at 100,000) and passes on the new code.

## Dry run (2026-10-05, read only)

`scripts/paper_repair_ledger.py` with no flags:

- **equity_curve:** 9 of 11 rows to fix (09-25 to 10-05). Cash goes from
  100,000 to 5,409.29 on each. Equity goes from about 194.3k-195.4k to
  99.7k-100.8k. The 10-03 row becomes 100,620.84; broker equity on the
  evening of 10-02 was 100,611.89. The opening row (09-15) and the 09-24 row
  (written before the fill) are unchanged.
- **ladder_states:** 8 of 10 rows to fix. The high-water mark becomes
  100,832.38. The worst drawdown is −1.1% (10-02). Every rung stays `normal`,
  and the repaired numbers imply `normal` too.
- **cancel:** one order is stuck in `pending_approval`: UTHR
  `bbd67a8862331fe741171242`, buy 14.2684, created 10-01. It has no broker
  order id, so it never reached the broker.
- **ack:** 176 unacked divergent runs (123 to 647). This step is blocked
  until a clean equities run exists.

## Why the target is 100% cash

- Bars for 2026-09-28 and 09-29 are missing for almost the whole universe:
  5 to 7 symbols those days, against about 7,900 on the days around them.
  The nightly ingest did not run.
- The data checks quarantined 109 symbols for the gap, SPY among them
  (quarantine rows 256, 367, 492). The latest target portfolio
  (`tp-ce4201e2292de182`, 10-05) holds no names: 100% cash.
- **If the halt were released as things stand, the next paper run would try
  to sell the whole SPY position.**

## Order of operations

Each step waits for the owner's go.

1. **Fix the broker DNS problem.** `paper-api.alpaca.markets` failed to
   resolve on 10-05. Nothing below works until it resolves.
2. **Repair the curve:**
   `scripts/paper_repair_ledger.py --step curve --apply`. Check the printed
   backup table name and the row count.
3. **Repair the ladder:**
   `scripts/paper_repair_ledger.py --step ladder --apply`. It refuses to run
   before step 2.
4. **Cancel the stale UTHR approval:**
   `scripts/paper_repair_ledger.py --step cancel --apply --order-id bbd67a8862331fe741171242`.
   This changes the database only. It refuses any order that has a broker
   order id.
5. **Backfill the missing bars.** Re-run the `eod_ingest` partitions for
   2026-09-28 and 2026-09-29. This is a heavy job; run it alone. The next data
   check should resolve the gap quarantines, and the next risk and portfolio
   run (00:30 ET) should produce a target with names in it. Check that
   target before step 8.
6. **Merge the branch into main.** Dagster loads code from main, so this is
   the deploy. Run the dry run again first: it must show 0 curve and 0
   ladder rows to fix.
7. **Wait for one clean reconciliation.** It runs every 15 minutes, 09:00 to
   16:45 ET on weekdays. The script's dry run shows the latest equities run
   under `ack.latest_equities_run`. It must be `clean` (scope `equities`).
8. **Decide, then acknowledge.** Acknowledging lifts the halt.
   - After the ack, paper execution trades toward that night's target, and
     stop maintenance places the SPY disaster stop at the paper broker.
   - Command:
     `scripts/paper_repair_ledger.py --step ack --apply --acked-by <name> --note "<why>"`.
     It refuses unless step 7 holds.
9. **Watch the first night.** Expect:
   - a clean reconciliation;
   - an `equity_curve` row with cash about 5,409 (or the new cash after any
     fills);
   - a `ladder_states` row with a real high-water mark.

## Rolling back

- **Data:** each applied step prints its backup table, for example
  `equity_curve_backup_<stamp>`. To restore, delete the table's rows and
  insert the backup's rows in one transaction.
- **Acks:** these are rows in `reconciliation_acks`. Deleting them brings
  the halt back.
- **Code:** revert the merge on main.

## Known gaps this branch does not fix

- **Stale approvals.** Nothing expires an order left in `pending_approval`.
  The approval TTL setting is not enforced. Step 4 clears the one that
  exists today.
- **Halted orders.** The intraday path does not check the halt and the
  quarantine before inserting an order.
- **`decision_price_lookup` is not wired**, so slippage against the decision
  price is not recorded.
- **Options and prediction-market fills are not in the cash ledger.** If one
  ever lands on the same broker account, it shows up as a cash divergence
  rather than being absorbed silently.
- **The opening row has no write time.** `equity_curve` has no timestamp
  column, so the opening row has to come from a day with no fills. The
  current opening row (09-15) meets that.
