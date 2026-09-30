# Range-floor strategy — buy the floor, sell the range, repeat (2026-09-30)

Script: `scripts/research/range_floor_backtest.py` (`--grid` for all variants).
Window: 2025-09-29 → 2026-09-29, 11,381 US symbols, price ≥ $5, ≥ $5M/day
consolidated-equivalent dollar volume. Point-in-time: the range on day t comes
only from the prior 252 sessions; fills at the next open; 15 bps per side.

## Result: does not work — every one of 16 pre-declared variants loses to SPY

| variant space | per-trade return | SPY over same days | excess vs SPY |
|---|---|---|---|
| all 16 variants | −0.79% … +0.20% | +0.97% … +1.75% | **−1.41% … −2.35%** |
| default (2 prior bounces, 60%-of-range target, 5% stop) | −0.18% | +1.30% | −1.48% (676 trades, 538 names) |

Default exits: 8% reached the target, 39% broke the floor (stop), 52% timed out.
A stock that has bounced off its floor twice breaks through it about 4 times
in 10 on the next test. Requiring 3 prior bounces, or requiring the entry day
to close above the floor, did not help (−1.6% to −2.4%).

## Why the "screen" looks like it works (hindsight)

128 stocks fit the pattern over the whole year (≥3 bounces, sideways, 15–100%
wide range) — e.g. GRBK, SMA, DEC, OSW, MQ, PAGS, RDN, OBDC. Running the SAME
strategy on only those names: **+7.1% per trade, 69% win rate, +4.7% vs SPY.**
That is selection, not skill: they were picked *because* their floors held.
The strategy cannot know in advance which floors will hold.

## Caveats

- One year, in a rising market (stocks at 52-week lows were laggards); the
  published literature (George & Hwang 2004) also finds near-low stocks
  underperform.
- Survivorship: delisted names are largely absent, so real breakdowns were
  worse than measured — the negative result is conservative in direction.
- Naive per-trade t-stats overstate significance (trades cluster in time).
