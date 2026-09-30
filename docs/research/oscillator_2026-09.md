# Short-cycle oscillators — buy the bottom of the recent range, sell the swing (2026-09-30)

Script: `scripts/research/oscillator_backtest.py`. Window 2025-09-29 → 2026-09-29,
10,341 US symbols, price ≥ $5, ≥ $5M/day. Filter mirrors the scanner's
`range_bound` detector (variance ratio ≤ 0.90 over 120d; ≥ 2 closes beyond both
20-day 2σ bands in 60d). Point-in-time, next-open fills, 15 bps per side,
−8% stop, 10-session time stop. 8 pre-declared variants, all reported.

| negative autocorr | entry | exit | trades | win % | avg/trade | SPY same days | excess | all-stocks excess |
|---|---|---|---|---|---|---|---|---|
| no | bottom 25% of 60d range | back to 20d mean | 4,545 | 52.1 | −0.34% | +0.26% | −0.60% | −0.58% |
| no | bottom 25% | top 25% | 3,926 | 46.7 | +0.10% | +0.65% | −0.55% | −0.87% |
| no | below lower band | 20d mean | 3,481 | 50.7 | −0.10% | +0.25% | −0.35% | −0.32% |
| no | below lower band | top 25% | 3,477 | 48.2 | +0.17% | +0.43% | −0.26% | −0.32% |
| yes | bottom 25% | 20d mean | 3,430 | 53.5 | −0.06% | +0.29% | −0.35% | — |
| yes | bottom 25% | top 25% | 2,942 | 48.3 | +0.46% | +0.74% | −0.27% | — |
| yes | below lower band | 20d mean | 2,456 | 51.5 | +0.04% | +0.31% | −0.28% | — |
| yes | below lower band | top 25% | 2,455 | 48.7 | +0.33% | +0.52% | −0.19% | — |

**Every variant trails SPY over the same holding days** (−0.19% to −0.60% per
~6-day trade). The oscillator filter barely differs from applying the same rules
to every stock, so it is not selecting anything useful. Before costs (add back
0.30% per round trip) the excess is roughly −0.3% to +0.1%: the swings are
real but about the size of trading costs, and a long-only swing trader sits out
of a rising market between trades.
