# Bankroll risk profile v1 result

Pre-registration: `docs/preregistration_bankroll_profile.md` (35fb825),
amended before any result was read. Runs on `--code-version bankroll-5dd26dd`,
which includes the four replay fixes. Analysis:
`scripts/eval/bankroll_analyze.py` → `evals/bankroll/results.json`. All runs
replay the same saved committee verdicts (`replay-63220a3a5e9a`): $0, no
model calls.

**Verdict: no difference shown at any bankroll.** No paired daily t reached
±2. That was the expected outcome. The profile stays off; the nightly paper configuration is unchanged.

| bankroll (tier) | arm | return | vs SPY | Sharpe | max DD | alpha/yr (t) | symbols traded | avg positions (incl. SPY) | costs |
|---|---|---|---|---|---|---|---|---|---|
| $1k (small) | flat | +11.23% | −3.49% | 1.88 | −7.06% | −2.15% (−0.34) | 12 | 3.6 | $3 |
| | profile | +12.80% | −1.92% | 2.09 | −7.28% | −0.45% (−0.11) | **2** | 1.5 | $1 |
| $10k (medium) | flat | +11.20% | −3.52% | 1.88 | −7.06% | −2.20% (−0.35) | 12 | 3.6 | $31 |
| | profile | +9.63% | −5.09% | 1.62 | −7.87% | −5.89% (−1.02) | 11 | 3.2 | $32 |
| $100k (large) | flat | +11.12% | −3.60% | 1.87 | −7.07% | −2.35% (−0.37) | 12 | 3.6 | $378 |
| | profile | +11.12% | −3.60% | 1.87 | −7.07% | −2.35% (−0.37) | 12 | 3.6 | $378 |

Paired daily return, profile − flat:

- $1k: +3.1%/yr, t +0.47
- $10k: −3.1%/yr, t −1.37
- $100k: identical (t undefined)

## Reading it

- **$100k: identical to flat.** The large tier keeps today's caps. Its 15%
  risk sleeve never bound, because crypto plus discovery-added names never
  exceeded 15% of the book. The profile only changes behavior at large
  sizes when the committee buys a lot of risky names at once.
- **$1k: 2 symbols.** With a 0% risk sleeve, every crypto and
  discovery-added buy was blocked, leaving SPY plus McDonald's. That came
  out +1.6 points better than flat. That suggests those names, taken together, lagged what the
  slot would otherwise hold (mostly SPY) in this window, but it wasn't
  measured name by name. One window, not significant.
- **$10k: −1.6 points.** Fewer names and a 10% sleeve. Not significant.
- **Concentration didn't happen.** Small accounts were meant to hold fewer,
  larger positions. But the committee only says "buy" to a few names at a
  time, so the position count never binds. The table's "avg positions" counts
  every name with a target weight, SPY ballast included (3.6 when flat). The
  committee's own names average 2.6 per session
  (`evals/rerun/ballast_share.json`). The book is mostly SPY ballast.

## What would make this test informative

The profile can only matter when the committee generates many concurrent
buys. Today it averages under 4 positions. Two things would change that:

- a longer window;
- a committee that buys more often.

Until then, small/medium/large mostly differ by whether crypto and
discovery names are allowed, which is a bet on how those did in one window.
The profile stays available (`--bankroll-profile v1`) for a future test,
with no re-tuning of its numbers.
