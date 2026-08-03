# ADR-0004 — Data provider set: EDGAR + FRED + Alpaca free + Tiingo paid + Finnhub free; yfinance break-glass only

**Status:** accepted · 2026-08-03

## Context
Hard budget ~$50–150/mo total, of which Claude gets the largest share. Needs: full-market EOD
bars, survivorship-clean history, point-in-time fundamentals, filings, insider activity, news,
macro, earnings calendar, prediction markets.

## Decision
| Provider | Role | Cost |
|---|---|---|
| SEC EDGAR | Filings, XBRL point-in-time fundamentals, Form 4, 13F | Free |
| FRED | Macro series (with vintages) | Free |
| Alpaca free data | Bars/quotes/corporate actions + broker | Free |
| Tiingo | Survivorship-aware adjusted EOD incl. delisted; second price source | ~$10/mo |
| Finnhub free | News headlines, earnings calendar, analyst actions | Free |
| Kalshi + Polymarket APIs | Prediction-market reads | Free |

Dual-source price validation (Alpaca vs Tiingo) is a designed-in quality gate, not redundancy
waste. **yfinance is break-glass only** — unofficial, scraping-based, breaks without notice;
it never appears in the production dependency path.

## Alternatives rejected
- **Polygon starter ($29/mo):** excellent data, but the marginal quality over Alpaca+Tiingo is
  worth less than the same dollars spent on Claude tokens at this stage.
- **FMP:** fundamentals quality inconsistent at starter tier vs EDGAR-derived facts.

## Consequences
~$10/mo data spend; every provider behind a port with provenance tags, so upgrades (e.g., to
Polygon) are adapter swaps justified by measured data-quality incidents.
