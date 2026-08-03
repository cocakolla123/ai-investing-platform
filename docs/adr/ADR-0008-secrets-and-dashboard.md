# ADR-0008 — Secrets via .env + sops/age (no Vault); dashboard Streamlit-then-React

**Status:** accepted · 2026-08-03

## Secrets
**Decision:** per-env `.env` files on the host; sops/age-encrypted copies committed for backup
parity; GitHub Actions encrypted secrets for CI; structlog redaction processors; per-service DB
users with minimal grants.
**Rejected:** HashiCorp Vault / cloud secret managers — an always-on secret-serving dependency
and operational surface far beyond a single-user host's needs.
**Revisit:** if the platform ever serves more than one operator or one host.

## Dashboard
**Decision:** Streamlit for M1–M5 (report browsing, portfolio/risk tables — days to build,
explicitly disposable), then React + Vite + TypeScript + shadcn/ui + TradingView
lightweight-charts + TanStack Query against the FastAPI API at M6, when alert acking,
drill-downs, and journal-search UX outgrow Streamlit.
**Rejected:** Next.js (no SSR/SEO need), Dash/Panel (worse long-term fit than either phase).
Grafana keeps ops/system metrics; the app dashboard shows investment views only.
