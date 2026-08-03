# ADR-0005 — Committee v2: 4-call adversarial structure instead of parallel role roster

**Status:** accepted (user-approved) · 2026-08-03

## Context
The original spec listed ~15 analyst roles. Naively, that is 10+ API calls per ticker per night
— far beyond the budget — and most of those calls would duplicate *extraction* work (reading
the same filings/news) rather than contribute independent *judgment*.

## Decision
Per deep-dive: (1) Sonnet **dossier analyst** builds a stance-free, evidence-referenced fact
base; (2) Sonnet **bull advocate** and (3) Sonnet **bear advocate** argue independently from
the same dossier without seeing each other; (4) **Opus arbiter** synthesizes with the risk
lens, decision audit, and self-critique folded in, and must rebut bear kill-cases with evidence
to emit `buy`. Macro and prediction-market digests run **once daily for the whole market**.
Event specialists (earnings calls, filing deep-reads) run only when their event fires. Haiku
handles screening/triage. All research via Message Batches (50% off).

## Rationale
Independence pays at the judgment level, not the extraction level. Preserving genuine
adversarial tension (bull/bear isolation + mandatory rebuttals) captures the value of a
committee; consolidating extraction cuts per-name cost ~45%, funding ~8–10 deep dives/night
inside $55–90/mo.

## Consequences
Every role from the spec maps to a call or a folded-in responsibility (traceability table in
doc 03). Risk: the single dossier becomes a shared-bias bottleneck — mitigated by advocates
receiving raw excerpts alongside the dossier and by per-agent calibration tracking (doc 10).
