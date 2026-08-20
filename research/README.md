# `research/` — inputs to the alpha-research rails (ADR-0015)

Not a python package and not imported by anything in `src/`. This directory
holds the two kinds of file the rails read from disk:

- `hypotheses/*.yaml` — pre-registrations, fed to
  `python -m dynamo.cli.alpha register --file ...`. Written **before** any
  test runs; the audition refuses a hypothesis whose `registered_at` is not
  strictly earlier than the audition invocation.
- `candidates/*.py` — candidate feature files, each exposing a module-level
  `CANDIDATE: FeatureSpec` (contract enforced by
  `dynamo.research.discovery.audition.load_candidate`). Candidates are thin:
  the arithmetic lives under `src/dynamo/research/quant/` so it is covered by
  ruff, pyright and pytest like everything else, and so tests and the
  candidate share exactly one definition.

A candidate file is **not** a production feature. Nothing here is in
`FEATURE_CATALOG`, nothing here changes a trained model, and a passed
audition only opens a `NEW_FEATURE` proposal for human review.

## Scope note: what today's auditions can and cannot show

The platform currently holds daily history for roughly **55 mega-cap names**.
That is a small cross-section, and it is a biased one for most signal ideas —
mega-caps are the most arbitraged, most continuously repriced, deepest-book
names in the market.

The consequence, stated plainly so nobody reads more into a verdict than is
there:

- An audition on today's panel is a **sanity check**, not a test of the
  hypothesis. It confirms the feature computes, is not degenerate, and does
  not smuggle in lookahead. It is close to uninformative about whether the
  effect exists in the universe the hypothesis is actually about.
- A **FAIL** on 55 mega-caps is weak evidence against a hypothesis aimed at
  small and mid caps. A **PASS** on 55 mega-caps is more likely to be a
  small-sample artefact than a discovery and should be treated as one — the
  rails' trials-adjusted bar exists precisely because the null is wide here.
- The audition **consumes a trial** from the hypothesis's family either way.
  That is the rails working as designed. Do not restructure families, delay
  registration, or split hypotheses to dodge the trial cost; the cost is the
  mechanism.
- The real test of any cross-sectional hypothesis arrives after the wide
  (~1,000 name) backfill. Re-testing on that panel is a **new hypothesis and
  a new trial**, not a re-run of the old one.

This applies to `range-reversion-meanrev` most sharply of all: range-bound
oscillation is a small/mid-cap phenomenon, and the mega-cap panel is the
worst available place to look for it.
