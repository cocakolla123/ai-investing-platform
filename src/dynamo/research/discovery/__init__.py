"""Alpha-research rails (ADR-0015): pre-registered hypothesis auditions.

Pure research core — hypothesis pre-registration discipline (registry.py) and
the candidate-feature audition harness (audition.py). Persistence lives in
``dynamo.persistence.repositories.discovery``; wiring in ``dynamo.cli.alpha``.

Rails only: nothing here runs scheduled, calls an LLM, or promotes anything.
Survivors become NEW_FEATURE ImprovementProposals for a human to judge.
"""
