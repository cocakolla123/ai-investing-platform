"""Fixtures for the alpha-research rails tests: tiny synthetic panels,
candidate files written to tmp_path, and in-memory fakes for every store.
No database, no network, no LLM."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dynamo.research.discovery.registry import Hypothesis, HypothesisStatus

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)

GOOD_CANDIDATE = """\
import pandas as pd

from dynamo.research.quant.catalog import FeatureContext, FeatureSpec


def _fn(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 6:
        return pd.Series(float("nan"), index=list(ctx.universe), dtype=float)
    return (p.iloc[-1] / p.iloc[-6] - 1.0).reindex(ctx.universe)


CANDIDATE = FeatureSpec(name="ret_5d_test_candidate", group="momentum", lookback_days=6, fn=_fn)
"""


@pytest.fixture
def candidate_file(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.py"
    path.write_text(GOOD_CANDIDATE, encoding="utf-8")
    return path


def make_hypothesis(
    hypothesis_id: str = "test-hypo",
    *,
    registered_at: datetime = NOW,
    status: HypothesisStatus = HypothesisStatus.REGISTERED,
    trial_family: str = "test-family",
    audit_results: dict[str, Any] | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        title="A test hypothesis",
        rationale="short-horizon continuation should persist in this universe",
        prediction="positive rank IC ~0.02",
        trial_family=trial_family,
        registered_at=registered_at,
        data_requirements="daily bars",
        status=status,
        audit_results=audit_results,
    )


def planted_audition_panel(
    n_sessions: int = 140,
    n_symbols: int = 12,
    seed: int = 11,
    signal_strength: float = 2.0,
):
    """AuditionPanel with a candidate that (optionally) drives the label.

    ``signal_strength=0`` makes the candidate pure noise. Base features are
    always noise, so any pooled-IC lift with the candidate is the candidate's.
    """
    from dynamo.research.discovery.audition import AuditionPanel

    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2024-01-02", periods=n_sessions)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([sessions, symbols], names=["ts", "symbol"])
    base = pd.DataFrame(
        {"noise_a": rng.normal(size=len(idx)), "noise_b": rng.normal(size=len(idx))},
        index=idx,
    )
    cand = pd.Series(rng.normal(size=len(idx)), index=idx, name="planted_candidate")
    label = signal_strength * cand + rng.normal(size=len(idx))
    labels = pd.DataFrame({"excess_return_20d": label}, index=idx)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_sessions, n_symbols)), axis=0)),
        index=sessions,
        columns=pd.Index(symbols),
    )
    return AuditionPanel(
        base_features=base,
        candidate=cand,
        labels=labels,
        prices=prices,
        sessions=tuple(d.date() for d in sessions),
    )


class FakeHypothesisStore:
    def __init__(self) -> None:
        self.rows: dict[str, Hypothesis] = {}
        self.audits: list[tuple[str, HypothesisStatus, dict[str, Any]]] = []

    def insert(self, h: Hypothesis) -> None:
        if h.id in self.rows:
            raise ValueError(f"duplicate hypothesis {h.id}")
        self.rows[h.id] = h

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        return self.rows.get(hypothesis_id)

    def list_all(self) -> list[Hypothesis]:
        return sorted(self.rows.values(), key=lambda h: h.registered_at)

    def record_audit(
        self, hypothesis_id: str, status: HypothesisStatus, audit_results: dict[str, Any]
    ) -> None:
        self.audits.append((hypothesis_id, status, audit_results))
        h = self.rows[hypothesis_id]
        self.rows[hypothesis_id] = h.model_copy(
            update={"status": status, "audit_results": audit_results}
        )

    def set_status(self, hypothesis_id: str, status: HypothesisStatus) -> None:
        h = self.rows[hypothesis_id]
        self.rows[hypothesis_id] = h.model_copy(update={"status": status})


class FakeTrialCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def increment_trials(self, family: str) -> int:
        self.counts[family] = self.counts.get(family, 0) + 1
        return self.counts[family]


class FakeProposalStore:
    def __init__(self) -> None:
        self.inserted: list[Any] = []

    def insert(self, proposal: Any) -> None:
        self.inserted.append(proposal)
