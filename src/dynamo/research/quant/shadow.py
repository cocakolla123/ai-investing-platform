"""Forward shadow scoring: candidate bundles scored beside production.

docs/preregistration_trend_shadow.md. Rows go to ``shadow_quant_scores``,
which nothing that trades reads. One bundle failing to load or score is
reported, never raised: a shadow must not be able to fail the nightly
scores job that production depends on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import pandas as pd

from dynamo.research.quant.scoring import ModelBundle, score_universe


def shadow_bundle_names(setting: str) -> list[str]:
    return [name.strip() for name in setting.split(",") if name.strip()]


def score_shadow_bundles(
    features: pd.DataFrame,
    as_of: date,
    names: Sequence[str],
    load: Callable[[str], ModelBundle],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """-> (rows for shadow_quant_scores, {bundle: error} for bundles that failed)."""
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for name in names:
        try:
            bundle = load(name)
            rows.extend(s.model_dump(mode="json") for s in score_universe(bundle, features, as_of))
        except Exception as exc:  # isolation: report, never fail production
            errors[name] = f"{type(exc).__name__}: {exc}"
    return rows, errors
