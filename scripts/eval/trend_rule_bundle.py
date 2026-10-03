"""Write the ``trend-rule-v1`` quant bundle (docs/preregistration_trend_rule.md).

A rule, not a trained model: the ranker is :class:`TrendRule` (12-1
momentum), downside and vol abstain. Saved with the ordinary bundle writer,
so replay, and later the live scorer, load it like any other bundle. It
carries no ship-gate metrics, because nothing was fitted; preflight reports
that as a warning, and the pre-registered replay result is the evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

from dynamo.research.quant.bundle import default_bundle_dir, save_bundle
from dynamo.research.quant.models import AbstainModel, TrendRule

VERSION = "trend-rule-v1"
IC, SIGMA = 0.04, 0.08


def main() -> None:
    params = {"rule": "mom_12_1 rank -> IC x sigma x z", "ic": IC, "sigma": SIGMA}
    path = save_bundle(
        default_bundle_dir(),
        VERSION,
        ranker=TrendRule(ic=IC, sigma=SIGMA),
        downside=AbstainModel(),
        vol=AbstainModel(),
        feature_names=["mom_12_1"],
        train_feature_quantiles=None,
        # Not a training window: the period whose evidence (eval Phase 5,
        # docs/ml_baselines_2026-10-03.md) motivated the rule.
        train_start=date(2021, 3, 1),
        train_end=date(2026, 9, 23),
        n_samples=0,
        config_hash=hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
        metrics={
            "kind": "rule",
            "params": params,
            "evidence": {
                "source": "docs/ml_baselines_2026-10-03.md",
                "pooled_rank_ic": 0.0399,
                "fold_t": 3.14,
                "folds": 19,
                "universe": "config/training_universe_wide.txt (1,315)",
            },
        },
        fold_ic_dispersion=0.0,
        vol_is_log=False,
    )
    print(path)


if __name__ == "__main__":
    main()
