"""Train the advisory meta-model from resolved LIVE experiences (plan Phase 3).

Usage:
    python -m dynamo.cli.meta_train [--database-url URL] [--algo gradient_boosting]
        [--model-id meta-...] [--neighbors 3]

Manual only — NEVER scheduled. ZERO LLM calls. Hard-gated on
``MIN_TRAIN_EXPERIENCES`` (150) resolved live experiences: below that the
trainer refuses and prints how far along the platform is. The trained model
is saved as a joblib artifact with a sha256 manifest and registered in the
database registry as a CANDIDATE (``model_type='meta'``) — production status
is reachable only through the promotion pipeline + signed human approval.

Run from the repository root (the ``orchestration`` composition package
provides the wired resource, mirroring ``dynamo.cli.experience_backfill``).
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-meta-train",
        description="Train the advisory meta-model on resolved live decision experiences",
    )
    parser.add_argument(
        "--database-url", default=None, help="override DATABASE_URL / settings (optional)"
    )
    parser.add_argument(
        "--algo",
        default="gradient_boosting",
        choices=("gradient_boosting", "logistic"),
        help="base classifier (default: gradient_boosting)",
    )
    parser.add_argument(
        "--model-id", default=None, help="registry id (default: meta-<algo>-<timestamp>)"
    )
    parser.add_argument(
        "--neighbors", type=int, default=3, help="k nearest prior experiences per row"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Lazy imports: the CLI is the composition layer; keeping them inside
    # main() keeps module import side-effect free and the core pure.
    from pathlib import Path

    import pandas as pd
    from orchestration.resources import DynamoResource, utcnow

    from dynamo.domain.models.learning import ModelVersion
    from dynamo.learning.artifacts import save_model
    from dynamo.learning.hierarchy import CalibratedProbability
    from dynamo.learning.meta_model import (
        MIN_TRAIN_EXPERIENCES,
        MetaModel,
        build_feature_row,
        features_hash,
    )
    from dynamo.persistence.repositories.learning import ModelRegistryRepository

    args = _build_parser().parse_args(argv)

    dynamo = DynamoResource(database_url=args.database_url or "")
    exp_repo = dynamo.experience_repo()

    # source='live' only, resolved only, with a usable win/loss label.
    experiences = [
        e
        for e in exp_repo.resolved_live()
        if e.label in ("win", "loss") and e.conviction is not None
    ]
    n = len(experiences)
    if n < MIN_TRAIN_EXPERIENCES:
        print(
            f"refused: {n} resolved live experiences with labels — the hard gate is "
            f"{MIN_TRAIN_EXPERIENCES}. Keep trading; the platform is "
            f"{n / MIN_TRAIN_EXPERIENCES:.0%} of the way there."
        )
        return 2

    rows: list[dict[str, float]] = []
    labels: list[bool] = []
    order: list[tuple[object, ...]] = []
    for exp in experiences:
        # similar() hard-filters as_of < the row's decision date — neighbor
        # features are strictly point-in-time, no outcome leakage.
        neighbors = exp_repo.similar(
            symbol=exp.symbol,
            as_of=exp.as_of,
            sector=exp.sector,
            regime=exp.regime,
            sleeve=exp.sleeve,
            limit=max(args.neighbors, 0),
        )
        calib = (
            None
            if exp.calibrated_probability is None
            else CalibratedProbability(
                value=exp.calibrated_probability,
                scope_used=exp.calibration_scope_used or "global",
                n_effective=0,  # not denormalized onto experience rows
            )
        )
        rows.append(build_feature_row(exp, calib, neighbors))
        labels.append(exp.label == "win")
        order.append((exp.as_of, exp.created_at, exp.id))

    # Rows are dicts keyed exactly by FEATURE_NAMES (asserted in the builder),
    # so plain construction preserves the deterministic column order.
    frame = pd.DataFrame(rows)
    model = MetaModel(algo=args.algo)
    metrics = model.train(frame, labels, order)

    train_start = min(e.as_of for e in experiences)
    train_end = max(e.as_of for e in experiences)
    now = utcnow()
    model_id = args.model_id or f"meta-{args.algo}-{now.strftime('%Y%m%d-%H%M%S')}"
    artifact_dir = Path(dynamo.settings().artifact_dir) / "models"
    manifest = save_model(
        model,
        artifact_dir,
        model_id,
        train_start=train_start,
        train_end=train_end,
        n_samples=n,
        extra={"metrics": metrics},
    )
    ModelRegistryRepository(dynamo.engine()).insert_candidate(
        ModelVersion(
            id=model_id,
            model_type="meta",
            algo=args.algo,
            features_hash=features_hash(),
            train_start=train_start,
            train_end=train_end,
            hyperparams={"algo": args.algo, "neighbors": args.neighbors, "seed": model.seed},
            metrics={**metrics, "artifact_sha256": manifest.sha256},
            artifact_path=str(artifact_dir / manifest.artifact_file),
            created_at=now,
        )
    )

    print(f"meta-model trained: {model_id} (algo={args.algo})")
    print(f"  experiences: {n} live, window {train_start} .. {train_end}")
    print(f"  artifact: {artifact_dir / manifest.artifact_file} sha256={manifest.sha256[:12]}…")
    print(f"  OOS Brier: {metrics['oos_brier']:.4f} vs base-rate {metrics['base_rate_brier']:.4f}")
    for key in sorted(metrics):
        if key.startswith("precision_conviction_"):
            print(f"  {key}: {metrics[key]:.3f}")
    print(
        "  registered as CANDIDATE (model_type='meta') — production requires the "
        "promotion pipeline + signed human approval; sizing stays dark until "
        "meta_sizing_enabled is flipped."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
