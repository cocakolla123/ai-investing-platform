"""Generational training CLI (plan Phase 5) — manual only, NEVER scheduled.

Usage:
    python -m dynamo.cli.generations [--database-url URL] [--algo gradient_boosting]
        [--final-holdout-months 1] [--include-replay replay-abc123 ...]

ZERO LLM calls. Loads resolved LIVE experiences (replay rows enter ONLY via
explicit ``--include-replay <run_id>`` — off by default, the sole gate),
claims the final holdout period in ``evaluation_claims`` (already claimed ⇒
REFUSE: the holdout is evaluated once, ever), runs the expanding monthly
generations against the frozen control, writes ``generation_results``, saves
per-generation artifacts, and registers the FINAL trained generation as a
registry CANDIDATE (``model_type='meta'``) — production status remains
reachable only through the promotion pipeline + signed human approval.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-generations",
        description="Generational meta-model training vs the frozen control",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--algo",
        default="gradient_boosting",
        choices=("gradient_boosting", "logistic"),
    )
    parser.add_argument(
        "--final-holdout-months",
        type=int,
        default=1,
        help="months excluded from EVERY fold and claimed in evaluation_claims",
    )
    parser.add_argument(
        "--include-replay",
        action="append",
        default=None,
        metavar="RUN_ID",
        help=(
            "opt-in replay experiences by run id (repeatable). The ONLY way "
            "replay rows enter training — off by default."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    from orchestration.resources import DynamoResource, utcnow

    from dynamo.backtest.walkforward import compute_config_hash
    from dynamo.domain.models.learning import ModelVersion
    from dynamo.learning.artifacts import save_model
    from dynamo.learning.generations import (
        final_holdout_period,
        generation_folds,
        run_generation,
    )
    from dynamo.learning.meta_model import features_hash
    from dynamo.persistence.repositories.learning import ModelRegistryRepository
    from dynamo.persistence.repositories.replay import GenerationsRepository

    args = _build_parser().parse_args(argv)
    include_replay = sorted(set(args.include_replay or []))

    dynamo = DynamoResource(database_url=args.database_url or "")
    exp_repo = dynamo.experience_repo()

    experiences = list(exp_repo.resolved_live())
    n_live = len(experiences)
    for run_id in include_replay:
        replay_rows = [e for e in exp_repo.by_run(run_id) if e.source == "replay"]
        if not replay_rows:
            print(f"warning: --include-replay {run_id}: no replay experiences found")
        experiences.extend(replay_rows)
    print(
        f"experiences: {n_live} live"
        + (f" + {len(experiences) - n_live} replay ({', '.join(include_replay)})"
           if include_replay else " (replay excluded — pass --include-replay to opt in)")
    )

    folds = generation_folds(experiences, final_holdout_months=args.final_holdout_months)
    if not folds:
        print(
            "refused: not enough monthly history for a single generation "
            f"(need >= {args.final_holdout_months + 2} months of resolved experiences)."
        )
        return 2

    config = {
        "algo": args.algo,
        "final_holdout_months": args.final_holdout_months,
        "include_replay": include_replay,
        "features_hash": features_hash(),
    }
    config_hash = compute_config_hash(config)
    gen_repo = GenerationsRepository(dynamo.engine())

    holdout = final_holdout_period(
        experiences, final_holdout_months=args.final_holdout_months
    )
    if holdout is not None:
        start, end = holdout
        existing = gen_repo.is_claimed(start, end, "final_holdout")
        if existing is not None:
            print(
                f"refused: final holdout {start} .. {end} already claimed "
                f"(config {str(existing['config_hash'])[:12]} at {existing['claimed_at']}). "
                "The holdout is evaluated once, ever — accumulate a new month "
                "or restart the protocol."
            )
            return 2
        if not gen_repo.claim_period(
            period_start=start,
            period_end=end,
            purpose="final_holdout",
            config_hash=config_hash,
            claimed_at=utcnow(),
        ):
            print("refused: lost the claim race for the final holdout period.")
            return 2
        print(f"claimed final holdout {start} .. {end} (excluded from every fold)")

    artifact_dir = Path(dynamo.settings().artifact_dir) / "models"
    now = utcnow()
    outcomes = []
    final_model = None
    final_outcome = None
    for fold in folds:
        outcome = run_generation(fold, algo=args.algo)
        outcomes.append(outcome)
        row = {
            "id": f"gen:{config_hash[:12]}:{outcome.generation:03d}",
            "created_at": now,
            "generation": outcome.generation,
            "train_start": outcome.train_start,
            "train_end": outcome.train_end,
            "test_start": outcome.test_start,
            "test_end": outcome.test_end,
            "model_version": outcome.model_version,
            "n_train": outcome.n_train,
            "n_test": outcome.n_test,
            "brier_meta": outcome.brier_meta,
            "brier_control": outcome.brier_control,
            "precision_meta": outcome.precision_meta,
            "precision_control": outcome.precision_control,
            "ev_meta": outcome.ev_meta,
            "ev_control": outcome.ev_control,
            "value_of_learning": outcome.value_of_learning,
            "payload": {
                "config": config,
                "config_hash": config_hash,
                "skipped": outcome.skipped,
                "skip_reason": outcome.skip_reason,
            },
        }
        gen_repo.insert_generation(row)
        if outcome.model is not None:
            model_id = f"{outcome.model_version}-{config_hash[:8]}"
            save_model(
                outcome.model,
                artifact_dir,
                model_id,
                train_start=outcome.train_start,
                train_end=outcome.train_end,
                n_samples=outcome.n_train,
                extra={
                    "generation": outcome.generation,
                    "brier_meta": outcome.brier_meta,
                    "brier_control": outcome.brier_control,
                    "value_of_learning": outcome.value_of_learning,
                    "config_hash": config_hash,
                },
            )
            final_model, final_outcome = outcome.model, outcome

    # Generation table.
    print(
        f"{'gen':>3}  {'train':<23} {'test':<23} {'n_tr':>5} {'n_te':>5} "
        f"{'brier_meta':>10} {'brier_ctl':>10} {'VoL':>8}  status"
    )
    for o in outcomes:
        if o.skipped:
            metrics = f"{'-':>10} {'-':>10} {'-':>8}  skipped: {o.skip_reason}"
        else:
            metrics = (
                f"{o.brier_meta:>10.4f} {o.brier_control:>10.4f} "
                f"{o.value_of_learning:>+8.4f}  trained"
            )
        print(
            f"{o.generation:>3}  {o.train_start} .. {o.train_end} "
            f"{o.test_start} .. {o.test_end} {o.n_train:>5} {o.n_test:>5} {metrics}"
        )
    print("VoL = brier_control - brier_meta; POSITIVE means learning beats the control.")

    if final_model is not None and final_outcome is not None:
        model_id = f"{final_outcome.model_version}-{config_hash[:8]}"
        ModelRegistryRepository(dynamo.engine()).insert_candidate(
            ModelVersion(
                id=model_id,
                model_type="meta",
                algo=args.algo,
                features_hash=features_hash(),
                train_start=final_outcome.train_start,
                train_end=final_outcome.train_end,
                hyperparams={"algo": args.algo, "config_hash": config_hash},
                metrics={
                    "brier_meta": final_outcome.brier_meta,
                    "brier_control": final_outcome.brier_control,
                    "value_of_learning": final_outcome.value_of_learning,
                },
                artifact_path=str(artifact_dir / f"{model_id}.joblib"),
                created_at=now,
            )
        )
        print(
            f"registered {model_id} as CANDIDATE (model_type='meta') — production "
            "requires the promotion pipeline + signed human approval."
        )
    else:
        print("no generation reached the training gate — nothing registered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
