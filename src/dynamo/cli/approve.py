"""Approval-artifact tool: generate the key, sign artifacts, verify them.

This is the tool the promotion CLI's stage-7 gate and the live gate both
verify against (``dynamo.execution.gate.check_approval``). Both consume the
same YAML shape:

.. code-block:: yaml

    kind: promotion            # or: live
    target_id: my-strategy     # promotion only; must equal --target
    scope: [my-strategy]       # symbols/sleeves, or [all]
    expiry: 2026-12-31
    approver: owner
    issued_at: 2026-08-19T12:00:00+00:00
    note: "reviewed backtest bundle run-123"
    signature: <hex hmac-sha256 of the canonical body>

The signature covers *every key except* ``signature`` itself, canonicalized
by :func:`dynamo.execution.gate.canonical_approval_body`. Editing any field
after signing invalidates it.

Usage::

    python -m dynamo.cli.approve keygen
    python -m dynamo.cli.approve sign --kind promotion --target my-strategy \
        --expiry 2026-12-31 --approver owner --note "bundle run-123"
    python -m dynamo.cli.approve verify --file approvals/promotion-my-strategy.yaml \
        --target my-strategy

The signing key lives in ``$DYNAMO_APPROVAL_KEY`` and MUST NOT live in the
repo (not in ``.env`` inside the working tree either — see
``docs/runbooks/kill-switch.md``). Losing it means every existing artifact
stops verifying, which is the intended failure mode.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-approve",
        description="Generate, sign and verify Dynamo approval artifacts (HMAC-SHA256)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("keygen", help="print a fresh signing key (never written to disk)")

    sign = sub.add_parser("sign", help="write a signed approval artifact")
    sign.add_argument("--kind", default="promotion", choices=("promotion", "live"))
    sign.add_argument("--target", required=True, help="registry target id / approval subject")
    sign.add_argument(
        "--scope",
        default=None,
        help="comma-separated scope (default: the target; use 'all' for everything)",
    )
    sign.add_argument("--expiry", required=True, help="YYYY-MM-DD, inclusive")
    sign.add_argument("--approver", required=True, help="human who is approving")
    sign.add_argument("--note", default="", help="what was reviewed")
    sign.add_argument(
        "--out", default=None, help="output path (default approvals/<kind>-<target>.yaml)"
    )
    sign.add_argument(
        "--force", action="store_true", help="overwrite an existing artifact at --out"
    )

    verify = sub.add_parser("verify", help="verify an artifact against the current key")
    verify.add_argument("--file", required=True)
    verify.add_argument("--target", default=None, help="scope target to check coverage for")
    verify.add_argument("--today", default=None, help="YYYY-MM-DD override for the expiry check")
    return parser


def main(argv: list[str] | None = None) -> int:
    import secrets
    from datetime import UTC, date, datetime
    from pathlib import Path

    from dynamo.config.settings import get_settings
    from dynamo.execution.gate import (
        APPROVAL_KEY_ENV,
        MISSING_KEY_HELP,
        approval_key_from_env,
        check_approval,
        load_approval,
        sign_approval_body,
    )

    args = _build_parser().parse_args(argv)

    if args.command == "keygen":
        print(secrets.token_hex(32))
        print(
            f"\nExport it (outside the repo working tree):\n"
            f"    export {APPROVAL_KEY_ENV}=<the key above>\n"
            "Store a copy in your password manager. It is NOT recoverable, and\n"
            "every artifact signed with the old key stops verifying if you rotate.",
            file=sys.stderr,
        )
        return 0

    key = approval_key_from_env()
    if key is None:
        print(f"refused: {MISSING_KEY_HELP}", file=sys.stderr)
        return 2

    if args.command == "sign":
        scope = (
            tuple(s.strip() for s in args.scope.split(",") if s.strip())
            if args.scope
            else (args.target,)
        )
        body: dict[str, object] = {
            "kind": args.kind,
            "target_id": args.target,
            "scope": list(scope),
            "expiry": date.fromisoformat(args.expiry).isoformat(),
            "approver": args.approver,
            "issued_at": datetime.now(UTC).isoformat(),
        }
        if args.note:
            body["note"] = args.note
        signature = sign_approval_body(body, key)

        out = Path(
            args.out
            if args.out
            else Path(get_settings().approvals_dir) / f"{args.kind}-{args.target}.yaml"
        )
        if out.exists() and not args.force:
            print(f"refused: {out} already exists (pass --force to replace)", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)

        import yaml

        out.write_text(
            yaml.safe_dump({**body, "signature": signature}, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {out}")
        print(f"  signature {signature}")
        return 0

    # verify
    path = Path(args.file)
    if not path.exists():
        print(f"refused: no such approval artifact: {path}", file=sys.stderr)
        return 2
    artifact = load_approval(path)
    today = date.fromisoformat(args.today) if args.today else date.today()
    result = check_approval(artifact, key=key, today=today, scope_target=args.target)
    if result.permitted:
        print(f"OK: {path} verified (expiry {artifact.expiry.isoformat()}, scope {artifact.scope})")
        return 0
    print(f"refused ({result.reason.value}): {result.detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
