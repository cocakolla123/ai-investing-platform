# Approval artifacts

Two gates in this platform refuse to act without a **signed human approval
artifact**, and both verify it with the *same* routine
(`dynamo.execution.gate.check_approval`, HMAC-SHA256 +
`hmac.compare_digest`):

| Gate | Where | What it blocks |
|---|---|---|
| Live-trading gate (doc 09 §7) | `live_orders_permitted()` | every LIVE order submission |
| Promotion stage 7 (doc 08 §7) | `python -m dynamo.cli.promotion run --stage 7` | `shadow -> production` |

An artifact is a YAML file whose signature covers **every key except
`signature` itself**, canonicalized by `canonical_approval_body()`
(`json.dumps(..., sort_keys=True, separators=(",", ":"))`). Editing any field
after signing invalidates it.

```yaml
approver: owner
expiry: '2026-12-31'
issued_at: '2026-08-19T18:04:11.902184+00:00'
kind: promotion
note: reviewed backtest bundle run-123
scope:
- my-strategy
target_id: my-strategy
signature: 8b1f...   # hex hmac-sha256
```

## The signing key

The key lives in the environment variable `DYNAMO_APPROVAL_KEY`. **It is
deliberately not set anywhere in this repository** — not in `.env`, not in
`.env.example`, not in `docker-compose.yml`. If it were, the artifact and the
means of forging it would sit in the same place and the gate would be theatre.

Generate one once:

```console
$ python -m dynamo.cli.approve keygen
3f2a...            # 64 hex chars on stdout
Export it (outside the repo working tree):
    export DYNAMO_APPROVAL_KEY=<the key above>
```

Put it in your password manager and export it from your shell profile (or a
file outside the working tree that you `source`). Rotating it invalidates
every artifact ever signed — that is the intended failure mode, not a bug.

## Signing an approval

```console
$ export DYNAMO_APPROVAL_KEY=...
$ python -m dynamo.cli.approve sign \
      --kind promotion --target my-strategy \
      --expiry 2026-12-31 --approver owner \
      --note "reviewed artifacts/backtests/run-123"
wrote approvals/promotion-my-strategy.yaml
  signature 8b1f...

$ python -m dynamo.cli.approve verify \
      --file approvals/promotion-my-strategy.yaml --target my-strategy
OK: approvals/promotion-my-strategy.yaml verified (expiry 2026-12-31, scope ('my-strategy',))
```

For the live (real-money) gate use `--kind live --scope all` (or a comma-separated list of
symbols/sleeves) and a **short** expiry — the artifact is a dated permission,
not a standing one.

## Failure modes, and what they look like

Every refusal is loud and names the reason. None of them can be overridden;
there is no `--force`.

| Situation | Refusal |
|---|---|
| `DYNAMO_APPROVAL_KEY` unset | `approval_bad_signature` + the four-step "how to generate a key" instructions |
| File missing | `no such approval artifact: <path>` |
| Signature does not match the key | `approval_bad_signature: HMAC mismatch ...` |
| Any field edited after signing | same — the body is what is signed |
| `expiry` in the past | `approval_expired` |
| `--target` not in `scope` | `approval_out_of_scope` |
| No `--approval` passed to stage 7 | `shadow -> production for <id> requires a signed approval artifact` |

## What is committed

Nothing in this directory except this README. Signed artifacts are gitignored:
an approval is an operational act with an expiry, not source code, and
committing one would let a stale artifact ride along with a checkout.
