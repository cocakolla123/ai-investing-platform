"""Claude Code CLI adapter implementing the ResearchModel port (ADR-0014).

Routes committee calls through the LOCAL `claude` CLI so nightly research is
covered by the user's flat-rate Claude Code subscription instead of per-token
API billing. Selected via ``Settings.research_backend = "claude_code"``; the
default backend remains the Anthropic API.

CLI contract (verified against the installed binary, ``claude --version`` =
**2.1.219 (Claude Code)**, macOS native build under
``~/Library/Application Support/Claude/claude-code/<ver>/claude.app/Contents/MacOS/claude``):

- ``claude -p --output-format json --model <model>`` — non-interactive print
  mode; emits ONE JSON envelope on stdout with (at least) ``is_error``,
  ``subtype``, ``result`` (the assistant text), and ``usage``
  (``input_tokens``/``output_tokens``/``cache_read_input_tokens``/…).
  The prompt is piped via **stdin** (argv has an OS length ceiling; committee
  contexts do not).
- ``--model`` accepts an alias (``fable``/``opus``/``sonnet``) or a full name
  (``claude-fable-5``); we pass settings values through after trimming.
- ``--system-prompt <prompt>`` — replaces the default system prompt.
- ``--tools ""`` — disables ALL built-in tools (help: 'Use "" to disable all
  tools'); combined with ``--strict-mcp-config`` (with no ``--mcp-config``,
  ignores every configured MCP server) and ``--disable-slash-commands``,
  the invocation is pure text reasoning: no tools, no MCP, no skills.
- ``--no-session-persistence`` — print-mode sessions are not saved to disk.
- The CLI also offers ``--json-schema`` for structured output; we deliberately
  use the same JSON-instruction-in-prompt strategy as ``AnthropicClient``
  (schema appended to the instruction, parse + validate, one retry) so both
  backends share one battle-tested contract.

Design notes:

- ``CONTEXT_CACHE_BOUNDARY`` segments are joined with plain newlines — prompt
  caching is meaningless through the CLI (each call is a fresh process).
- **Batches are a sequential local loop**: ``submit_batch`` executes every
  request immediately (in submission order), stores results in memory, and
  returns a synthetic batch id; ``batch_status`` returns ``"ended"`` so the
  existing sensor -> advance-stage orchestration flow works unchanged.
  Unknown batch ids (e.g. after a process restart) also report ``"ended"``
  with zero results — the advance op resolves them as empty and the next
  nightly run resubmits via deterministic custom_ids, mirroring the batch
  expiry path.
- Plan-window exhaustion (subscription usage limit) maps to
  :class:`PlanLimitExhausted`. With ``claude_code_fallback_to_api=True`` and
  a configured ``AnthropicClient``, the call falls through to the API
  per-call; otherwise the error propagates and the nightly loop stops
  gracefully (same shape as ``BudgetExceededError`` handling).
- Usage is recorded on every successful call with REAL token counts but
  ``cost_usd=0.0`` (flat-rate subscription — not metered per token) and
  ``call_kind="claude_code:<role>"`` so the daily report's cost section shows
  the activity honestly.
- Prompt contents are NEVER logged; error logs carry lengths only.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dynamo.config.settings import Settings
from dynamo.domain.ports import CONTEXT_CACHE_BOUNDARY, ResearchModel
from dynamo.llm.anthropic_client import (
    _FALLBACK_SUFFIX,
    SchemaValidationError,
    _parse_json,
    _required_keys_present,
)
from dynamo.llm.cost_meter import CallTrace, CostMeter

log = logging.getLogger(__name__)

_TIERS = ("screen", "analyst", "arbiter")

# Fallback locations checked after $PATH (the native macOS build does not
# symlink itself onto $PATH; the app-support glob picks the newest version).
_BINARY_FALLBACKS = (
    "~/.claude/local/claude",
    "~/Library/Application Support/Claude/claude-code/*/claude.app/Contents/MacOS/claude",
)

# Subscription plan-window exhaustion, as surfaced in CLI error text.
_PLAN_LIMIT_RE = re.compile(
    r"(usage\s+limit|rate\s+limit|limit\s+(reached|exceeded|will\s+reset|resets)"
    r"|out\s+of\s+(usage|quota)|(5-hour|weekly|session)\s+limit)",
    re.IGNORECASE,
)
# CLI auth failures (expired OAuth, logged out) — terminal for the whole run.
_AUTH_ERROR_RE = re.compile(
    r"failed to authenticate|oauth.*(expired|invalid)|not logged in|please.*log ?in",
    re.IGNORECASE,
)


class PlanLimitExhausted(RuntimeError):
    """Claude Code subscription window exhausted — wait for the next window."""


class ClaudeCodeAuthError(RuntimeError):
    """CLI OAuth/auth failure — terminal, never soft-skipped (fail the run loudly)."""


class ClaudeCodeError(RuntimeError):
    """CLI invocation failed for a non-plan-limit reason (after one retry)."""


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def _parse_envelope(stdout: str) -> dict[str, Any] | None:
    """The CLI's JSON envelope, or None when stdout is not one."""
    try:
        parsed = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tail(text: str, limit: int = 300) -> str:
    """Last ``limit`` chars of CLI output, one line — enough to diagnose."""
    flat = " ".join(text.split())
    return flat[-limit:]


def _raise_for_error_envelope(envelope: dict[str, Any]) -> None:
    """Classify an ``is_error`` envelope into the client's typed errors."""
    subtype = str(envelope.get("subtype", "unknown"))
    body = str(envelope.get("result", ""))
    if _PLAN_LIMIT_RE.search(body) or _PLAN_LIMIT_RE.search(subtype):
        raise PlanLimitExhausted("Claude Code plan window exhausted (CLI reported a usage limit)")
    if _AUTH_ERROR_RE.search(body):
        # Auth failures are terminal for the whole run — every later call
        # fails identically, so soft-skipping would silently void days.
        raise ClaudeCodeAuthError(
            f"claude CLI authentication failed: {body[:200]} "
            "(run the CLI interactively and /login, then retry)"
        )
    raise ClaudeCodeError(f"claude CLI returned error envelope (subtype={subtype}): {body[:200]}")


# runner(cmd, prompt_stdin, timeout_seconds) -> CliResult. Injectable for tests.
CliRunner = Callable[[list[str], str, float], CliResult]


def _subprocess_runner(cmd: list[str], prompt: str, timeout: float) -> CliResult:
    # Fixed argv, shell=False, prompt via stdin — no injection surface.
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        # Neutral cwd: keeps the CLI's CLAUDE.md auto-discovery away from
        # whatever project directory the orchestrator happens to run in.
        cwd=tempfile.gettempdir(),
    )
    return CliResult(proc.returncode, proc.stdout, proc.stderr)


def find_claude_binary(configured: str = "") -> str | None:
    """Resolve the `claude` executable: explicit setting > $PATH > known installs."""
    if configured:
        path = os.path.expanduser(configured)
        return path if os.access(path, os.X_OK) else None
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    for pattern in _BINARY_FALLBACKS:
        matches = sorted(glob.glob(os.path.expanduser(pattern)), reverse=True)
        for candidate in matches:
            if os.access(candidate, os.X_OK):
                return candidate
    return None


class ClaudeCodeResearchModel:
    """Implements dynamo.domain.ports.ResearchModel via the local Claude Code CLI."""

    # submit_batch is a sequential local loop and its results live ONLY in
    # this instance's memory (no durable batch store). A collector in another
    # process — e.g. Dagster's sensor-triggered advance job, which runs in a
    # fresh run worker — sees batch_status "ended" with ZERO results and
    # resolves the batch empty, silently discarding every call the submitter
    # already paid for. Orchestration reads this flag and keeps submission
    # and collection in ONE process (the sync deep-dive path) instead.
    executes_batches_in_process = True

    def __init__(
        self,
        settings: Settings,
        cost_meter: CostMeter,
        *,
        binary: str,
        api_fallback: ResearchModel | None = None,
        runner: CliRunner | None = None,
    ) -> None:
        self._meter = cost_meter
        self._binary = binary
        self._runner: CliRunner = runner if runner is not None else _subprocess_runner
        self._timeout = float(settings.claude_code_timeout_seconds)
        self._fallback_enabled = settings.claude_code_fallback_to_api
        self._api_fallback = api_fallback
        self._tier_models: dict[str, str] = {
            "screen": settings.claude_code_model_screen.strip(),
            "analyst": settings.claude_code_model_analyst.strip(),
            "arbiter": settings.claude_code_model_arbiter.strip(),
        }
        # One-hop degradation on a model-specific window exhaustion; only the
        # arbiter has one (Fable's weekly allowance is separate from the
        # shared pool — "if fable runs out then go to opus").
        self._tier_fallbacks: dict[str, str] = {
            "arbiter": settings.claude_code_arbiter_fallback.strip(),
        }
        # batch_id -> ordered results, produced eagerly by submit_batch.
        self._batch_results: dict[str, list[tuple[str, dict[str, Any] | None]]] = {}

    # --- helpers ---

    def _model_for(self, model_tier: str) -> str:
        if model_tier not in _TIERS:
            raise ValueError(f"unknown model_tier {model_tier!r}; expected one of {_TIERS}")
        return self._tier_models[model_tier]

    def _build_command(self, *, model: str, system: str) -> list[str]:
        return [
            self._binary,
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]

    @staticmethod
    def _build_prompt(*, context: str, instruction: str, schema: dict[str, Any]) -> str:
        # No prompt cache through the CLI -> boundary markers are just joined.
        flat_context = "\n\n".join(seg for seg in context.split(CONTEXT_CACHE_BOUNDARY) if seg)
        suffixed = instruction + _FALLBACK_SUFFIX + json.dumps(schema, sort_keys=True)
        return f"{flat_context}\n\n{suffixed}" if flat_context else suffixed

    def _invoke(self, cmd: list[str], prompt: str) -> dict[str, Any]:
        """One CLI call -> parsed JSON envelope. Raises on error.

        A hung CLI subprocess gets ONE retry. Measured: a single 300s timeout
        on a 784-char triage prompt killed a 115-day committee replay 8 days
        in — a transient local hang, not a plan limit (those raise
        PlanLimitExhausted from the CLI's own output and must not be
        retried). Two hangs in a row is a real environment problem and stays
        terminal.
        """
        try:
            result = self._runner(cmd, prompt, self._timeout)
        except subprocess.TimeoutExpired:
            log.warning(
                "claude CLI timed out after %.0fs (prompt %d chars) — retrying once",
                self._timeout,
                len(prompt),
            )
            try:
                result = self._runner(cmd, prompt, self._timeout)
            except subprocess.TimeoutExpired as exc:
                raise ClaudeCodeError(
                    f"claude CLI timed out twice at {self._timeout:.0f}s "
                    f"(prompt {len(prompt)} chars)"
                ) from exc
        if result.returncode != 0:
            # Only inspect error text on FAILED invocations — a successful
            # model response may legitimately mention limits.
            if _PLAN_LIMIT_RE.search(f"{result.stdout}\n{result.stderr}"):
                raise PlanLimitExhausted(
                    "Claude Code plan window exhausted (CLI reported a usage limit)"
                )
            # With --output-format json the CLI reports its own failures (auth,
            # unsupported model, API errors) as an is_error envelope on stdout
            # AND exits 1. Checking the exit code first used to discard that
            # envelope, so every such failure surfaced as a bare "exited 1" —
            # the 2026-09-16 committee failure is undiagnosable for exactly
            # this reason — and the auth branch below could never fire for it.
            failed = _parse_envelope(result.stdout)
            if failed is not None and failed.get("is_error"):
                _raise_for_error_envelope(failed)
            detail = _tail(result.stderr) or _tail(result.stdout)
            log.error(
                "claude CLI exited %d (prompt %d chars): %s",
                result.returncode,
                len(prompt),
                detail,
            )
            raise ClaudeCodeError(
                f"claude CLI exited {result.returncode}" + (f": {detail}" if detail else "")
            )
        try:
            envelope = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ClaudeCodeError(
                f"claude CLI emitted a non-JSON envelope ({len(result.stdout)} chars)"
            ) from exc
        if not isinstance(envelope, dict):
            raise ClaudeCodeError("claude CLI envelope was not a JSON object")
        if envelope.get("is_error"):
            _raise_for_error_envelope(envelope)
        return envelope

    def _record(
        self, envelope: dict[str, Any], *, role: str, model: str, trace: CallTrace | None = None
    ) -> None:
        """Real token counts, cost 0.0 — subscription is flat-rate, not metered."""
        usage = envelope.get("usage") or {}
        self._meter.record(
            model_id=model,
            call_kind=f"claude_code:{role}",
            input_tokens=int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("cache_creation_input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cost_usd=0.0,
            batch=False,
            trace=trace,
        )

    def _complete_once(
        self,
        *,
        role: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        model_tier: str,
        max_tokens: int,
        attempt: int = 1,
        trace: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """One attempt: run the CLI, record usage, parse/validate. None = invalid output.

        The ledger row is written after parsing so it can say whether this
        attempt's output was usable; ``trace`` supplies what only the caller
        knows (role, request id, retrieved journal ids).

        ``max_tokens`` is accepted for port parity; the CLI exposes no
        per-call output-token ceiling, so it is intentionally unused.
        """
        del max_tokens
        model = self._model_for(model_tier)
        self._meter.check_budget()  # kill-switch parity; $0 entries never trip it
        cmd = self._build_command(model=model, system=system)
        prompt = self._build_prompt(context=context, instruction=instruction, schema=schema)
        started = time.perf_counter()
        envelope = self._invoke(cmd, prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed = _parse_json(str(envelope.get("result", "")))
        ok = parsed is not None and _required_keys_present(parsed, schema)
        extra = trace or {}
        equiv = envelope.get("total_cost_usd")
        self._record(
            envelope,
            role=role,
            model=model,
            trace=CallTrace(
                role=str(extra.get("role") or role),
                request_id=extra.get("request_id"),
                prompt_hash=CallTrace.hash_prompt(system),
                retrieved_ids=CallTrace.ids(extra.get("retrieved_ids")),
                latency_ms=latency_ms,
                attempt=attempt,
                validation_ok=ok,
                equiv_cost_usd=float(equiv) if isinstance(equiv, int | float) else None,
            ),
        )
        return parsed if ok else None

    # --- ResearchModel port ---

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        model_tier: str,
        max_tokens: int = 4096,
        trace: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``trace`` (optional, this client only) carries ledger-trace fields
        the caller knows: ``role``, ``request_id``, ``retrieved_ids``."""
        last_error: ClaudeCodeError | None = None
        for attempt in range(2):  # one retry on invalid output or CLI failure
            try:
                parsed = self._complete_once(
                    role=role,
                    system=system,
                    context=context,
                    instruction=instruction,
                    schema=schema,
                    model_tier=model_tier,
                    max_tokens=max_tokens,
                    attempt=attempt + 1,
                    trace=trace,
                )
            except PlanLimitExhausted:
                # Model-specific window fallback (user directive 2026-09-14:
                # "if fable runs out then go to opus"). Fable carries its own
                # weekly allowance separate from the shared pool, so the
                # arbiter tier — and only that tier — degrades one step to
                # Opus instead of soft-failing the whole night. One hop, no
                # chains: if Opus is also exhausted, the normal soft-fail
                # path takes over and the night resumes tomorrow.
                fallback = self._tier_fallbacks.get(model_tier)
                if fallback and self._tier_models.get(model_tier) != fallback:
                    log.warning(
                        "claude_code: %s window exhausted for tier %r — retrying on %s",
                        self._tier_models.get(model_tier),
                        model_tier,
                        fallback,
                    )
                    degraded = dict(self._tier_models)
                    degraded[model_tier] = fallback
                    self._tier_models = degraded  # sticky for this process/night
                    continue
                if self._fallback_enabled and self._api_fallback is not None:
                    log.warning(
                        "claude_code: plan window exhausted — falling back to the API for role %r",
                        role,
                    )
                    return self._api_fallback.complete_json(
                        role=role,
                        system=system,
                        context=context,
                        instruction=instruction,
                        schema=schema,
                        model_tier=model_tier,
                        max_tokens=max_tokens,
                    )
                raise
            except ClaudeCodeError as exc:
                last_error = exc
                continue
            last_error = None  # this attempt reached the model; failure = bad output
            if parsed is not None:
                return parsed
        if last_error is not None:
            raise last_error
        model = self._model_for(model_tier)
        raise SchemaValidationError(
            f"role {role!r} ({model}): output failed schema validation after retry"
        )

    def submit_batch(self, requests: Sequence[dict[str, Any]]) -> str:
        """Sequential local loop — no Message Batches API through the CLI.

        Executes every request immediately (submission order). A mid-batch
        :class:`PlanLimitExhausted` (with no API fallback) marks the remaining
        requests as failed (None) rather than raising: the quarantine path
        absorbs them and the next nightly run resubmits via deterministic
        custom_ids — the same recovery as an expired API batch.
        """
        batch_id = f"claude-code-local-{uuid.uuid4().hex[:12]}"
        results: list[tuple[str, dict[str, Any] | None]] = []
        exhausted = False
        for req in requests:
            custom_id = str(req["custom_id"])
            if exhausted:
                results.append((custom_id, None))
                continue
            try:
                parsed: dict[str, Any] | None = self.complete_json(
                    role="batch",
                    system=req["system"],
                    context=req["context"],
                    instruction=req["instruction"],
                    schema=req["schema"],
                    model_tier=req["model_tier"],
                    max_tokens=int(req.get("max_tokens", 4096)),
                    trace={
                        "role": req.get("role"),
                        "request_id": custom_id,
                        "retrieved_ids": req.get("retrieved_ids"),
                    },
                )
            except PlanLimitExhausted:
                log.warning(
                    "claude_code: plan window exhausted mid-batch %s — remaining "
                    "requests marked failed for deterministic resubmission",
                    batch_id,
                )
                exhausted = True
                parsed = None
            except (ClaudeCodeError, SchemaValidationError):
                parsed = None  # caller quarantines and resubmits next run
            results.append((custom_id, parsed))
        self._batch_results[batch_id] = results
        return batch_id

    def batch_status(self, batch_id: str) -> str:
        # Always terminal: submit_batch already ran everything. Unknown ids
        # (process restart) are also "ended" — batch_results then yields
        # nothing and the advance op resolves the batch as empty.
        return "ended"

    def batch_results(self, batch_id: str) -> Iterable[tuple[str, dict[str, Any] | None]]:
        return self._iter_batch_results(batch_id)

    def _iter_batch_results(self, batch_id: str) -> Iterator[tuple[str, dict[str, Any] | None]]:
        yield from self._batch_results.get(batch_id, [])
