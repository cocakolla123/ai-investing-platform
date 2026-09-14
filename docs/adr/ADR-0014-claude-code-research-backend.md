# ADR-0014 — Claude Code research backend (subscription-routed committee, Fable arbiter)

Status: accepted. Date: 2026-08-18.

## Context

Nightly paper-phase research runs ~35 committee calls (screen/analyst/arbiter
tiers) through the Anthropic API at per-token rates, metered against
`CLAUDE_MONTHLY_BUDGET_USD`. The user also pays for a Claude Code
subscription — flat-rate, with the `claude` CLI installed and authenticated on
the host (native macOS build, verified `claude --version` = 2.1.219). Routing
committee calls through the local CLI makes the nightly research **$0
marginal cost** and, because subscription pricing is flat, lets the strongest
available model (Fable 5, `claude-fable-5`) take the arbiter seat at no
premium over Sonnet.

## Decision

- New adapter `src/dynamo/llm/claude_code_client.py`
  (`ClaudeCodeResearchModel`) implements the `ResearchModel` port by shelling
  out to `claude -p --output-format json --model <model>`, prompt via stdin,
  system prompt via `--system-prompt`. Tool use is hard-disabled
  (`--tools ""` + `--strict-mcp-config` + `--disable-slash-commands`): the
  committee is pure text reasoning, no tools or network from inside the call.
- Selected via `Settings.research_backend = "claude_code"`; the **default
  stays `"api"`** and that path is byte-identical to before. Missing CLI
  binary degrades to quant-only mode with a loud log, same as a missing API
  key.
- Tier models come from `claude_code_model_screen/analyst/arbiter`
  (defaults `claude-haiku-4-5` / `claude-sonnet-5` / `claude-fable-5`).
- JSON enforcement reuses the AnthropicClient fallback strategy verbatim:
  schema appended to the instruction, parse + required-keys validation, one
  retry, then `SchemaValidationError`. (The CLI's `--json-schema` structured
  output exists but is deliberately unused so both backends share one
  contract.)
- **Batch methods are a sequential local loop**: `submit_batch` executes all
  requests immediately, returns a synthetic batch id; `batch_status` is
  always `"ended"`; `batch_results` yields `(custom_id, parsed|None)`. The
  sensor → advance-stage orchestration flow runs unchanged; unknown ids
  (process restart) resolve as empty and deterministic custom_ids resubmit
  next night — the same recovery as an expired API batch.
- Plan-window exhaustion maps to `PlanLimitExhausted`. Default
  (`claude_code_fallback_to_api=False`): the error propagates and the nightly
  loop stops gracefully until the next window (the established
  `BudgetExceededError` pattern). With the flag on and `ANTHROPIC_API_KEY`
  set, each exhausted call falls through to the API instead.
- Usage recording: real token counts from the CLI envelope, `cost_usd=0.0`
  (flat-rate — not metered per token), `call_kind="claude_code:<role>"`, so
  the daily cost report shows the activity honestly. `claude-fable-5` is
  registered in the pricing table with zero rates and its documented
  training-data cutoff (Jan 2026 → 2026-01-31, conservative month-end rule).

## Tradeoffs

- **ToS note**: this is personal-use automation within the user's own
  subscription limits — the platform runs on the user's machine, under the
  user's authenticated CLI, for the user's own research. It does not resell
  access or evade metering; when the plan window is exhausted it stops.
- **Plan windows**: subscription usage caps (5-hourly/weekly) can exhaust
  mid-run; the API backend has no such cliff.
- **No batch discount, no prompt caching**: batches degrade to a sequential
  loop at full (flat) rate; `CONTEXT_CACHE_BOUNDARY` segments are joined with
  newlines since each CLI call is a fresh process.
- **JSON enforcement is prompt-level only** (no server-side structured
  output), mitigated by validation + one retry + the quarantine path.
- **Scale ceiling**: right-sized for ~35 nightly calls; wrong for large
  replay campaigns — keep `research_backend="api"` for those.

## Addendum (2026-09-13): batches never cross a process boundary

The original decision claimed the sensor -> advance-stage flow "runs
unchanged" over the synthetic local batches. Under the Dagster daemon that
was wrong in a way hand-run CLIs never exposed: `submit_batch` keeps its
results only in the submitting process's memory, while the sensor-triggered
`committee_advance_job` executes in a **separate run worker process**. Its
fresh model instance reports the batch "ended" with zero results, the
advance op resolves it empty, and every call the submitter already executed
is discarded — then the next night's deterministic custom_ids resubmit and
lose it all again. With `deep_dive_cap_per_night=8` above the sync
threshold (3), the scheduled committee would burn ~32 CLI calls a night and
never persist a verdict.

Fix: the client declares `executes_batches_in_process = True`, and
`submit_committee_deep_dives` routes such models through the sync deep-dive
path for ANY set size, so submission and persistence share one process. The
API backend (real Message Batches, durable server-side) keeps the batch
flow unchanged.

## Replay implication (documentation only — no replay code changed)

The committee-replay knowledge gate reads `ModelSpec.knowledge_cutoff` plus
`replay_cutoff_margin_days` (30). With the API backend's default arbiter
(`claude-opus-5`, cutoff 2026-05-31) a clean replay window cannot start
before **2026-07-01**. With this backend's tiers (Fable 5 and Sonnet 5 both
cutoff 2026-01-31; Haiku 4.5 earlier) the binding cutoff is 2026-01-31, so a
clean committee-replay window can start **2026-03-02**. Registering
`claude-fable-5` in the pricing/spec table is what makes the existing gate
handle this automatically — nothing in `src/dynamo/replay/` changes.
