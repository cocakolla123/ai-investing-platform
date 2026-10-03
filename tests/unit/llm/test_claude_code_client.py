"""ClaudeCodeResearchModel: CLI contract, JSON parsing, plan limits, batches.

The CLI is never executed — a fake runner captures argv/stdin and returns
canned envelopes. No network, no real `claude -p` model invocations.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from dynamo.config.settings import Settings
from dynamo.domain.ports import CONTEXT_CACHE_BOUNDARY
from dynamo.llm.anthropic_client import SchemaValidationError
from dynamo.llm.claude_code_client import (
    ClaudeCodeError,
    ClaudeCodeResearchModel,
    CliResult,
    PlanLimitExhausted,
    find_claude_binary,
)
from dynamo.llm.cost_meter import CostMeter, InMemoryLedgerWriter

SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
GOOD = {"answer": "42"}


def isolated_settings(**overrides: Any) -> Settings:
    """``Settings`` built from explicit kwargs only — no ambient environment.

    Two separate leaks have to be closed, and they need different tools:

    - The repo's ``.env`` sets ``RESEARCH_BACKEND=claude_code`` (ADR-0014, so
      the nightly pipeline cannot silently bill the metered API). ``Settings``
      declares ``env_file=".env"``, so a bare ``Settings()`` run from inside
      the repo picks that up. ``_env_file=None`` drops the dotenv source.
    - An *exported* ``RESEARCH_BACKEND`` is read from ``os.environ`` by the
      env source regardless of ``_env_file``. Only an explicit init kwarg
      outranks it (init > env > dotenv in pydantic-settings' precedence).

    So a test asserting on a *default* passes that default explicitly. The
    value under test stays visible at the call site instead of depending on
    whatever the developer happens to have exported.
    """
    kwargs: dict[str, Any] = {"RESEARCH_BACKEND": "api"}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def envelope(
    payload: Any,
    *,
    is_error: bool = False,
    subtype: str = "success",
    usage: dict[str, int] | None = None,
) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "is_error": is_error,
            "result": text,
            "usage": usage
            if usage is not None
            else {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "session_id": "test-session",
            "total_cost_usd": 0.0123,
            "num_turns": 1,
        }
    )


class FakeRunner:
    """Captures every invocation; pops canned CliResults (or raises them)."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[tuple[list[str], str, float]] = []

    def __call__(self, cmd: list[str], prompt: str, timeout: float) -> CliResult:
        self.calls.append((cmd, prompt, timeout))
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def ok(stdout: str) -> CliResult:
    return CliResult(0, stdout, "")


def make_settings(**overrides: Any) -> Settings:
    return isolated_settings(ANTHROPIC_API_KEY="", **overrides)


def make_client(
    runner: FakeRunner,
    *,
    settings: Settings | None = None,
    api_fallback: Any | None = None,
) -> tuple[ClaudeCodeResearchModel, InMemoryLedgerWriter]:
    writer = InMemoryLedgerWriter()
    meter = CostMeter(writer, 90.0)
    client = ClaudeCodeResearchModel(
        settings if settings is not None else make_settings(),
        meter,
        binary="/fake/claude",
        api_fallback=api_fallback,
        runner=runner,
    )
    return client, writer


def complete(client: ClaudeCodeResearchModel, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "role": "dossier",
        "system": "SYSTEM",
        "context": "CONTEXT",
        "instruction": "INSTRUCTION",
        "schema": SCHEMA,
        "model_tier": "analyst",
    }
    kwargs.update(overrides)
    return client.complete_json(**kwargs)


class TestCommandLine:
    def test_tier_mapping_and_model_flag(self) -> None:
        for tier, model in [
            ("screen", "claude-haiku-4-5"),
            ("analyst", "claude-sonnet-5"),
            ("arbiter", "claude-fable-5"),
        ]:
            runner = FakeRunner([ok(envelope(GOOD))])
            client, _ = make_client(runner)
            complete(client, model_tier=tier)
            cmd = runner.calls[0][0]
            assert cmd[cmd.index("--model") + 1] == model

    def test_unknown_tier_raises(self) -> None:
        client, _ = make_client(FakeRunner([]))
        with pytest.raises(ValueError, match="unknown model_tier"):
            complete(client, model_tier="bogus")

    def test_tool_disallow_and_print_flags_present(self) -> None:
        runner = FakeRunner([ok(envelope(GOOD))])
        client, _ = make_client(runner)
        complete(client)
        cmd = runner.calls[0][0]
        assert cmd[0] == "/fake/claude"
        assert "-p" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert cmd[cmd.index("--tools") + 1] == ""  # all built-in tools disabled
        assert "--strict-mcp-config" in cmd  # no MCP servers
        assert "--disable-slash-commands" in cmd  # no skills
        assert "--no-session-persistence" in cmd

    def test_system_prompt_flag_and_prompt_assembly(self) -> None:
        runner = FakeRunner([ok(envelope(GOOD))])
        client, _ = make_client(runner)
        context = f"BASE{CONTEXT_CACHE_BOUNDARY}DOSSIER"
        complete(client, context=context)
        cmd, prompt, timeout = runner.calls[0]
        assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM"
        assert "SYSTEM" not in prompt  # system rides the flag, not stdin
        # Cache boundary joined with plain newlines; instruction + schema suffix.
        assert CONTEXT_CACHE_BOUNDARY not in prompt
        assert prompt.startswith("BASE\n\nDOSSIER")
        assert "INSTRUCTION" in prompt
        assert "ONLY a single valid JSON object" in prompt
        assert json.dumps(SCHEMA, sort_keys=True) in prompt
        assert timeout == 300.0


class TestParsing:
    def test_happy_path_extracts_result_text(self) -> None:
        runner = FakeRunner([ok(envelope(GOOD))])
        client, _ = make_client(runner)
        assert complete(client) == GOOD

    def test_code_fenced_json_is_parsed(self) -> None:
        fenced = "```json\n" + json.dumps(GOOD) + "\n```"
        runner = FakeRunner([ok(envelope(fenced))])
        client, _ = make_client(runner)
        assert complete(client) == GOOD

    def test_one_retry_on_invalid_json(self) -> None:
        runner = FakeRunner([ok(envelope("not json")), ok(envelope(GOOD))])
        client, _ = make_client(runner)
        assert complete(client) == GOOD
        assert len(runner.calls) == 2

    def test_missing_required_key_retries_then_raises(self) -> None:
        bad = {"unexpected": True}
        runner = FakeRunner([ok(envelope(bad)), ok(envelope(bad))])
        client, _ = make_client(runner)
        with pytest.raises(SchemaValidationError):
            complete(client)
        assert len(runner.calls) == 2

    def test_non_json_envelope_retries_then_raises(self) -> None:
        runner = FakeRunner([ok("garbage"), ok("garbage")])
        client, _ = make_client(runner)
        with pytest.raises(ClaudeCodeError, match="non-JSON envelope"):
            complete(client)

    def test_timeout_retries_then_raises(self) -> None:
        """Four attempts before terminal: one inner retry per invocation
        (hung CLI subprocess) x one outer retry per completion. A single
        300s hang killed a 115-day committee replay 8 days in; a ~20-minute
        persistent outage is a real environment problem and stays terminal.
        """
        expired = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        runner = FakeRunner([expired, expired, expired, expired])
        client, _ = make_client(runner)
        with pytest.raises(ClaudeCodeError, match="timed out"):
            complete(client)
        assert len(runner.calls) == 4

    def test_one_timeout_then_success_recovers(self) -> None:
        """The campaign-killer case: a transient hang must not be terminal."""
        expired = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        runner = FakeRunner([expired, ok(envelope(GOOD))])
        client, _ = make_client(runner)
        assert complete(client) is not None
        assert len(runner.calls) == 2


class TestPlanLimit:
    def test_nonzero_exit_with_limit_text_maps(self) -> None:
        runner = FakeRunner([CliResult(1, "", "5-hour limit reached · resets 3pm")])
        client, _ = make_client(runner)
        with pytest.raises(PlanLimitExhausted):
            complete(client)
        assert len(runner.calls) == 1  # no retry on plan exhaustion

    def test_error_envelope_with_limit_text_maps(self) -> None:
        runner = FakeRunner(
            [ok(envelope("You have hit your usage limit.", is_error=True, subtype="error"))]
        )
        client, _ = make_client(runner)
        with pytest.raises(PlanLimitExhausted):
            complete(client)

    def test_nonzero_exit_without_limit_text_is_generic_error(self) -> None:
        crash = CliResult(1, "", "segfault")
        runner = FakeRunner([crash, crash])
        client, _ = make_client(runner)
        with pytest.raises(ClaudeCodeError, match="exited 1"):
            complete(client)

    def test_limit_text_in_successful_response_is_not_a_false_positive(self) -> None:
        payload = {"answer": "the position hit its risk limit reached threshold"}
        runner = FakeRunner([ok(envelope(payload))])
        client, _ = make_client(runner)
        assert complete(client) == payload


class TestApiFallback:
    class _FakeApi:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def complete_json(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {"answer": "from-api"}

    def test_fallback_disabled_by_default(self) -> None:
        api = self._FakeApi()
        runner = FakeRunner([CliResult(1, "", "usage limit reached")])
        client, _ = make_client(runner, api_fallback=api)
        with pytest.raises(PlanLimitExhausted):
            complete(client)
        assert api.calls == []

    def test_fallback_to_api_per_call(self) -> None:
        api = self._FakeApi()
        settings = make_settings(claude_code_fallback_to_api=True)
        runner = FakeRunner([CliResult(1, "", "usage limit reached")])
        client, _ = make_client(runner, settings=settings, api_fallback=api)
        assert complete(client) == {"answer": "from-api"}
        (call,) = api.calls
        assert call["role"] == "dossier"
        assert call["model_tier"] == "analyst"
        assert call["schema"] == SCHEMA

    def test_fallback_flag_without_client_still_raises(self) -> None:
        settings = make_settings(claude_code_fallback_to_api=True)
        runner = FakeRunner([CliResult(1, "", "usage limit reached")])
        client, _ = make_client(runner, settings=settings, api_fallback=None)
        with pytest.raises(PlanLimitExhausted):
            complete(client)


class TestCostRecording:
    def test_records_real_tokens_at_zero_cost(self) -> None:
        usage = {
            "input_tokens": 1200,
            "output_tokens": 340,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
        }
        runner = FakeRunner([ok(envelope(GOOD, usage=usage))])
        client, writer = make_client(runner)
        complete(client)
        (entry,) = writer.entries
        assert entry.cost_usd == 0.0  # flat-rate subscription — never metered
        assert entry.input_tokens == 1210  # input + cache writes
        assert entry.output_tokens == 340
        assert entry.cache_read_tokens == 50
        assert entry.call_kind == "claude_code:dossier"
        assert entry.model_id == "claude-sonnet-5"
        assert entry.batch is False

    def test_usage_recorded_even_when_output_invalid(self) -> None:
        runner = FakeRunner([ok(envelope("not json")), ok(envelope(GOOD))])
        client, writer = make_client(runner)
        complete(client)
        assert len(writer.entries) == 2  # both attempts metered


def batch_request(custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "model_tier": "analyst",
        "system": "SYSTEM",
        "context": "CONTEXT",
        "instruction": f"INSTRUCTION {custom_id}",
        "schema": SCHEMA,
        "max_tokens": 2048,
    }


class TestBatchSemantics:
    def test_sequential_loop_ended_immediately(self) -> None:
        runner = FakeRunner([ok(envelope({"answer": "a"})), ok(envelope({"answer": "b"}))])
        client, writer = make_client(runner)
        batch_id = client.submit_batch([batch_request("id-a"), batch_request("id-b")])
        assert batch_id.startswith("claude-code-local-")
        assert client.batch_status(batch_id) == "ended"
        results = list(client.batch_results(batch_id))
        assert results == [("id-a", {"answer": "a"}), ("id-b", {"answer": "b"})]
        assert len(runner.calls) == 2  # executed at submit time, in order
        assert all(e.call_kind == "claude_code:batch" for e in writer.entries)

    def test_invalid_output_yields_none_and_keeps_going(self) -> None:
        runner = FakeRunner(
            [
                ok(envelope("junk")),
                ok(envelope("junk")),  # retry also fails -> None
                ok(envelope({"answer": "b"})),
            ]
        )
        client, _ = make_client(runner)
        batch_id = client.submit_batch([batch_request("id-a"), batch_request("id-b")])
        assert list(client.batch_results(batch_id)) == [
            ("id-a", None),
            ("id-b", {"answer": "b"}),
        ]

    def test_plan_exhaustion_mid_batch_marks_remaining_failed(self) -> None:
        runner = FakeRunner(
            [
                ok(envelope({"answer": "a"})),
                CliResult(1, "", "usage limit reached"),
            ]
        )
        client, _ = make_client(runner)
        batch_id = client.submit_batch(
            [batch_request("id-a"), batch_request("id-b"), batch_request("id-c")]
        )
        assert list(client.batch_results(batch_id)) == [
            ("id-a", {"answer": "a"}),
            ("id-b", None),
            ("id-c", None),
        ]
        assert len(runner.calls) == 2  # id-c never invoked

    def test_unknown_batch_id_reports_ended_with_no_results(self) -> None:
        client, _ = make_client(FakeRunner([]))
        assert client.batch_status("from-a-previous-process") == "ended"
        assert list(client.batch_results("from-a-previous-process")) == []


class TestBinaryDiscovery:
    def test_configured_executable_path_wins(self, tmp_path: Any) -> None:
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        assert find_claude_binary(str(binary)) == str(binary)

    def test_configured_missing_path_returns_none(self, tmp_path: Any) -> None:
        assert find_claude_binary(str(tmp_path / "nope")) is None


class TestResourcesWiring:
    @staticmethod
    def _resource(settings: Settings) -> Any:
        from orchestration.resources import DynamoResource

        resource = DynamoResource(database_url="sqlite://")
        resource._settings = settings
        return resource

    def test_default_api_backend_unchanged(self) -> None:
        from dynamo.llm.anthropic_client import AnthropicClient

        resource = self._resource(isolated_settings(ANTHROPIC_API_KEY="test-key"))
        assert resource.settings().research_backend == "api"  # regression: default
        assert isinstance(resource.research_model(), AnthropicClient)

    def test_api_backend_without_key_is_none(self) -> None:
        resource = self._resource(isolated_settings(ANTHROPIC_API_KEY=""))
        assert resource.research_model() is None

    def test_claude_code_backend_returns_cli_client(self, tmp_path: Any) -> None:
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        resource = self._resource(
            isolated_settings(
                ANTHROPIC_API_KEY="",
                RESEARCH_BACKEND="claude_code",
                CLAUDE_CODE_BINARY=str(binary),
            )
        )
        model = resource.research_model()
        assert isinstance(model, ClaudeCodeResearchModel)
        assert model is resource.research_model()  # cached

    def test_claude_code_backend_missing_binary_is_none(self, tmp_path: Any) -> None:
        resource = self._resource(
            isolated_settings(
                ANTHROPIC_API_KEY="",
                RESEARCH_BACKEND="claude_code",
                CLAUDE_CODE_BINARY=str(tmp_path / "missing"),
            )
        )
        assert resource.research_model() is None

    def test_claude_code_backend_wires_api_fallback(self, tmp_path: Any) -> None:
        from dynamo.llm.anthropic_client import AnthropicClient

        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        resource = self._resource(
            isolated_settings(
                ANTHROPIC_API_KEY="test-key",
                RESEARCH_BACKEND="claude_code",
                CLAUDE_CODE_BINARY=str(binary),
                claude_code_fallback_to_api=True,
            )
        )
        model = resource.research_model()
        assert isinstance(model, ClaudeCodeResearchModel)
        assert isinstance(model._api_fallback, AnthropicClient)


class TestPricingRegistry:
    def test_fable_registered_flat_rate_with_cutoff(self) -> None:
        from datetime import date

        from dynamo.llm.pricing import calculate_cost, spec_for

        spec = spec_for("claude-fable-5")
        assert spec.input_per_mtok == 0.0
        assert spec.output_per_mtok == 0.0
        assert spec.knowledge_cutoff == date(2026, 1, 31)
        assert calculate_cost("claude-fable-5", input_tokens=1_000_000, output_tokens=50_000) == 0.0


def test_auth_error_envelope_raises_terminal_auth_error() -> None:
    """Expired-OAuth envelopes must abort loudly, never soft-skip (real-run lesson)."""
    from dynamo.llm.claude_code_client import ClaudeCodeAuthError

    envelope = json.dumps(
        {
            "is_error": True,
            "subtype": "success",
            "result": ("Failed to authenticate: OAuth session expired and could not be refreshed"),
            "usage": {},
        }
    )
    client, _ = make_client(FakeRunner([ok(envelope)]))
    with pytest.raises(ClaudeCodeAuthError):
        complete(client)


class TestArbiterFallback:
    def test_fable_window_exhaustion_degrades_arbiter_to_opus(self) -> None:
        """User directive: 'if fable runs out then go to opus'. The arbiter
        tier gets ONE hop; the retried call must carry the fallback model."""
        runner = FakeRunner(
            [CliResult(1, "", "5-hour limit reached · resets 3pm"), ok(envelope(GOOD))]
        )
        client, _ = make_client(runner)
        result = complete(client, model_tier="arbiter")
        assert result is not None
        assert len(runner.calls) == 2
        first_cmd, second_cmd = (c[0] for c in runner.calls)
        assert "claude-opus-5" in second_cmd
        assert first_cmd != second_cmd

    def test_non_arbiter_tiers_do_not_hop(self) -> None:
        runner = FakeRunner([CliResult(1, "", "limit reached · resets 3pm")])
        client, _ = make_client(runner)
        with pytest.raises(PlanLimitExhausted):
            complete(client, model_tier="analyst")
        assert len(runner.calls) == 1


class TestFailedExitEnvelope:
    """The CLI exits 1 AND prints an is_error JSON envelope for its own
    failures. The envelope used to be discarded, leaving a bare "exited 1"."""

    # Verbatim shape captured from the real CLI (2026-09-23) for a model the
    # installed version does not support.
    UNSUPPORTED = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "API Error: 400 Claude Code 2.1.269 does not support this model; "
            "version 2.1.280 or newer is required",
            "usage": {},
        }
    )

    def test_error_envelope_on_exit_1_explains_the_failure(self) -> None:
        stderr = '[claude-code:unrecognized_model] {"model":"claude-opus-5-5"}'
        failed = CliResult(1, self.UNSUPPORTED, stderr)
        runner = FakeRunner([failed, failed])  # a generic error is retried once
        client, _ = make_client(runner)
        with pytest.raises(ClaudeCodeError, match="does not support this model"):
            complete(client)

    def test_auth_envelope_on_exit_1_is_terminal_auth_error(self) -> None:
        from dynamo.llm.claude_code_client import ClaudeCodeAuthError

        body = json.dumps(
            {
                "is_error": True,
                "subtype": "success",
                "result": "Failed to authenticate: OAuth expired",
            }
        )
        client, _ = make_client(FakeRunner([CliResult(1, body, "")]))
        with pytest.raises(ClaudeCodeAuthError):
            complete(client)

    def test_non_json_failure_carries_the_stderr_tail(self) -> None:
        crash = CliResult(1, "", "Error: ENOSPC: no space left on device, write")
        client, _ = make_client(FakeRunner([crash, crash]))
        with pytest.raises(ClaudeCodeError, match="ENOSPC"):
            complete(client)


class TestCallTrace:
    """Every ledger row says which role/request it served, how long it took,
    which attempt it was and whether its output was usable (eval Phase 4)."""

    def test_batch_rows_carry_role_request_retrieval_and_latency(self) -> None:
        runner = FakeRunner([ok(envelope({"answer": "a"}))])
        client, writer = make_client(runner)
        req = batch_request("run:2026-08-03-MRK-bull") | {
            "role": "bull",
            "retrieved_ids": ["journal-1", "exp-2"],
        }
        client.submit_batch([req])
        (entry,) = writer.entries
        assert entry.call_kind == "claude_code:batch"  # unchanged
        tr = entry.trace
        assert tr is not None
        assert tr.role == "bull"
        assert tr.request_id == "run:2026-08-03-MRK-bull"
        assert tr.retrieved_ids == ("journal-1", "exp-2")
        assert tr.attempt == 1 and tr.validation_ok is True
        assert tr.latency_ms is not None and tr.latency_ms >= 0.0
        assert tr.equiv_cost_usd == pytest.approx(0.0123)
        assert tr.prompt_hash is not None and len(tr.prompt_hash) == 16

    def test_retry_after_invalid_output_is_attempt_two(self) -> None:
        runner = FakeRunner([ok(envelope("junk")), ok(envelope(GOOD))])
        client, writer = make_client(runner)
        client.complete_json(
            role="arbiter",
            system="SYSTEM",
            context="c",
            instruction="i",
            schema=SCHEMA,
            model_tier="arbiter",
        )
        traces = [e.trace for e in writer.entries]
        assert [(t.attempt, t.validation_ok, t.role) for t in traces if t] == [
            (1, False, "arbiter"),
            (2, True, "arbiter"),
        ]

    def test_prompt_hash_tracks_the_system_prompt_only(self) -> None:
        runner = FakeRunner([ok(envelope(GOOD)) for _ in range(3)])
        client, writer = make_client(runner)
        for system, ctx in (("S1", "a"), ("S1", "b"), ("S2", "a")):
            client.complete_json(
                role="bull",
                system=system,
                context=ctx,
                instruction="i",
                schema=SCHEMA,
                model_tier="analyst",
            )
        h = [e.trace.prompt_hash for e in writer.entries if e.trace]
        assert h[0] == h[1] != h[2]
