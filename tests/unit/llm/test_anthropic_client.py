"""AnthropicClient: budget gate, structured-output fallback, retry, batches.

All API objects are stubs — no network, no real API calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.config.settings import Settings
from dynamo.llm.anthropic_client import AnthropicClient, SchemaValidationError
from dynamo.llm.cost_meter import BudgetExceededError, CostMeter, InMemoryLedgerWriter

SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
GOOD = {"answer": "42"}


def make_response(payload: Any, *, stop_reason: str = "end_turn") -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        stop_reason=stop_reason,
        model="claude-sonnet-4-6",
    )


class StubBatches:
    def __init__(self, results: list[Any] | None = None) -> None:
        self.created_with: list[Any] = []
        self._results = results or []

    def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
        self.created_with.append(requests)
        return SimpleNamespace(id="batch-abc")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def results(self, batch_id: str) -> list[Any]:
        return self._results


class StubMessages:
    def __init__(self, responses: list[Any], *, reject_structured: bool = False) -> None:
        self._responses = list(responses)
        self.reject_structured = reject_structured
        self.calls: list[dict[str, Any]] = []
        self.batches = StubBatches()

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.reject_structured and "output_config" in kwargs:
            raise TypeError("unexpected keyword argument 'output_config'")
        return self._responses.pop(0)


def make_client(
    messages: StubMessages, *, spent: float = 0.0, budget: float = 90.0
) -> tuple[AnthropicClient, CostMeter, InMemoryLedgerWriter]:
    writer = InMemoryLedgerWriter()
    meter = CostMeter(writer, budget)
    if spent:
        meter.record(
            model_id="claude-sonnet-4-6",
            call_kind="prior",
            input_tokens=0,
            output_tokens=0,
            cost_usd=spent,
        )
    settings = Settings(ANTHROPIC_API_KEY="test-key")
    stub = SimpleNamespace(messages=messages)
    return AnthropicClient(settings, meter, client=stub), meter, writer


def complete(client: AnthropicClient) -> dict[str, Any]:
    return client.complete_json(
        role="dossier",
        system="SYSTEM",
        context="CONTEXT",
        instruction="INSTRUCTION",
        schema=SCHEMA,
        model_tier="analyst",
    )


class TestCompleteJson:
    def test_happy_path_structured_output(self) -> None:
        messages = StubMessages([make_response(GOOD)])
        client, _, writer = make_client(messages)
        assert complete(client) == GOOD
        call = messages.calls[0]
        assert call["model"] == "claude-sonnet-5"
        assert call["output_config"] == {
            "format": {"type": "json_schema", "schema": SCHEMA}
        }
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        blocks = call["messages"][0]["content"]
        assert blocks[0]["text"] == "CONTEXT"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert blocks[1]["text"] == "INSTRUCTION"
        assert "temperature" not in call
        assert "top_p" not in call
        assert len(writer.entries) == 1  # cost recorded from usage

    def test_budget_gate_blocks_before_any_api_call(self) -> None:
        messages = StubMessages([make_response(GOOD)])
        client, _, _ = make_client(messages, spent=90.0, budget=90.0)
        with pytest.raises(BudgetExceededError):
            complete(client)
        assert messages.calls == []

    def test_fallback_when_sdk_rejects_output_config(self) -> None:
        messages = StubMessages([make_response(GOOD)], reject_structured=True)
        client, _, _ = make_client(messages)
        assert complete(client) == GOOD
        assert len(messages.calls) == 2  # structured attempt, then fallback
        fallback = messages.calls[1]
        assert "output_config" not in fallback
        instruction_text = fallback["messages"][0]["content"][1]["text"]
        assert "JSON Schema" in instruction_text
        # fallback is sticky: next call skips the structured attempt
        messages._responses.append(make_response(GOOD))
        complete(client)
        assert "output_config" not in messages.calls[2]

    def test_fallback_parses_fenced_json(self) -> None:
        fenced = "```json\n" + json.dumps(GOOD) + "\n```"
        messages = StubMessages([make_response(fenced)])
        client, _, _ = make_client(messages)
        assert complete(client) == GOOD

    def test_invalid_output_retries_once_then_raises(self) -> None:
        messages = StubMessages(
            [make_response({"wrong": 1}), make_response({"also_wrong": 2})]
        )
        client, _, writer = make_client(messages)
        with pytest.raises(SchemaValidationError):
            complete(client)
        assert len(messages.calls) == 2
        assert len(writer.entries) == 2  # cost still recorded for both attempts

    def test_refusal_counts_as_invalid(self) -> None:
        messages = StubMessages(
            [make_response(GOOD, stop_reason="refusal"), make_response(GOOD)]
        )
        client, _, _ = make_client(messages)
        assert complete(client) == GOOD
        assert len(messages.calls) == 2

    def test_unknown_tier_raises(self) -> None:
        client, _, _ = make_client(StubMessages([]))
        with pytest.raises(ValueError, match="model_tier"):
            client.complete_json(
                role="r",
                system="s",
                context="c",
                instruction="i",
                schema=SCHEMA,
                model_tier="jumbo",
            )


def batch_request(custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "model_tier": "analyst",
        "system": "SYSTEM",
        "context": "CONTEXT",
        "instruction": "INSTRUCTION",
        "schema": SCHEMA,
        "max_tokens": 1024,
    }


class TestBatches:
    def test_submit_builds_params_and_returns_id(self) -> None:
        messages = StubMessages([])
        client, _, _ = make_client(messages)
        batch_id = client.submit_batch([batch_request("2026-08-03-ACME-dossier")])
        assert batch_id == "batch-abc"
        [requests] = messages.batches.created_with
        assert requests[0]["custom_id"] == "2026-08-03-ACME-dossier"
        params = requests[0]["params"]
        assert params["model"] == "claude-sonnet-5"
        assert params["max_tokens"] == 1024
        # Batch requests use 1h-TTL caching (batch-processing doc: batches can
        # exceed the 5m TTL; discounts stack on hits).
        assert params["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert params["messages"][0]["content"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_segmented_context_gets_per_segment_breakpoints(self) -> None:
        from dynamo.domain.ports import CONTEXT_CACHE_BOUNDARY

        messages = StubMessages([make_response(GOOD)])
        client, _, _ = make_client(messages)
        ctx = CONTEXT_CACHE_BOUNDARY.join(["BASE", "DOSSIER", "ADVOCATES", "EXTRA", "MORE"])
        client.complete_json(
            role="arbiter",
            system="SYSTEM",
            context=ctx,
            instruction="INSTRUCTION",
            schema=SCHEMA,
            model_tier="arbiter",
        )
        blocks = messages.calls[0]["messages"][0]["content"]
        # 3 cached segments + 1 uncached overflow + instruction = 5 blocks;
        # 4 total breakpoints incl. system (the API maximum).
        assert [b["text"] for b in blocks[:3]] == ["BASE", "DOSSIER", "ADVOCATES"]
        assert all(b["cache_control"] == {"type": "ephemeral"} for b in blocks[:3])
        assert "cache_control" not in blocks[3]
        assert CONTEXT_CACHE_BOUNDARY in blocks[3]["text"]
        assert blocks[4]["text"] == "INSTRUCTION"

    def test_submit_blocked_when_over_budget(self) -> None:
        messages = StubMessages([])
        client, _, _ = make_client(messages, spent=100.0)
        with pytest.raises(BudgetExceededError):
            client.submit_batch([batch_request("x")])
        assert messages.batches.created_with == []

    def test_status(self) -> None:
        client, _, _ = make_client(StubMessages([]))
        assert client.batch_status("batch-abc") == "in_progress"

    def test_results_yield_parsed_or_none_and_meter_cost(self) -> None:
        entries = [
            SimpleNamespace(
                custom_id="ok",
                result=SimpleNamespace(type="succeeded", message=make_response(GOOD)),
            ),
            SimpleNamespace(custom_id="boom", result=SimpleNamespace(type="errored")),
            SimpleNamespace(custom_id="late", result=SimpleNamespace(type="expired")),
            SimpleNamespace(
                custom_id="no",
                result=SimpleNamespace(
                    type="succeeded", message=make_response(GOOD, stop_reason="refusal")
                ),
            ),
        ]
        messages = StubMessages([])
        messages.batches = StubBatches(results=entries)
        client, _, writer = make_client(messages)
        client.submit_batch([batch_request("ok"), batch_request("no")])
        results = dict(client.batch_results("batch-abc"))
        assert results == {"ok": GOOD, "boom": None, "late": None, "no": None}
        # cost recorded only for succeeded results, at batch pricing
        assert all(e.batch for e in writer.entries)
        assert len(writer.entries) == 2


# --------------------------------------------------------------------------
# network isolation regression (backtest-campaign Phase 4): the client must
# NEVER enable tools / web search / MCP / beta features — replay determinism
# and the leakage contract both depend on the model having no side channels.
# --------------------------------------------------------------------------

BANNED_PARAM_KEYS = {"tools", "tool_choice", "web_search", "mcp_servers", "betas"}


def test_build_params_never_contains_tool_or_web_keys() -> None:
    """Every _build_params combination (structured x batch, i.e. the sync,
    batch, structured and JSON-fallback paths) yields only the fixed
    parameter set — no tools/web_search/mcp/betas, ever."""
    client, _, _ = make_client(StubMessages([]))
    for structured in (True, False):
        for batch in (True, False):
            params = client._build_params(
                model="claude-sonnet-5",
                system="sys",
                context="ctx",
                instruction="do it",
                schema=SCHEMA,
                max_tokens=64,
                structured=structured,
                batch=batch,
            )
            assert not BANNED_PARAM_KEYS & set(params)
            expected = {"model", "max_tokens", "system", "messages"}
            if structured:
                expected.add("output_config")
            assert set(params) == expected


def test_wire_params_have_no_tool_keys_on_sync_and_batch_paths() -> None:
    """Belt and suspenders at the wire: the kwargs actually handed to
    messages.create and the per-request params handed to batches.create are
    free of tool/web/beta keys — including the structured-rejection fallback."""
    for reject_structured in (False, True):
        messages = StubMessages(
            [make_response(GOOD), make_response(GOOD)],
            reject_structured=reject_structured,
        )
        client, _, _ = make_client(messages)
        complete(client)
        assert messages.calls
        for call in messages.calls:
            assert not BANNED_PARAM_KEYS & set(call)
    messages = StubMessages([])
    client, _, _ = make_client(messages)
    client.submit_batch(
        [
            {
                "custom_id": "run:2026-06-01-LIQ-dossier",
                "model_tier": "analyst",
                "system": "sys",
                "context": "ctx",
                "instruction": "do it",
                "schema": SCHEMA,
                "max_tokens": 64,
            }
        ]
    )
    (requests,) = messages.batches.created_with
    for entry in requests:
        assert not BANNED_PARAM_KEYS & set(entry["params"])
