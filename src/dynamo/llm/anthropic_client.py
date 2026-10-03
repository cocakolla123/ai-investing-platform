"""Anthropic adapter implementing the ResearchModel port. Design: doc 03.

- Structured output via output_config json_schema, with runtime detection and a
  JSON-instruction fallback for SDKs/APIs that reject it.
- Prompt-cache friendly block ordering: system (cached) -> context (cached) ->
  instruction (volatile).
- Cost metered on every response; `check_budget()` is consulted before every
  submission — built in so it cannot be bypassed.
- No temperature/top_p overrides, no assistant prefill. API keys are never logged.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import anthropic

from dynamo.config.settings import Settings
from dynamo.domain.ports import CONTEXT_CACHE_BOUNDARY
from dynamo.llm.cost_meter import CallTrace, CostMeter

_TIERS = ("screen", "analyst", "arbiter")

_FALLBACK_SUFFIX = (
    "\n\nRespond with ONLY a single valid JSON object that conforms to this JSON Schema"
    " — no prose, no markdown code fences:\n"
)


class SchemaValidationError(RuntimeError):
    """Model output failed JSON/schema validation after one retry."""


def _required_keys_present(parsed: dict[str, Any], schema: dict[str, Any]) -> bool:
    required = schema.get("required", [])
    return all(key in parsed for key in required)


def _extract_text(response: Any) -> str:
    parts = [
        getattr(block, "text", "")
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", "") == "text"
    ]
    return "".join(parts)


def _parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class AnthropicClient:
    """Implements dynamo.domain.ports.ResearchModel."""

    def __init__(
        self,
        settings: Settings,
        cost_meter: CostMeter,
        client: Any | None = None,
    ) -> None:
        self._meter = cost_meter
        self._client: Any = (
            client
            if client is not None
            else anthropic.Anthropic(api_key=settings.anthropic_api_key)
        )
        self._tier_models: dict[str, str] = {
            "screen": settings.model_screen,
            "analyst": settings.model_analyst,
            "arbiter": settings.model_arbiter,
        }
        # None = unknown (try structured output), False = fall back to JSON-in-prompt.
        self._structured_supported: bool | None = None
        # batch_id -> {custom_id: schema}, for validating batch results.
        self._batch_schemas: dict[str, dict[str, dict[str, Any]]] = {}

    # --- helpers ---

    def _model_for(self, model_tier: str) -> str:
        if model_tier not in _TIERS:
            raise ValueError(f"unknown model_tier {model_tier!r}; expected one of {_TIERS}")
        return self._tier_models[model_tier]

    def _build_params(
        self,
        *,
        model: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        max_tokens: int,
        structured: bool,
        batch: bool = False,
    ) -> dict[str, Any]:
        if not structured:
            instruction = instruction + _FALLBACK_SUFFIX + json.dumps(schema, sort_keys=True)
        # Batches run concurrently over long windows -> 1h TTL maximizes
        # best-effort hits (batch-processing doc tip); sync path uses the 5m
        # default. Batch + cache-read discounts stack (~95% off on hits).
        cache: dict[str, str] = (
            {"type": "ephemeral", "ttl": "1h"} if batch else {"type": "ephemeral"}
        )
        # Context arrives as stable->volatile segments split on the cache
        # boundary (doc 03 section 3). Each of the first 3 segments gets its
        # own breakpoint (4 total incl. system) so later calls sharing a
        # prefix re-read it from cache; extra segments stay uncached.
        segments = [s for s in context.split(CONTEXT_CACHE_BOUNDARY) if s]
        content: list[dict[str, Any]] = []
        for seg in segments[:3]:
            content.append({"type": "text", "text": seg, "cache_control": dict(cache)})
        overflow = CONTEXT_CACHE_BOUNDARY.join(segments[3:])
        if overflow:
            content.append({"type": "text", "text": overflow})
        content.append({"type": "text", "text": instruction})
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": dict(cache)}],
            "messages": [{"role": "user", "content": content}],
        }
        if structured:
            params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return params

    def _create(
        self,
        *,
        model: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> Any:
        if self._structured_supported is not False:
            params = self._build_params(
                model=model,
                system=system,
                context=context,
                instruction=instruction,
                schema=schema,
                max_tokens=max_tokens,
                structured=True,
            )
            try:
                response = self._client.messages.create(**params)
            except (TypeError, anthropic.BadRequestError):
                # SDK or API does not accept output_config — fall back permanently.
                self._structured_supported = False
            else:
                self._structured_supported = True
                return response
        params = self._build_params(
            model=model,
            system=system,
            context=context,
            instruction=instruction,
            schema=schema,
            max_tokens=max_tokens,
            structured=False,
        )
        return self._client.messages.create(**params)

    def _parse_response(self, response: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
        if getattr(response, "stop_reason", None) == "refusal":
            return None
        parsed = _parse_json(_extract_text(response))
        if parsed is None or not _required_keys_present(parsed, schema):
            return None
        return parsed

    def _record(
        self,
        response: Any,
        *,
        call_kind: str,
        model: str,
        batch: bool,
        trace: CallTrace | None = None,
    ) -> None:
        model_id = getattr(response, "model", None) or model
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._meter.record_usage(
                model_id=str(model_id), call_kind=call_kind, usage=usage, batch=batch, trace=trace
            )

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
    ) -> dict[str, Any]:
        model = self._model_for(model_tier)
        for attempt in range(2):  # one retry on invalid output
            self._meter.check_budget()
            started = time.perf_counter()
            response = self._create(
                model=model,
                system=system,
                context=context,
                instruction=instruction,
                schema=schema,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            parsed = self._parse_response(response, schema)
            self._record(
                response,
                call_kind=role,
                model=model,
                batch=False,
                trace=CallTrace(
                    role=role,
                    prompt_hash=CallTrace.hash_prompt(system),
                    latency_ms=latency_ms,
                    attempt=attempt + 1,
                    validation_ok=parsed is not None,
                ),
            )
            if parsed is not None:
                return parsed
        raise SchemaValidationError(
            f"role {role!r} ({model}): output failed schema validation after retry"
        )

    def submit_batch(self, requests: Sequence[dict[str, Any]]) -> str:
        self._meter.check_budget()
        structured = self._structured_supported is not False
        batch_requests: list[dict[str, Any]] = []
        schemas: dict[str, dict[str, Any]] = {}
        for req in requests:
            model = self._model_for(req["model_tier"])
            params = self._build_params(
                model=model,
                system=req["system"],
                context=req["context"],
                instruction=req["instruction"],
                schema=req["schema"],
                max_tokens=int(req.get("max_tokens", 4096)),
                structured=structured,
                batch=True,
            )
            custom_id = str(req["custom_id"])
            schemas[custom_id] = req["schema"]
            batch_requests.append({"custom_id": custom_id, "params": params})
        batch = self._client.messages.batches.create(requests=batch_requests)
        batch_id = str(batch.id)
        self._batch_schemas[batch_id] = schemas
        return batch_id

    def batch_status(self, batch_id: str) -> str:
        batch = self._client.messages.batches.retrieve(batch_id)
        return str(batch.processing_status)

    def batch_results(self, batch_id: str) -> Iterable[tuple[str, dict[str, Any] | None]]:
        return self._iter_batch_results(batch_id)

    def _iter_batch_results(self, batch_id: str) -> Iterator[tuple[str, dict[str, Any] | None]]:
        schemas = self._batch_schemas.get(batch_id, {})
        for entry in self._client.messages.batches.results(batch_id):
            custom_id = str(entry.custom_id)
            result = entry.result
            if getattr(result, "type", None) != "succeeded":
                # errored / expired / canceled — caller resubmits via deterministic ids
                yield custom_id, None
                continue
            message = result.message
            self._record(
                message, call_kind="batch", model=str(getattr(message, "model", "")), batch=True
            )
            parsed = self._parse_response(message, schemas.get(custom_id, {}))
            yield custom_id, parsed
