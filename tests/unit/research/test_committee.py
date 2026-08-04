"""Committee flow, guardrails, disagreement bounds, custom_id determinism."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from fakes import (
    ARBITER_PAYLOAD,
    BEAR_PAYLOAD,
    BULL_PAYLOAD,
    DOSSIER_PAYLOAD,
    FakeResearchModel,
)
from hypothesis import given
from hypothesis import strategies as st

from dynamo.domain.models.research import (
    Action,
    AdvocateBrief,
    CommitteeVerdict,
    DossierAnalysis,
    QuantAlignment,
    QuantScore,
    Side,
)
from dynamo.research.committee import (
    GUARDRAIL_AGENT,
    CommitteeEngine,
    compute_disagreement,
    custom_id_for,
    quant_alignment_for,
    split_custom_id,
)
from dynamo.research.prompt_loader import load_all_prompts

AS_OF = date(2026, 8, 3)

# Minimal stand-in schemas — the fake never validates against them.
SCHEMAS: dict[str, dict[str, Any]] = {
    "dossier": {"type": "object", "required": ["symbol"]},
    "advocate": {"type": "object", "required": ["symbol"]},
    "arbiter": {"type": "object", "required": ["action"]},
}


def make_engine(
    responses: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[CommitteeEngine, FakeResearchModel]:
    model = FakeResearchModel(
        responses
        or {
            "dossier": [DOSSIER_PAYLOAD],
            "bull": [BULL_PAYLOAD],
            "bear": [BEAR_PAYLOAD],
            "arbiter": [ARBITER_PAYLOAD],
        }
    )
    return CommitteeEngine(model, load_all_prompts(), SCHEMAS), model


def make_quant(eer: float | None) -> QuantScore:
    return QuantScore(symbol="ACME", as_of=AS_OF, model_version="qm-1", expected_excess_return=eer)


def make_brief(side: Side, conviction: float, kill_cases: tuple = ()) -> AdvocateBrief:
    return AdvocateBrief(
        symbol="ACME",
        side=side,
        thesis="t",
        key_points=(),
        kill_cases=kill_cases,
        expected_move_pct=10.0,
        horizon_days=30,
        conviction=conviction,
    )


class TestDeepDiveFlow:
    def test_returns_domain_objects_from_canned_json(self) -> None:
        engine, model = make_engine()
        dossier, bull, bear, verdict = engine.run_deep_dive(
            "ACME", AS_OF, "CONTEXT", make_quant(0.05), journal_context_used=["j-1"]
        )
        assert isinstance(dossier, DossierAnalysis)
        assert dossier.prompt_version == "1.0.0"
        assert bull.side is Side.BULL
        assert bear.side is Side.BEAR
        assert isinstance(verdict, CommitteeVerdict)
        # rebutted high kill case + moderate disagreement -> buy stands
        assert verdict.action is Action.BUY
        assert verdict.quant_alignment is QuantAlignment.AGREE
        assert verdict.journal_context_used == ("j-1",)
        assert [c["role"] for c in model.calls] == ["dossier", "bull", "bear", "arbiter"]
        assert [c["model_tier"] for c in model.calls] == [
            "analyst",
            "analyst",
            "analyst",
            "arbiter",
        ]

    def test_advocates_and_arbiter_see_dossier_in_context(self) -> None:
        engine, model = make_engine()
        engine.run_deep_dive("ACME", AS_OF, "CONTEXT", None)
        dossier_json = json.dumps(DOSSIER_PAYLOAD, sort_keys=True, default=str)
        for call in model.calls[1:]:
            assert call["context"].startswith("CONTEXT")
            assert dossier_json in call["context"]
        arbiter_context = model.calls[3]["context"]
        assert "Bull advocate brief" in arbiter_context
        assert "Bear advocate brief" in arbiter_context


class TestGuardrails:
    def test_high_disagreement_downgrades_buy_to_hold(self) -> None:
        bull = make_brief(Side.BULL, 0.9)
        bear = make_brief(Side.BEAR, 0.9)
        verdict = CommitteeEngine.assemble_verdict(
            symbol="ACME",
            as_of=AS_OF,
            arbiter_payload={**ARBITER_PAYLOAD, "conviction": 0.9},
            bull=bull,
            bear=bear,
        )
        assert verdict.disagreement_score > 0.6
        assert verdict.action is Action.HOLD
        assert any(d.agent == GUARDRAIL_AGENT for d in verdict.dissents)
        assert "downgraded buy to hold" in verdict.self_critique

    def test_unrebutted_high_kill_case_downgrades(self) -> None:
        bear_payload = {**BEAR_PAYLOAD, "conviction": 0.1}
        bear = AdvocateBrief.model_validate({**bear_payload, "side": "bear"})
        bull = make_brief(Side.BULL, 0.8)
        payload = {**ARBITER_PAYLOAD, "bear_rebuttals": []}
        verdict = CommitteeEngine.assemble_verdict(
            symbol="ACME", as_of=AS_OF, arbiter_payload=payload, bull=bull, bear=bear
        )
        assert verdict.disagreement_score <= 0.6  # only the kill case fired
        assert verdict.action is Action.HOLD
        assert "Customer concentration collapse" in verdict.self_critique
        assert any(d.agent == GUARDRAIL_AGENT for d in verdict.dissents)

    def test_rebutted_kill_case_and_low_disagreement_keeps_buy(self) -> None:
        bear = AdvocateBrief.model_validate({**BEAR_PAYLOAD, "conviction": 0.2})
        bull = make_brief(Side.BULL, 0.8)
        verdict = CommitteeEngine.assemble_verdict(
            symbol="ACME", as_of=AS_OF, arbiter_payload=ARBITER_PAYLOAD, bull=bull, bear=bear
        )
        assert verdict.action is Action.BUY
        assert all(d.agent != GUARDRAIL_AGENT for d in verdict.dissents)

    def test_non_buy_actions_never_downgraded(self) -> None:
        bear = AdvocateBrief.model_validate(BEAR_PAYLOAD)
        bull = make_brief(Side.BULL, 1.0)
        payload = {**ARBITER_PAYLOAD, "action": "sell", "bear_rebuttals": []}
        verdict = CommitteeEngine.assemble_verdict(
            symbol="ACME", as_of=AS_OF, arbiter_payload=payload, bull=bull, bear=bear
        )
        assert verdict.action is Action.SELL


class TestDisagreement:
    @given(
        bull=st.floats(min_value=0.0, max_value=1.0),
        bear=st.floats(min_value=0.0, max_value=1.0),
        arbiter=st.floats(min_value=0.0, max_value=1.0),
        action=st.sampled_from(list(Action)),
    )
    def test_score_always_in_unit_interval(
        self, bull: float, bear: float, arbiter: float, action: Action
    ) -> None:
        score = compute_disagreement(
            bull_conviction=bull,
            bear_conviction=bear,
            action=action,
            arbiter_conviction=arbiter,
        )
        assert 0.0 <= score <= 1.0

    def test_unanimous_committee_scores_zero(self) -> None:
        score = compute_disagreement(
            bull_conviction=1.0,
            bear_conviction=0.0,
            action=Action.BUY,
            arbiter_conviction=1.0,
        )
        assert score == 0.0


class TestQuantAlignment:
    def test_alignment_matrix(self) -> None:
        assert quant_alignment_for(Action.BUY, make_quant(0.04)) is QuantAlignment.AGREE
        assert quant_alignment_for(Action.BUY, make_quant(-0.04)) is QuantAlignment.CONFLICT
        assert quant_alignment_for(Action.SELL, make_quant(-0.04)) is QuantAlignment.AGREE
        assert quant_alignment_for(Action.SELL, make_quant(0.04)) is QuantAlignment.CONFLICT
        assert quant_alignment_for(Action.HOLD, make_quant(0.04)) is QuantAlignment.NEUTRAL
        assert quant_alignment_for(Action.BUY, make_quant(None)) is QuantAlignment.NEUTRAL
        assert quant_alignment_for(Action.BUY, None) is QuantAlignment.NEUTRAL


class TestBatchHelpers:
    def test_custom_ids_deterministic_and_reversible(self) -> None:
        engine, _ = make_engine()
        items = {"BRK-B": "ctx-b", "ACME": "ctx-a"}
        first = engine.build_batch_requests("dossier", items, AS_OF)
        second = engine.build_batch_requests("dossier", items, AS_OF)
        assert first == second
        assert [r["custom_id"] for r in first] == [
            "2026-08-03-ACME-dossier",
            "2026-08-03-BRK-B-dossier",
        ]
        for request in first:
            as_of, symbol, stage = split_custom_id(request["custom_id"])
            assert as_of == AS_OF
            assert stage == "dossier"
            assert request["context"] == items[symbol]

    def test_split_handles_hyphenated_symbols(self) -> None:
        cid = custom_id_for(AS_OF, "BRK-B", "arbiter")
        assert split_custom_id(cid) == (AS_OF, "BRK-B", "arbiter")

    def test_parse_stage_results_separates_failures(self) -> None:
        engine, _ = make_engine()
        results = [
            ("2026-08-03-ACME-bull", {"thesis": "x"}),
            ("2026-08-03-BRK-B-bull", None),
            ("2026-08-03-ACME-bear", {"thesis": "y"}),  # other stage — ignored
        ]
        parsed, failed = engine.parse_stage_results(results, "bull")
        assert parsed == {"ACME": {"thesis": "x"}}
        assert failed == ["2026-08-03-BRK-B-bull"]


class TestVerdictId:
    def test_deterministic_and_input_sensitive(self) -> None:
        bear = AdvocateBrief.model_validate(BEAR_PAYLOAD)
        bull = make_brief(Side.BULL, 0.8)
        kwargs: dict[str, Any] = {
            "symbol": "ACME",
            "as_of": AS_OF,
            "bull": bull,
            "bear": bear,
        }
        v1 = CommitteeEngine.assemble_verdict(arbiter_payload=ARBITER_PAYLOAD, **kwargs)
        v2 = CommitteeEngine.assemble_verdict(arbiter_payload=ARBITER_PAYLOAD, **kwargs)
        v3 = CommitteeEngine.assemble_verdict(
            arbiter_payload={**ARBITER_PAYLOAD, "conviction": 0.5}, **kwargs
        )
        assert v1.id == v2.id
        assert v1.id.startswith("ACME-2026-08-03-")
        assert v1.id != v3.id
