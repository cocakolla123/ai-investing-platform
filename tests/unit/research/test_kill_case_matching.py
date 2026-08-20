"""Rebuttal <-> kill-case matching: the Phase 0 "make `buy` reachable" fix.

The decisive regression test replays the three real ABBV arbiter payloads from
control run `replay-e5a49317bbee` (2026-05-08 / 05-13 / 05-27). The arbiter
chose `buy` on all three; raw-substring matching in
`_unrebutted_high_kill_cases` failed to pair its paraphrased, `"HIGH: "`-
prefixed rebuttals with the bear's kill-case descriptions, so all three were
silently downgraded to `hold`. They must now survive as `buy` — while a
genuinely unrebutted high-severity kill case must still downgrade.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fakes import BEAR_HIGH_KILL_CASE, BEAR_PAYLOAD, EVIDENCE

from dynamo.domain.models.research import Action, AdvocateBrief, Side
from dynamo.research.committee import (
    GUARDRAIL_AGENT,
    KILL_CASE_INDEX_HEADING,
    CommitteeEngine,
    arbiter_context_for,
    compute_disagreement,
    render_indexed_kill_cases,
)

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "committee"
ABBV_FIXTURES = sorted(FIXTURES.glob("abbv-*-arbiter-buy.json"))


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def assemble(fixture: dict[str, Any]) -> Any:
    return CommitteeEngine.assemble_verdict(
        symbol=fixture["symbol"],
        as_of=date.fromisoformat(fixture["as_of"]),
        arbiter_payload=fixture["arbiter"],
        bull=AdvocateBrief.model_validate(fixture["bull"]),
        bear=AdvocateBrief.model_validate(fixture["bear"]),
    )


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


class TestStoredABBVPayloads:
    """The proof the fix works on real data, not on hand-written fakes."""

    def test_three_fixtures_are_present(self) -> None:
        assert len(ABBV_FIXTURES) == 3

    @pytest.mark.parametrize("path", ABBV_FIXTURES, ids=lambda p: p.stem)
    def test_arbiter_buy_survives_the_guardrail(self, path: Path) -> None:
        fixture = load_fixture(path)
        # Precondition: the arbiter really did choose buy, and the shipped
        # guardrail really did downgrade it.
        assert fixture["arbiter"]["action"] == "buy"
        assert fixture["stored_action"] == "hold"
        # And none of these rebuttals carries an index — they exercise the
        # tier-2 text fallback exactly as stored history will.
        assert all(r.get("kill_case_index") is None for r in fixture["arbiter"]["bear_rebuttals"])

        verdict = assemble(fixture)
        assert verdict.action is Action.BUY
        assert all(d.agent != GUARDRAIL_AGENT for d in verdict.dissents)
        assert "[guardrail]" not in verdict.self_critique

    @pytest.mark.parametrize("path", ABBV_FIXTURES, ids=lambda p: p.stem)
    def test_disagreement_no_longer_vetoes_a_genuine_buy(self, path: Path) -> None:
        fixture = load_fixture(path)
        verdict = assemble(fixture)
        assert verdict.disagreement_score <= 0.6
        # The action term is gone, so the score is strictly the advocates'.
        assert verdict.disagreement_score == pytest.approx(
            compute_disagreement(
                bull_conviction=fixture["bull"]["conviction"],
                bear_conviction=fixture["bear"]["conviction"],
            )
        )

    @pytest.mark.parametrize("path", ABBV_FIXTURES, ids=lambda p: p.stem)
    def test_dropping_evidence_from_rebuttals_reinstates_the_downgrade(self, path: Path) -> None:
        """Substance is required: the matcher alone must not clear the bar."""
        fixture = load_fixture(path)
        stripped = {
            **fixture["arbiter"],
            "bear_rebuttals": [{**r, "evidence": []} for r in fixture["arbiter"]["bear_rebuttals"]],
        }
        verdict = CommitteeEngine.assemble_verdict(
            symbol=fixture["symbol"],
            as_of=date.fromisoformat(fixture["as_of"]),
            arbiter_payload=stripped,
            bull=AdvocateBrief.model_validate(fixture["bull"]),
            bear=AdvocateBrief.model_validate(fixture["bear"]),
        )
        assert verdict.action is Action.HOLD
        assert any(d.agent == GUARDRAIL_AGENT for d in verdict.dissents)


class TestMatchingTiers:
    AS_OF = date(2026, 8, 3)

    def _verdict(self, rebuttals: list[dict[str, Any]], bear_payload: Any = None) -> Any:
        bear = AdvocateBrief.model_validate({**(bear_payload or BEAR_PAYLOAD), "conviction": 0.2})
        return CommitteeEngine.assemble_verdict(
            symbol="ACME",
            as_of=self.AS_OF,
            arbiter_payload={
                "action": "buy",
                "conviction": 0.7,
                "thesis": "t",
                "bull_case": "b",
                "bear_case": "b",
                "bear_rebuttals": rebuttals,
                "risks": [],
                "exit_conditions": [],
                "expected_upside_pct": 10.0,
                "expected_downside_pct": 5.0,
                "horizon_days": 30,
                "position_size_hint_pct": 2.0,
                "dissents": [],
                "self_critique": "c",
            },
            bull=make_brief(Side.BULL, 0.8),
            bear=bear,
        )

    def test_tier1_index_matches_despite_total_paraphrase(self) -> None:
        verdict = self._verdict(
            [
                {
                    "kill_case": "HIGH: the objection about who buys from us",
                    "rebuttal": "Renewed through 2028.",
                    "evidence": [EVIDENCE],
                    "kill_case_index": 0,
                }
            ]
        )
        assert verdict.action is Action.BUY

    def test_tier2_matches_a_severity_prefixed_paraphrase(self) -> None:
        verdict = self._verdict(
            [
                {
                    "kill_case": "HIGH: Collapse of customer concentration",
                    "rebuttal": "Renewed through 2028.",
                    "evidence": [EVIDENCE],
                }
            ]
        )
        assert verdict.action is Action.BUY

    def test_wrong_index_and_unrelated_text_still_downgrades(self) -> None:
        """The guardrail must not become a no-op."""
        verdict = self._verdict(
            [
                {
                    "kill_case": "HIGH: currency translation headwind in EMEA",
                    "rebuttal": "Hedged.",
                    "evidence": [EVIDENCE],
                    "kill_case_index": 7,
                }
            ]
        )
        assert verdict.action is Action.HOLD
        assert "Customer concentration collapse" in verdict.self_critique
        assert any(d.agent == GUARDRAIL_AGENT for d in verdict.dissents)

    def test_unrelated_paraphrase_does_not_fool_the_fuzzy_tier(self) -> None:
        verdict = self._verdict(
            [
                {
                    "kill_case": "MEDIUM: a sustained rise in freight and packaging costs",
                    "rebuttal": "Contracted rates.",
                    "evidence": [EVIDENCE],
                }
            ]
        )
        assert verdict.action is Action.HOLD

    def test_rebuttal_without_evidence_never_counts(self) -> None:
        verdict = self._verdict(
            [
                {
                    "kill_case": "Customer concentration collapse",
                    "rebuttal": "Not a concern.",
                    "evidence": [],
                    "kill_case_index": 0,
                }
            ]
        )
        assert verdict.action is Action.HOLD

    def test_index_points_at_the_bear_payload_position_not_the_high_only_position(
        self,
    ) -> None:
        """A medium kill case ahead of the high one must not shift the join key."""
        bear_payload = {
            **BEAR_PAYLOAD,
            "kill_cases": [
                {
                    "description": "Freight costs rise",
                    "severity": "medium",
                    "trigger": "diesel > $6",
                },
                BEAR_HIGH_KILL_CASE,
            ],
        }
        rebuttal = {
            "kill_case": "restated elsewhere",
            "rebuttal": "Renewed through 2028.",
            "evidence": [EVIDENCE],
        }
        assert self._verdict([{**rebuttal, "kill_case_index": 1}], bear_payload).action is (
            Action.BUY
        )
        assert self._verdict([{**rebuttal, "kill_case_index": 0}], bear_payload).action is (
            Action.HOLD
        )


class TestDisagreementRescaling:
    def test_legitimate_buy_with_aligned_advocates_clears_the_veto(self) -> None:
        # Bull convinced, bear unconvinced: exactly the shape of a justified buy.
        score = compute_disagreement(bull_conviction=0.75, bear_conviction=0.30)
        assert score <= 0.6

    def test_action_no_longer_moves_the_score(self) -> None:
        assert compute_disagreement(bull_conviction=0.42, bear_conviction=0.32) == (
            compute_disagreement(bull_conviction=0.42, bear_conviction=0.32)
        )
        # add vs buy used to differ purely by ACTION_SCORE; there is now no
        # action input at all, so the artifact cannot recur.
        assert "action" not in compute_disagreement.__code__.co_varnames

    def test_two_maximally_convinced_opposed_advocates_still_veto(self) -> None:
        assert compute_disagreement(bull_conviction=0.9, bear_conviction=0.9) > 0.6

    def test_zero_conviction_on_both_sides_is_zero(self) -> None:
        assert compute_disagreement(bull_conviction=0.0, bear_conviction=0.0) == 0.0


class TestIndexedKillCaseRendering:
    def test_kill_cases_are_numbered_in_payload_order(self) -> None:
        rendered = render_indexed_kill_cases(BEAR_PAYLOAD)
        assert KILL_CASE_INDEX_HEADING in rendered
        assert "kill_cases[0] (severity=high): Customer concentration collapse" in rendered
        assert "kill_case_index" in rendered

    def test_missing_or_malformed_kill_cases_render_safely(self) -> None:
        assert "(none)" in render_indexed_kill_cases({})
        assert "(none)" in render_indexed_kill_cases({"kill_cases": "not a list"})

    def test_arbiter_context_carries_the_indexed_block(self) -> None:
        context = arbiter_context_for("CONTEXT", {}, {}, BEAR_PAYLOAD, None)
        assert KILL_CASE_INDEX_HEADING in context
        assert "kill_cases[0]" in context
        assert context.index(KILL_CASE_INDEX_HEADING) < context.index("## Quant score")
