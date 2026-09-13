"""Post-run leakage audit (backtest-campaign Phase 4) — pure unit tests over
fabricated persisted rows: clean runs PASS, planted future dates / tampered
hashes / missing provenance FLAG, and the conservative allowlist admits
legitimately forward-looking dossier phrasing."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from dynamo.replay.audit import (
    audit_run,
    scan_text_for_future_dates,
    write_audit_artifacts,
)

AS_OF = date(2026, 6, 15)
RUN_ID = "replay-abc123def456"


def _context_row(
    text: str,
    *,
    stage: str = "dossier",
    symbol: str = "LIQ",
    as_of: date = AS_OF,
    sha: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{RUN_ID}:{as_of.isoformat()}-{symbol}-{stage}",
        "run_id": RUN_ID,
        "as_of": as_of,
        "symbol": symbol,
        "stage": stage,
        "context_sha256": sha or hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "context_text": text,
        "model_id": "claude-sonnet-5",
    }


def _run_row(**config: Any) -> dict[str, Any]:
    base = {
        "model_knowledge": {
            "cutoffs": {"claude-opus-5": "2026-05-31"},
            "margin_days": 30,
            "blocked": True,
            "blocking_models": ["claude-opus-5"],
            "acknowledged": True,
        },
        "embedder_backend": "hash-fallback",
    }
    base.update(config)
    return {
        "id": RUN_ID,
        "start_as_of": AS_OF,
        "end_as_of": AS_OF,
        "config_json": base,
    }


def _verdict_row(payload: dict[str, Any], as_of: date = AS_OF) -> dict[str, Any]:
    return {
        "id": f"{RUN_ID}:v-1",
        "run_id": RUN_ID,
        "symbol": "LIQ",
        "as_of": as_of,
        "action": "buy",
        "payload": payload,
    }


CLEAN_TEXT = (
    "## News digest\n- [n-1] 2026-06-10T09:00:00+00:00 (test) LIQ wins contract\n"
    "## Quant score\nas_of: 2026-06-15\n"
    "revenue grew 12% in May 2026 versus January 2026."
)


# -- scanner + allowlist -----------------------------------------------------


def test_scanner_clean_text_passes() -> None:
    findings, allowlisted = scan_text_for_future_dates(CLEAN_TEXT, AS_OF, ref="r")
    assert findings == [] and allowlisted == 0


def test_scanner_flags_future_iso_date_with_excerpt() -> None:
    text = "the merger closed on 2026-09-15 and the stock rallied"
    findings, _ = scan_text_for_future_dates(text, AS_OF, ref="r")
    assert len(findings) == 1
    assert "2026-09-15" in findings[0].detail
    assert "merger closed" in findings[0].excerpt


def test_scanner_flags_future_month_reference_and_bare_year() -> None:
    findings, _ = scan_text_for_future_dates(
        "results announced in September 2026 were strong", AS_OF, ref="r"
    )
    assert len(findings) == 1
    findings, _ = scan_text_for_future_dates(
        "the recession that hit in 2027 changed everything", AS_OF, ref="r"
    )
    assert len(findings) == 1


def test_allowlist_admits_horizon_and_expiry_phrasing() -> None:
    # Realistic dossier/verdict text: session horizons carry no year at all;
    # expiries and guidance mention future dates legitimately.
    for text in (
        "position sized over a 20-session horizon",
        "the thesis expires 2026-09-30 if the catalyst slips",
        "options expiring September 2026 imply a 6% move",
        "management guidance for FY 2027 was reaffirmed",
        "next quarter (Q3 2026) consensus estimates imply 4% growth",
        "exit_condition: reassess by end of August 2026",
    ):
        findings, _ = scan_text_for_future_dates(text, AS_OF, ref="r")
        assert findings == [], text


# -- full report -------------------------------------------------------------


def test_clean_run_passes_all_categories(tmp_path: Path) -> None:
    contexts = [_context_row(CLEAN_TEXT)]
    verdicts = [
        _verdict_row({"thesis": "buy on momentum", "horizon_days": 20, "as_of": AS_OF.isoformat()})
    ]
    report = audit_run(_run_row(), contexts, verdicts)
    assert report.passed
    assert {c.status for c in report.categories.values()} == {"PASS"}
    json_path, md_path = write_audit_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["passed"] is True and payload["run_id"] == RUN_ID
    md = md_path.read_text()
    assert "PASS" in md and "Known caveats" in md
    # The acknowledged knowledge gate lands in the caveats, not as a finding.
    assert any("acknowledged" in c for c in report.caveats)


def test_planted_future_date_flags_context_with_row_ref(tmp_path: Path) -> None:
    bad = CLEAN_TEXT + "\nOn 2026-08-03 the company preannounced results."
    report = audit_run(_run_row(), [_context_row(bad)], [])
    cat = report.categories["context_dates"]
    assert cat.status == "FLAGGED" and not report.passed
    assert cat.findings[0].ref == f"{RUN_ID}:{AS_OF.isoformat()}-LIQ-dossier"
    assert "2026-08-03" in cat.findings[0].detail
    _json_path, md_path = write_audit_artifacts(report, tmp_path)
    assert "2026-08-03" in md_path.read_text()


def test_sha_tamper_flags() -> None:
    row = _context_row(CLEAN_TEXT, sha="0" * 64)
    report = audit_run(_run_row(), [row], [])
    assert report.categories["context_sha"].status == "FLAGGED"
    assert "does not match" in report.categories["context_sha"].findings[0].detail


def test_verdict_payload_scanned_against_row_as_of() -> None:
    payload = {
        "thesis": "the 2026-07-20 earnings beat is already public",
        "as_of": AS_OF.isoformat(),
    }
    report = audit_run(_run_row(), [], [_verdict_row(payload)])
    assert report.categories["verdict_dates"].status == "FLAGGED"


def test_missing_provenance_stamps_flag_config() -> None:
    run = {"id": RUN_ID, "start_as_of": AS_OF, "config_json": {}}
    report = audit_run(run, [], [])
    cat = report.categories["config"]
    assert cat.status == "FLAGGED"
    details = " ".join(f.detail for f in cat.findings)
    assert "model_knowledge" in details and "embedder_backend" in details


def test_decimal_tails_are_not_years() -> None:
    """The wide-committee audit flagged 954 'post-as_of dates' and every
    sampled one was the tail of a return metric: 5.2089% read as year 2089,
    -3.2094% as 2094. A digit or decimal point before a 4-digit group makes
    it arithmetic, not a calendar reference."""
    from dynamo.replay.audit import BARE_YEAR_RE

    for numeric in ("5.2089", "-3.2094", "benchmark_ret_120d_pct: 5.2089", "0.2050"):
        assert not BARE_YEAR_RE.search(numeric), numeric
    for year in ("in 2089 the", "by 2094.", "(2050)"):
        assert BARE_YEAR_RE.search(year), year
