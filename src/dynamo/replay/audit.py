"""Post-run leakage audit (backtest-campaign Phase 4).

Pure functions over persisted rows: the CLI (``replay audit --run-id``) loads
the run row, its ``replay_contexts`` and ``replay_verdicts`` and hands them
here. Four checks:

1. **context_dates** — every persisted decision-time context is scanned for
   date-like strings (ISO dates, "Month YYYY", bare 20xx years) LATER than
   that row's ``as_of``. A small, conservative allowlist admits legitimately
   forward-looking phrasing (horizons, expiries, guidance/estimate language);
   everything else is flagged with an excerpt for human review.
2. **context_sha** — each row's ``context_sha256`` must match its text
   (tamper evidence for the audit trail).
3. **verdict_dates** — the same date scan over each verdict's persisted
   payload (thesis / exit conditions / stage payloads).
4. **config** — the run's ``config_json`` must carry the resolved model
   knowledge cutoffs + gate decision and the embedder backend the run used.

The audit is INFORMATIONAL: exit code 1 means "a human must look", not
"the run is proven leaky" — date mentions have benign causes (price targets,
fiscal-year labels) the allowlist deliberately does not try to exhaust.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# Known fidelity caveats (accepted in the plan) stamped into every report —
# the audit cannot test these, so it must say so out loud.
KNOWN_CAVEATS: tuple[str, ...] = (
    "instruments/sectors are not vintaged: sector labels are today's, not the "
    "classification at as_of (negligible for mega-caps; documented).",
    "news backfills reflect the provider's CURRENT archive: provider-side "
    "revisionism (deleted/edited items) is untestable from our side.",
    "no social-media history exists at any price via the Reddit/StockTwits "
    "APIs: the live social-burst signal has no past to replay.",
    "crypto news coverage in the historical archive is sparse-to-absent: "
    "crypto dossiers lean on price/volume context.",
)

_EXCERPT_WINDOW = 60  # chars kept either side of a flagged match

_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_INDEX = {m: i + 1 for i, m in enumerate(_MONTHS)}
_MONTH_INDEX.update({m[:3]: i + 1 for i, m in enumerate(_MONTHS)})
_MONTH_INDEX["sept"] = 9

ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
MONTH_YEAR_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(20\d{2})\b",
    re.IGNORECASE,
)
BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Conservative allowlist for legitimately forward-looking dossier/verdict
# language: trade horizons and exits, option/contract expiries, guidance and
# estimate phrasing, fiscal-period labels. Applied to a +/-60-char window
# around each date match; anything not covered stays flagged for the human.
ALLOWLIST_RE = re.compile(
    r"(horizon|expir\w*|matur\w*|deadline|due\b|by (the )?end of|"
    r"next (quarter|month|year|session)|guidance|outlook|forecast\w*|"
    r"estimate\w*|expected|expects?\b|target\w*|project\w*|fiscal|"
    r"\bFY\s?'?\d{2,4}|\bQ[1-4][\s-]|exit[_ ]condition|close_ts|valid[_ ]to|"
    r"resolution|settle\w*|upcoming|scheduled)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditFinding:
    ref: str  # row id / verdict id the finding points at
    as_of: str  # that row's as_of (ISO)
    detail: str
    excerpt: str = ""


@dataclass
class AuditCategory:
    status: str  # PASS | FLAGGED
    detail: str
    checked: int = 0
    allowlisted: int = 0
    findings: list[AuditFinding] = field(default_factory=list)


@dataclass
class AuditReport:
    run_id: str
    generated_at: str
    categories: dict[str, AuditCategory]
    caveats: tuple[str, ...] = KNOWN_CAVEATS

    @property
    def passed(self) -> bool:
        return all(c.status == "PASS" for c in self.categories.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "categories": {name: asdict(cat) for name, cat in self.categories.items()},
            "known_caveats": list(self.caveats),
        }


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _excerpt(text: str, start: int, end: int) -> str:
    lo = max(start - _EXCERPT_WINDOW, 0)
    hi = min(end + _EXCERPT_WINDOW, len(text))
    return ("…" if lo else "") + text[lo:hi].replace("\n", " ") + ("…" if hi < len(text) else "")


def _allowlisted(text: str, start: int, end: int) -> bool:
    lo = max(start - _EXCERPT_WINDOW, 0)
    hi = min(end + _EXCERPT_WINDOW, len(text))
    return ALLOWLIST_RE.search(text[lo:hi]) is not None


def scan_text_for_future_dates(
    text: str, as_of: date, *, ref: str
) -> tuple[list[AuditFinding], int]:
    """(flagged findings, allowlisted count) for one text vs its as_of.

    Precedence: full ISO dates, then "Month YYYY", then bare years — later
    (weaker) patterns skip spans already consumed by stronger ones, so one
    date string yields at most one candidate.
    """
    findings: list[AuditFinding] = []
    allowlisted = 0
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in consumed)

    def consider(start: int, end: int, found: date | None, label: str) -> None:
        consumed.append((start, end))
        if found is None or found <= as_of:
            return
        nonlocal allowlisted
        if _allowlisted(text, start, end):
            allowlisted += 1
            return
        findings.append(
            AuditFinding(
                ref=ref,
                as_of=as_of.isoformat(),
                detail=f"{label} {text[start:end]!r} is after as_of {as_of.isoformat()}",
                excerpt=_excerpt(text, start, end),
            )
        )

    for m in ISO_DATE_RE.finditer(text):
        y, mo, d = (int(g) for g in m.groups())
        try:
            parsed: date | None = date(y, mo, d)
        except ValueError:
            parsed = None
        consider(m.start(), m.end(), parsed, "date")

    for m in MONTH_YEAR_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        month = _MONTH_INDEX.get(m.group(1).lower().rstrip("."))
        parsed = date(int(m.group(2)), month, 1) if month else None
        consider(m.start(), m.end(), parsed, "month reference")

    for m in BARE_YEAR_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        year = int(m.group(1))
        # A bare year equal to as_of's is unremarkable; later years are not.
        parsed = date(year, 1, 1) if year > as_of.year else None
        consider(m.start(), m.end(), parsed, "year")

    return findings, allowlisted


def _audit_context_dates(contexts: list[dict[str, Any]]) -> AuditCategory:
    cat = AuditCategory(status="PASS", detail="persisted contexts scanned for post-as_of dates")
    for row in contexts:
        as_of = _as_date(row.get("as_of"))
        if as_of is None:
            cat.findings.append(
                AuditFinding(ref=str(row.get("id")), as_of="?", detail="row has no as_of")
            )
            continue
        found, allowed = scan_text_for_future_dates(
            str(row.get("context_text", "")), as_of, ref=str(row.get("id"))
        )
        cat.checked += 1
        cat.allowlisted += allowed
        cat.findings.extend(found)
    if cat.findings:
        cat.status = "FLAGGED"
    return cat


def _audit_context_sha(contexts: list[dict[str, Any]]) -> AuditCategory:
    import hashlib

    cat = AuditCategory(status="PASS", detail="context_sha256 verified against context_text")
    for row in contexts:
        cat.checked += 1
        text = str(row.get("context_text", ""))
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected != str(row.get("context_sha256", "")):
            cat.findings.append(
                AuditFinding(
                    ref=str(row.get("id")),
                    as_of=str(row.get("as_of")),
                    detail="context_sha256 does not match context_text (tampered or corrupt)",
                )
            )
    if cat.findings:
        cat.status = "FLAGGED"
    return cat


def _audit_verdict_dates(verdicts: list[dict[str, Any]]) -> AuditCategory:
    cat = AuditCategory(status="PASS", detail="verdict payloads scanned for post-as_of dates")
    for row in verdicts:
        as_of = _as_date(row.get("as_of"))
        if as_of is None:
            cat.findings.append(
                AuditFinding(ref=str(row.get("id")), as_of="?", detail="row has no as_of")
            )
            continue
        text = json.dumps(row.get("payload", {}), sort_keys=True, default=str)
        found, allowed = scan_text_for_future_dates(text, as_of, ref=str(row.get("id")))
        cat.checked += 1
        cat.allowlisted += allowed
        # A verdict payload always contains its OWN as_of and the horizon
        # machinery; those are <= as_of / allowlisted. Real findings stay.
        cat.findings.extend(found)
    if cat.findings:
        cat.status = "FLAGGED"
    return cat


def _audit_config(run: dict[str, Any]) -> tuple[AuditCategory, list[str]]:
    """Cross-check config_json for the Phase-4 provenance stamps. Returns the
    category plus extra caveat lines (an ACKNOWLEDGED knowledge gate is a
    documented caveat, not a finding)."""
    cat = AuditCategory(status="PASS", detail="config_json provenance stamps")
    extra_caveats: list[str] = []
    config = dict(run.get("config_json") or {})
    mk = config.get("model_knowledge")
    if not isinstance(mk, dict) or "cutoffs" not in mk:
        cat.findings.append(
            AuditFinding(
                ref=str(run.get("id")),
                as_of=str(run.get("start_as_of")),
                detail="config_json.model_knowledge missing — resolved model cutoffs "
                "and the gate decision were not recorded",
            )
        )
    else:
        cat.detail = (
            f"cutoffs={mk.get('cutoffs')} blocked={mk.get('blocked')} "
            f"acknowledged={mk.get('acknowledged')}"
        )
        if mk.get("blocked") and mk.get("acknowledged"):
            extra_caveats.append(
                "model knowledge gate was BLOCKED and explicitly acknowledged: the "
                f"window starts within {mk.get('margin_days')}d of a committee "
                f"model's documented training cutoff ({mk.get('cutoffs')}). "
                "Committee models may 'remember' this period; results carry that "
                "asterisk by design."
            )
        elif mk.get("blocked") and not mk.get("acknowledged"):
            cat.findings.append(
                AuditFinding(
                    ref=str(run.get("id")),
                    as_of=str(run.get("start_as_of")),
                    detail="gate recorded blocked=True without acknowledgement — "
                    "this run should never have spent",
                )
            )
    if not config.get("embedder_backend"):
        cat.findings.append(
            AuditFinding(
                ref=str(run.get("id")),
                as_of=str(run.get("start_as_of")),
                detail="config_json.embedder_backend missing — the embedder the run "
                "used was not recorded",
            )
        )
    cat.checked = 1
    if cat.findings:
        cat.status = "FLAGGED"
    return cat, extra_caveats


def audit_run(
    run: dict[str, Any],
    contexts: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> AuditReport:
    """Assemble the full report from persisted rows (pure; no I/O)."""
    config_cat, extra_caveats = _audit_config(run)
    return AuditReport(
        run_id=str(run.get("id")),
        generated_at=datetime.now(UTC).isoformat(),
        categories={
            "context_dates": _audit_context_dates(contexts),
            "context_sha": _audit_context_sha(contexts),
            "verdict_dates": _audit_verdict_dates(verdicts),
            "config": config_cat,
        },
        caveats=(*KNOWN_CAVEATS, *extra_caveats),
    )


def write_audit_artifacts(report: AuditReport, out_dir: Path) -> tuple[Path, Path]:
    """``leakage_audit.json`` + human-readable ``leakage_audit.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "leakage_audit.json"
    md_path = out_dir / "leakage_audit.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2, default=str) + "\n")

    lines = [
        f"# Leakage audit — {report.run_id}",
        "",
        f"Generated: {report.generated_at}",
        f"Overall: **{'PASS' if report.passed else 'FLAGGED'}**"
        + ("" if report.passed else " (informational — a human reviews the findings)"),
        "",
    ]
    for name, cat in sorted(report.categories.items()):
        lines += [
            f"## {name} — {cat.status}",
            "",
            f"{cat.detail} (checked {cat.checked}, allowlisted {cat.allowlisted})",
            "",
        ]
        for f in cat.findings:
            lines.append(f"- `{f.ref}` (as_of {f.as_of}): {f.detail}")
            if f.excerpt:
                lines.append(f"  > {f.excerpt}")
        if cat.findings:
            lines.append("")
    lines += ["## Known caveats (untestable; accepted in the plan)", ""]
    lines += [f"- {c}" for c in report.caveats]
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path
