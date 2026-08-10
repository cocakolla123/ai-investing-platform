"""Deterministic dossier context assembly. Design: doc 03 section 3.

The context string is serialized stable -> volatile (instrument profile, quant
score, price/vol stats, filing excerpts, news digest, macro digest, journal
hits) with sorted keys and fixed number formats, so calls 1-4 for a ticker
share an identical cacheable prefix and re-runs are byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dynamo.domain.models.market import Frozen, Instrument, NewsItem
from dynamo.domain.models.research import QuantScore


class FilingExcerpt(Frozen):
    """A selected filing section (never the whole filing)."""

    accession_no: str
    form_type: str
    section: str
    text: str


# Earnings-call transcript truncation: prepared remarks live at the head,
# Q&A close at the tail — keep both, drop the middle.
TRANSCRIPT_HEAD_CHARS = 6000
TRANSCRIPT_TAIL_CHARS = 2000


def transcript_excerpt(
    content: str,
    *,
    symbol: str,
    event_date: str,
    source: str,
    head_chars: int = TRANSCRIPT_HEAD_CHARS,
    tail_chars: int = TRANSCRIPT_TAIL_CHARS,
) -> str:
    """Deterministic labeled excerpt of one call transcript for the dossier.

    Short transcripts pass through whole; long ones keep the first
    ``head_chars`` and last ``tail_chars`` with an explicit elision marker so
    the model knows material was cut.
    """
    text = content.strip()
    header = f"[transcript {symbol} {event_date} via {source}]"
    if len(text) <= head_chars + tail_chars:
        return f"{header}\n{text}"
    return (
        f"{header}\n{text[:head_chars]}"
        f"\n\n[... transcript truncated: middle omitted ...]\n\n"
        f"{text[-tail_chars:]}"
    )


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _instrument_lines(instrument: Instrument) -> list[str]:
    return [
        "## Instrument profile",
        f"symbol: {instrument.symbol}",
        f"name: {instrument.name}",
        f"exchange: {instrument.exchange or 'n/a'}",
        f"sector: {instrument.sector or 'n/a'}",
        f"industry: {instrument.industry or 'n/a'}",
        f"is_etf: {str(instrument.is_etf).lower()}",
        f"status: {instrument.status.value}",
    ]


def _quant_lines(quant_score: QuantScore | None) -> list[str]:
    lines = ["## Quant score"]
    if quant_score is None:
        lines.append("no quant score available")
        return lines
    lines += [
        f"model_version: {quant_score.model_version}",
        f"as_of: {quant_score.as_of.isoformat()}",
        f"expected_excess_return: {_fmt(quant_score.expected_excess_return)}",
        f"downside_prob: {_fmt(quant_score.downside_prob)}",
        f"vol_forecast: {_fmt(quant_score.vol_forecast)}",
        f"rank: {_fmt(quant_score.rank)}",
        f"confidence: {_fmt(quant_score.confidence)}",
        f"feature_coverage: {_fmt(quant_score.feature_coverage)}",
    ]
    return lines


def assemble_dossier_context(
    *,
    instrument: Instrument,
    quant_score: QuantScore | None,
    price_stats: Mapping[str, float],
    filing_excerpts: Sequence[FilingExcerpt] = (),
    news_items: Sequence[NewsItem] = (),
    macro_digest_text: str = "",
    journal_hits: Sequence[str] = (),
    earnings_call_excerpt: str = "",
) -> str:
    """Deterministic string from typed inputs (sorted keys, fixed formats).

    ``earnings_call_excerpt`` (see :func:`transcript_excerpt`) adds an
    "Earnings call transcript" section only when non-empty, so contexts
    without a recent call are byte-identical to before the feature existed.
    """
    lines: list[str] = []
    lines += _instrument_lines(instrument)
    lines += ["", *_quant_lines(quant_score)]

    lines += ["", "## Price/volume stats"]
    if price_stats:
        lines += [f"{key}: {_fmt(float(price_stats[key]))}" for key in sorted(price_stats)]
    else:
        lines.append("none")

    lines += ["", "## Filing excerpts"]
    if filing_excerpts:
        for exc in sorted(filing_excerpts, key=lambda e: (e.accession_no, e.section)):
            lines += [
                f"### {exc.form_type} {exc.accession_no} — {exc.section}",
                exc.text.strip(),
            ]
    else:
        lines.append("none")

    if earnings_call_excerpt:
        lines += ["", "## Earnings call transcript", earnings_call_excerpt.strip()]

    lines += ["", "## News digest"]
    if news_items:
        for item in sorted(news_items, key=lambda n: (n.published_at.isoformat(), n.id)):
            summary = f" — {item.summary.strip()}" if item.summary else ""
            lines.append(
                f"- [{item.id}] {item.published_at.isoformat()} "
                f"({item.source}) {item.headline.strip()}{summary}"
            )
    else:
        lines.append("none")

    lines += ["", "## Macro digest", macro_digest_text.strip() or "none"]

    lines += ["", "## Journal hits"]
    if journal_hits:
        lines += [f"- {hit.strip()}" for hit in journal_hits]
    else:
        lines.append("none")

    return "\n".join(lines)
