"""Deterministic dossier context assembly. Design: doc 03 section 3.

The context string is serialized stable -> volatile (instrument profile,
fundamentals, quant score, price/vol stats, peer relative strength, filing
excerpts, news digest, macro digest, journal hits) with sorted keys and fixed
number formats, so calls 1-4 for a ticker share an identical cacheable prefix
and re-runs are byte-identical.

The ``fundamentals`` and ``peers`` sections close the design gap the committee
campaign measured: before them this function had no parameter that could carry
a financial statement or a comparable, so 944,859 ``xbrl_facts`` rows and a
fully populated set of sector ETFs never reached the model. 90.8% of verdicts
declared a fundamentals gap and 287 of 295 a peer gap; bull conviction never
once cleared 0.6.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dynamo.domain.models.market import Frozen, Instrument, NewsItem
from dynamo.domain.models.research import QuantScore
from dynamo.research.fundamentals import FundamentalsSnapshot, render_line
from dynamo.research.market_stats import PeerContext


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


def _news_lines(items: Sequence[NewsItem]) -> list[str]:
    """One line per item, chronological and deterministic (id breaks ties)."""
    return [
        f"- [{item.id}] {item.published_at.isoformat()} "
        f"({item.source}) {item.headline.strip()}"
        + (f" — {item.summary.strip()}" if item.summary else "")
        for item in sorted(items, key=lambda n: (n.published_at.isoformat(), n.id))
    ]


def _fundamentals_lines(fundamentals: FundamentalsSnapshot | None) -> list[str]:
    """``## Fundamentals`` — every line carries the ``source_ref`` an advocate
    must cite; absent metrics render ``n/a`` and are never invented."""
    lines = ["## Fundamentals"]
    if fundamentals is None or not fundamentals.groups:
        lines.append("none")
        return lines
    lines.append(
        f"basis: point-in-time XBRL facts filed <= {fundamentals.as_of.isoformat()}; "
        "source_ref format xbrl:<symbol>:<tag>:<period_end>:filed-<date>"
    )
    for group in fundamentals.groups:
        lines.append(f"### {group.title}")
        lines += [render_line(line) for line in group.lines]
    return lines


def _peer_lines(peers: PeerContext | None) -> list[str]:
    """``## Peer / relative strength`` — the name against its sector ETF and
    the broad benchmark (287 of 295 verdicts reported this data as missing)."""
    lines = ["## Peer / relative strength"]
    if peers is None or not peers.stats:
        lines.append("none")
        return lines
    lines.append(
        f"sector: {peers.sector or 'n/a'} | sector_etf: {peers.sector_etf or 'n/a'} "
        f"| benchmark: {peers.benchmark}"
    )
    lines += [f"{key}: {_fmt(float(peers.stats[key]))}" for key in sorted(peers.stats)]
    return lines


def assemble_dossier_context(
    *,
    instrument: Instrument,
    quant_score: QuantScore | None,
    price_stats: Mapping[str, float],
    fundamentals: FundamentalsSnapshot | None = None,
    peers: PeerContext | None = None,
    filing_excerpts: Sequence[FilingExcerpt] = (),
    news_items: Sequence[NewsItem] = (),
    market_news_items: Sequence[NewsItem] = (),
    macro_digest_text: str = "",
    journal_hits: Sequence[str] = (),
    earnings_call_excerpt: str = "",
) -> str:
    """Deterministic string from typed inputs (sorted keys, fixed formats).

    ``earnings_call_excerpt`` (see :func:`transcript_excerpt`) adds an
    "Earnings call transcript" section only when non-empty, so contexts
    without a recent call are byte-identical to before the feature existed.
    ``news_items`` are the items that name the subject company;
    ``market_news_items`` are related-but-not-about-it items, rendered under
    their own heading so context is never mistaken for evidence.
    """
    lines: list[str] = []
    lines += _instrument_lines(instrument)
    lines += ["", *_fundamentals_lines(fundamentals)]
    lines += ["", *_quant_lines(quant_score)]

    lines += ["", "## Price/volume stats"]
    if price_stats:
        lines += [f"{key}: {_fmt(float(price_stats[key]))}" for key in sorted(price_stats)]
    else:
        lines.append("none")

    lines += ["", *_peer_lines(peers)]

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
        lines += _news_lines(news_items)
    else:
        lines.append("none")

    if market_news_items:
        lines += [
            "",
            "## Market news",
            f"(not about {instrument.symbol} — sector/market context only)",
            *_news_lines(market_news_items),
        ]

    lines += ["", "## Macro digest", macro_digest_text.strip() or "none"]

    lines += ["", "## Journal hits"]
    if journal_hits:
        lines += [f"- {hit.strip()}" for hit in journal_hits]
    else:
        lines.append("none")

    return "\n".join(lines)
