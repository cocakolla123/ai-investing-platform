"""Deterministic dossier context assembly (stable -> volatile, sorted keys)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.market import Instrument, NewsItem
from dynamo.domain.models.research import QuantScore
from dynamo.research.dossier import FilingExcerpt, assemble_dossier_context

INSTRUMENT = Instrument(symbol="ACME", name="Acme Corp", sector="Industrials")
QUANT = QuantScore(
    symbol="ACME",
    as_of=date(2026, 8, 3),
    model_version="qm-1",
    expected_excess_return=0.0512345,
    rank=7,
)
NEWS = (
    NewsItem(
        id="n-2",
        symbols=("ACME",),
        headline="Acme wins contract",
        source="finnhub",
        published_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    ),
    NewsItem(
        id="n-1",
        symbols=("ACME",),
        headline="Acme guides up",
        summary="Raises FY guidance",
        source="finnhub",
        published_at=datetime(2026, 8, 1, 9, 0, tzinfo=UTC),
    ),
)
EXCERPTS = (
    FilingExcerpt(
        accession_no="0001-26-000042", form_type="10-Q", section="MD&A", text="Margins improved."
    ),
)


def build(**overrides: object) -> str:
    kwargs: dict = {
        "instrument": INSTRUMENT,
        "quant_score": QUANT,
        "price_stats": {"vol_20d": 0.25, "close": 101.5, "adv_usd": 1_000_000.0},
        "filing_excerpts": EXCERPTS,
        "news_items": NEWS,
        "macro_digest_text": "Regime stable.",
        "journal_hits": ["j-1: similar setup in 2024"],
    }
    kwargs.update(overrides)
    return assemble_dossier_context(**kwargs)


def test_byte_identical_across_runs() -> None:
    assert build() == build()


def test_price_stats_keys_sorted_and_fixed_format() -> None:
    context = build()
    adv = context.index("adv_usd: 1000000.0000")
    close = context.index("close: 101.5000")
    vol = context.index("vol_20d: 0.2500")
    assert adv < close < vol


def test_stable_to_volatile_section_order() -> None:
    context = build()
    positions = [
        context.index("## Instrument profile"),
        context.index("## Fundamentals"),
        context.index("## Quant score"),
        context.index("## Price/volume stats"),
        context.index("## Peer / relative strength"),
        context.index("## Filing excerpts"),
        context.index("## News digest"),
        context.index("## Macro digest"),
        context.index("## Journal hits"),
    ]
    assert positions == sorted(positions)


def test_fundamentals_section_renders_every_line_with_its_citation() -> None:
    from datetime import datetime as _dt

    import pandas as pd

    from dynamo.research.fundamentals import build_fundamentals

    facts = pd.DataFrame(
        [
            {
                "symbol": "ACME",
                "tag": "Assets",
                "period_end": pd.Timestamp("2026-03-31"),
                "value": 5_000_000.0,
                "filed_at": _dt(2026, 4, 30, tzinfo=UTC),
            }
        ]
    )
    snapshot = build_fundamentals(facts, symbol="ACME", as_of=date(2026, 8, 3))
    context = build(fundamentals=snapshot)
    section = context.split("## Fundamentals", 1)[1].split("## Quant score", 1)[0]
    assert "basis: point-in-time XBRL facts filed <= 2026-08-03" in section
    assert (
        "total_assets: 5.0M USD [xbrl:ACME:Assets:2026-03-31:filed-2026-04-30]" in section
    )
    assert "revenue_ttm: n/a" in section  # absent, not invented


def test_fundamentals_and_peer_sections_absent_by_default() -> None:
    context = build()
    assert "## Fundamentals\nnone" in context
    assert "## Peer / relative strength\nnone" in context


def test_peer_section_renders_sorted_stats_with_its_header() -> None:
    from dynamo.research.market_stats import PeerContext

    peers = PeerContext(
        symbol="ACME",
        sector="Industrials",
        sector_etf="XLI",
        stats={"excess_vs_XLI_20d_pct": 3.5, "beta_120d_vs_SPY": 1.25},
    )
    context = build(peers=peers)
    section = context.split("## Peer / relative strength", 1)[1].split("##", 1)[0]
    assert "sector: Industrials | sector_etf: XLI | benchmark: SPY" in section
    assert section.index("beta_120d_vs_SPY: 1.2500") < section.index(
        "excess_vs_XLI_20d_pct: 3.5000"
    )


def test_market_news_is_a_separate_labelled_section() -> None:
    unrelated = NewsItem(
        id="m-1",
        symbols=("ACME",),
        headline="Sector index rallies",
        source="finnhub",
        published_at=datetime(2026, 8, 2, 8, tzinfo=UTC),
    )
    context = build(market_news_items=(unrelated,))
    digest = context.split("## News digest", 1)[1].split("## Market news", 1)[0]
    market = context.split("## Market news", 1)[1]
    assert "[n-1]" in digest and "[m-1]" not in digest
    assert "not about ACME" in market
    assert "[m-1]" in market
    assert "## Market news" not in build()  # absent when there is nothing to say


def test_news_sorted_by_published_at_not_input_order() -> None:
    context = build()
    assert context.index("[n-1]") < context.index("[n-2]")


def test_journal_hits_rendered_in_their_section() -> None:
    context = build(journal_hits=["j-1: similar setup in 2024", "lesson: respect the stop"])
    section = context.split("## Journal hits", 1)[1]
    assert "- j-1: similar setup in 2024" in section
    assert "- lesson: respect the stop" in section


def test_missing_optional_inputs_render_placeholders() -> None:
    context = build(
        quant_score=None,
        price_stats={},
        filing_excerpts=(),
        news_items=(),
        macro_digest_text="",
        journal_hits=(),
    )
    assert "no quant score available" in context
    assert context.count("none") >= 4


def test_transcript_section_only_when_excerpt_provided() -> None:
    from dynamo.research.dossier import transcript_excerpt

    baseline = build()
    assert "## Earnings call transcript" not in baseline  # byte-identical legacy path

    excerpt = transcript_excerpt(
        "CEO: Great quarter." * 3,
        symbol="ACME",
        event_date="2026-08-05",
        source="alphavantage",
    )
    context = build(earnings_call_excerpt=excerpt)
    section = context.index("## Earnings call transcript")
    assert context.index("## Filing excerpts") < section < context.index("## News digest")
    assert "[transcript ACME 2026-08-05 via alphavantage]" in context


def test_transcript_excerpt_truncates_head_and_tail() -> None:
    from dynamo.research.dossier import transcript_excerpt

    content = "H" * 7000 + "M" * 5000 + "T" * 3000
    excerpt = transcript_excerpt(
        content,
        symbol="ACME",
        event_date="2026-08-05",
        source="defeatbeta",
        head_chars=6000,
        tail_chars=2000,
    )
    assert excerpt.startswith("[transcript ACME 2026-08-05 via defeatbeta]\n" + "H" * 100)
    assert "[... transcript truncated: middle omitted ...]" in excerpt
    assert excerpt.endswith("T" * 2000)
    # head + tail + labels only — the middle is gone
    assert len(excerpt) < 6000 + 2000 + 200

    short = transcript_excerpt(
        "short call", symbol="ACME", event_date="2026-08-05", source="defeatbeta"
    )
    assert short.endswith("short call")
    assert "truncated" not in short
