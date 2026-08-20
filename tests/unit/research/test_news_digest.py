"""Symbol-filtering the news digest (81% of the control run's items were noise)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dynamo.domain.models.market import NewsItem
from dynamo.research.news_digest import name_tokens, partition_news


def item(
    news_id: str,
    headline: str,
    *,
    symbols: tuple[str, ...] = ("NVDA",),
    day: int = 1,
    summary: str = "",
) -> NewsItem:
    return NewsItem(
        id=news_id,
        symbols=symbols,
        headline=headline,
        summary=summary,
        source="test",
        published_at=datetime(2026, 6, day, 12, tzinfo=UTC),
    )


NVDA_ITEMS = (
    item("n1", "Nvidia beats on data-center revenue", day=1),
    item("n2", "Disney raises park prices", day=2),
    item("n3", "Micron guides higher", day=3),
    item("n4", "NVDA announces a buyback", day=4),
    item("n5", "Buffett trims a position", day=5),
)


def test_only_items_naming_the_subject_lead_the_digest() -> None:
    primary, market = partition_news(NVDA_ITEMS, symbol="NVDA", name="NVIDIA Corp")
    assert [n.id for n in primary] == ["n4", "n1"]  # newest first
    assert [n.id for n in market] == ["n5", "n3", "n2"]


def test_items_not_tagged_with_the_symbol_are_dropped_entirely() -> None:
    foreign = item("x1", "NVDA mentioned in passing", symbols=("AMD",), day=6)
    primary, market = partition_news((*NVDA_ITEMS, foreign), symbol="NVDA", name="NVIDIA Corp")
    assert "x1" not in {n.id for n in primary} | {n.id for n in market}


def test_recency_order_is_preserved_in_both_sections() -> None:
    primary, market = partition_news(NVDA_ITEMS, symbol="NVDA", name="NVIDIA Corp")
    assert [n.published_at for n in primary] == sorted(
        (n.published_at for n in primary), reverse=True
    )
    assert [n.published_at for n in market] == sorted(
        (n.published_at for n in market), reverse=True
    )


def test_company_name_matches_even_without_the_ticker() -> None:
    primary, _ = partition_news(
        (item("n9", "Walmart lifts guidance", symbols=("WMT",)),),
        symbol="WMT",
        name="Walmart Inc.",
    )
    assert [n.id for n in primary] == ["n9"]


def test_summary_counts_as_a_mention() -> None:
    primary, _ = partition_news(
        (item("n8", "Chip demand accelerates", summary="Nvidia named the main winner"),),
        symbol="NVDA",
        name="NVIDIA Corp",
    )
    assert [n.id for n in primary] == ["n8"]


def test_substring_matches_do_not_count() -> None:
    """A two-letter ticker must match as a word, not inside other words."""
    primary, market = partition_news(
        (
            item("n7", "Kalamazoo maker lifts guidance", symbols=("MA",)),
            item("n6", "MA raises its dividend", symbols=("MA",)),
        ),
        symbol="MA",
        name="Mastercard",
    )
    assert [n.id for n in primary] == ["n6"]
    assert [n.id for n in market] == ["n7"]


def test_caps_respect_the_digest_budget() -> None:
    many = tuple(item(f"m{i}", "Unrelated market wrap", day=1 + i % 28) for i in range(30))
    primary, market = partition_news(many, symbol="NVDA", name="NVIDIA", max_items=10)
    assert primary == ()
    assert len(market) == 5
    lots = tuple(item(f"p{i}", f"NVDA update {i}", day=1 + i % 28) for i in range(30))
    primary2, market2 = partition_news(lots, symbol="NVDA", name="NVIDIA", max_items=10)
    assert len(primary2) == 10
    assert market2 == ()


def test_deduplicates_by_id() -> None:
    duplicated = (NVDA_ITEMS[0], NVDA_ITEMS[0])
    primary, _ = partition_news(duplicated, symbol="NVDA", name="NVIDIA")
    assert [n.id for n in primary] == ["n1"]


@pytest.mark.parametrize(
    ("symbol", "name", "expected"),
    [
        ("BRK.B", "Berkshire Hathaway", ("berkshire", "brk", "brk.b", "hathaway")),
        ("KO", "The Coca-Cola Company", ("coca", "cola", "ko")),
    ],
)
def test_name_tokens_drop_corporate_filler(
    symbol: str, name: str, expected: tuple[str, ...]
) -> None:
    assert name_tokens(symbol, name) == expected
