"""News selection for the dossier (campaign findings, item 11).

Measured on the 300 dossiers of the control run: only **18.9%** of the news
items in a ticker's digest actually named that company. NVDA's digest carried
items about Disney, Micron, Buffett, AMD and OpenAI — all tagged NVDA by the
vendor, none about NVDA. The bear then argued from a digest that had nothing
in it, and won.

:func:`partition_news` splits a symbol's tagged items into the ones that name
the company (headline or summary mentions the ticker or a distinctive word of
its name) and the rest, which the dossier renders separately under
``## Market news`` so the model can tell context from evidence. Recency order
is preserved: selection is newest-first, and the caller keeps that order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from dynamo.domain.models.market import NewsItem

# Corporate-form and filler words that make a company name match everything.
_NAME_STOPWORDS = frozenset(
    {
        "inc",
        "inc.",
        "incorporated",
        "corp",
        "corp.",
        "corporation",
        "co",
        "co.",
        "company",
        "companies",
        "the",
        "and",
        "of",
        "plc",
        "ltd",
        "ltd.",
        "limited",
        "holdings",
        "holding",
        "group",
        "class",
        "common",
        "stock",
        "sa",
        "nv",
        "ag",
        "trust",
        "fund",
        "etf",
        "index",
        "shares",
    }
)


def name_tokens(symbol: str, name: str | None = None) -> tuple[str, ...]:
    """Distinctive lowercase tokens that identify the company in prose."""
    tokens = {symbol.lower()}
    if symbol.endswith(".B") or symbol.endswith(".A"):
        tokens.add(symbol.split(".")[0].lower())
    for raw in re.split(r"[^A-Za-z0-9]+", name or ""):
        token = raw.lower()
        if len(token) >= 3 and token not in _NAME_STOPWORDS:
            tokens.add(token)
    return tuple(sorted(tokens))


def mentions_symbol(item: NewsItem, tokens: Sequence[str]) -> bool:
    """Whether the item's own text names the company (word-boundary match)."""
    text = f"{item.headline} {item.summary or ''}".lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", text) for t in tokens)


def partition_news(
    items: Sequence[NewsItem],
    *,
    symbol: str,
    name: str | None = None,
    max_items: int = 10,
    max_market_items: int = 5,
) -> tuple[tuple[NewsItem, ...], tuple[NewsItem, ...]]:
    """(items about the symbol, other items) — newest first, deduped by id.

    Only items whose ``symbols`` array carries the dossier's symbol are
    considered at all; of those, the ones that name the company lead, and the
    remainder are offered as clearly-labelled market context (capped, and only
    up to the digest budget the ticker's own news leaves unused).
    """
    tagged = [item for item in items if symbol in tuple(item.symbols)]
    ordered = sorted(tagged, key=lambda n: (n.published_at, n.id), reverse=True)
    seen: set[str] = set()
    primary: list[NewsItem] = []
    other: list[NewsItem] = []
    tokens = name_tokens(symbol, name)
    for item in ordered:
        if item.id in seen:
            continue
        seen.add(item.id)
        (primary if mentions_symbol(item, tokens) else other).append(item)
    budget = max(0, min(max_market_items, max_items - len(primary[:max_items])))
    return tuple(primary[:max_items]), tuple(other[:budget])
