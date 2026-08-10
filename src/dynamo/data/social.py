"""Social-burst detector — pure functions over social ``NewsItem`` mentions.

Design: doc 02 §2 (detector catalog pattern) + owner-approved social layer
design. Kept separate from ``detectors.py`` deliberately: social activity is
an AWARENESS TRIGGER for Tier-2 surveillance (feeds the Haiku materiality
triage), never a direct buy signal, and the text itself is untrusted
(pump-and-dump manipulation surface) — only mention COUNTS are used here.

Zero LLM tokens; emits typed :class:`SurveillanceFlag` objects
(detector='social_burst') with a 0-1 severity, as-of discipline throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import cast

import pandas as pd

from dynamo.domain.models.market import NewsItem, SurveillanceFlag

COUNT_COLUMNS: tuple[str, ...] = ("symbol", "day", "source", "count")

# 1 / Phi^-1(3/4): scales MAD to be sigma-consistent for normal data.
_MAD_SIGMA = 1.4826


def _sev(value: float) -> float:
    """Clamp a severity into [0, 1] (detectors.py convention)."""
    return max(0.0, min(1.0, value))


def _day_total(daily: pd.Series, day: date) -> float:  # pyright: ignore[reportMissingTypeArgument]
    value = daily.get(day, 0)
    return 0.0 if value is None else float(value)  # pyright: ignore[reportArgumentType]


def mention_counts(
    items: Sequence[NewsItem],
    *,
    as_of: datetime,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """Per-symbol daily mention counts, split by source.

    Returns a frame with columns ``symbol, day, source, count`` where ``day``
    is the calendar date (UTC) of the mention. Only items published within
    ``(as_of - lookback_days, as_of]`` count (as-of discipline: nothing after
    ``as_of``). Each item counts once per symbol (dedup by ``(id, symbol)``).
    """
    start = as_of - timedelta(days=lookback_days)
    counts: dict[tuple[str, date, str], int] = {}
    seen: set[tuple[str, str]] = set()
    for item in items:
        if item.published_at <= start or item.published_at > as_of:
            continue
        for symbol in item.symbols:
            key = (item.id, symbol)
            if key in seen:
                continue
            seen.add(key)
            bucket = (symbol, item.published_at.date(), item.source)
            counts[bucket] = counts.get(bucket, 0) + 1
    rows = sorted(counts.items())
    return pd.DataFrame(
        {
            "symbol": [sym for (sym, _, _), _ in rows],
            "day": [day for (_, day, _), _ in rows],
            "source": [src for (_, _, src), _ in rows],
            "count": [n for _, n in rows],
        }
    )


def social_burst(
    counts_frame: pd.DataFrame,
    *,
    as_of: datetime,
    zscore_threshold: float = 4.0,
    min_mentions: int = 10,
    lookback_days: int = 14,
) -> list[SurveillanceFlag]:
    """Flag symbols whose today mention count bursts vs their own baseline.

    z-score of today's total against the prior ``lookback_days`` daily totals
    (zero-filled). When the baseline std is 0 (flat history), falls back to a
    robust median/MAD z-score; when MAD is also 0, the raw excess over the
    median is used as the score. Symbols below ``min_mentions`` today are
    never flagged (noise floor), regardless of z.
    """
    flags: list[SurveillanceFlag] = []
    if counts_frame.empty:
        return flags
    today = as_of.date()
    baseline_dates = [today - timedelta(days=d) for d in range(1, lookback_days + 1)]
    for symbol, grp in counts_frame.groupby("symbol", sort=True):
        daily = cast(pd.Series, grp.groupby("day")["count"].sum())  # pyright: ignore[reportMissingTypeArgument]
        today_count = int(_day_total(daily, today))
        if today_count < min_mentions:
            continue
        baseline = pd.Series([_day_total(daily, d) for d in baseline_dates])
        mean, std = float(baseline.mean()), float(baseline.std())
        if std > 0:
            z = (today_count - mean) / std
        else:
            median = float(baseline.median())
            mad = float((baseline - median).abs().median())
            z = (
                (today_count - median) / (_MAD_SIGMA * mad)
                if mad > 0
                else float(today_count) - median
            )
        if z <= zscore_threshold:
            continue
        today_sources = grp[grp["day"] == today].groupby("source")["count"].sum()
        sources = ",".join(f"{src}={int(n)}" for src, n in sorted(today_sources.items()))
        flags.append(
            SurveillanceFlag(
                symbol=str(symbol),
                detector="social_burst",
                severity=_sev(z / (2.0 * zscore_threshold)),
                details={
                    "today_count": float(today_count),
                    "baseline_mean": round(mean, 3),
                    "zscore": round(z, 3),
                    "sources": sources,
                },
                ts=as_of,
            )
        )
    return flags


def run_social_scan(
    reddit_items: Sequence[NewsItem],
    stocktwits_items: Sequence[NewsItem],
    *,
    as_of: datetime,
    zscore_threshold: float = 4.0,
    min_mentions: int = 10,
    lookback_days: int = 14,
) -> list[SurveillanceFlag]:
    """Combine social sources → mention counts → burst flags (the entry point)."""
    counts = mention_counts(
        [*reddit_items, *stocktwits_items], as_of=as_of, lookback_days=lookback_days
    )
    return social_burst(
        counts,
        as_of=as_of,
        zscore_threshold=zscore_threshold,
        min_mentions=min_mentions,
        lookback_days=lookback_days,
    )
