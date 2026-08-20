"""defeatbeta-api earnings-call transcripts — bulk fallback/backfill source.

The ``transcripts`` extra (``uv sync --extra transcripts``) installs
``defeatbeta-api``, which serves transcripts from a HuggingFace dataset via
duckdb — no API key, no per-request quota, but hours-to-days behind the tape.
The import is guarded: without the extra every call logs one skip line and
returns empty (doc 01 §6 graceful degradation), keeping the core install slim.

Library surface (defeatbeta-api)::

    Ticker(symbol).earning_call_transcripts()
        .get_transcripts_list()   # symbol, fiscal_year, fiscal_quarter, report_date
        .get_transcript(fiscal_year, fiscal_quarter)  # paragraph_number, speaker, content
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd
import structlog

from dynamo.domain.models.market import Transcript

log = structlog.get_logger(__name__)

SOURCE = "defeatbeta"


def _load_ticker_factory() -> Any | None:
    """The ``Ticker`` class when the ``transcripts`` extra is installed, else None."""
    try:
        module = importlib.import_module("defeatbeta_api.data.ticker")
    except Exception:  # ImportError or any of the package's own import-time failures
        return None
    return getattr(module, "Ticker", None)


class DefeatBetaTranscripts:
    """Bulk transcript fetch for a symbol list; every failure degrades to empty."""

    def __init__(self, ticker_factory: Any | None = None) -> None:
        self._factory = ticker_factory

    @property
    def available(self) -> bool:
        if self._factory is None:
            self._factory = _load_ticker_factory()
        return self._factory is not None

    def transcripts(self, symbols: Sequence[str], *, since: date) -> list[Transcript]:
        """Transcripts reported on/after ``since`` for ``symbols`` (may be empty)."""
        if not self.available:
            log.info(
                "defeatbeta_skipped",
                reason="transcripts extra not installed (uv sync --extra transcripts)",
            )
            return []
        out: list[Transcript] = []
        for symbol in symbols:
            try:
                out.extend(self._for_symbol(symbol, since))
            except Exception as exc:  # never let one symbol break the batch
                log.warning("defeatbeta_symbol_failed", symbol=symbol, error=str(exc))
        return out

    def _for_symbol(self, symbol: str, since: date) -> list[Transcript]:
        assert self._factory is not None
        holder = self._factory(symbol).earning_call_transcripts()
        listing = holder.get_transcripts_list()
        if listing is None or len(listing) == 0:
            return []
        out: list[Transcript] = []
        for row in pd.DataFrame(listing).to_dict(orient="records"):
            report_date = _as_date(row.get("report_date"))
            year = row.get("fiscal_year")
            quarter = row.get("fiscal_quarter")
            if report_date is None or year is None or quarter is None:
                continue
            if report_date < since:
                continue
            paragraphs = holder.get_transcript(int(year), int(quarter))
            transcript = _build_transcript(symbol, report_date, int(year), int(quarter), paragraphs)
            if transcript is not None:
                out.append(transcript)
        return out


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    return None  # NaT and friends


def _build_transcript(
    symbol: str, report_date: date, year: int, quarter: int, paragraphs: Any
) -> Transcript | None:
    if paragraphs is None or len(paragraphs) == 0:
        return None
    parts: list[str] = []
    speakers: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in pd.DataFrame(paragraphs).to_dict(orient="records"):
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        speaker = str(row.get("speaker") or "unknown").strip()
        parts.append(f"{speaker}: {content}")
        if speaker not in seen:
            seen.add(speaker)
            speakers.append({"speaker": speaker, "title": ""})
    if not parts:
        return None
    return Transcript(
        symbol=symbol,
        event_date=report_date,
        source=SOURCE,
        period=f"{year}Q{quarter}",
        content="\n\n".join(parts),
        speakers=tuple(speakers),
    )
