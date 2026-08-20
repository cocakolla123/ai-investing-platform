"""Backfill the three empty event tables — the four dark detectors' feedstock.

    python -m dynamo.cli.backfill_events insider  --days 90
    python -m dynamo.cli.backfill_events filings  --days 90
    python -m dynamo.cli.backfill_events earnings --start 2026-07-01
    python -m dynamo.cli.backfill_events status

Why
---
``insider_tx``, ``earnings_events`` and ``filings`` all held **0 rows**. The
adapters, the assets and the detectors all existed and were all correct;
nothing had ever put history in the tables, so ``insider_cluster``,
``earnings_surprise`` and ``filing_event`` could not fire even in principle,
and the insider frame inside ``FeatureInputs`` was empty every night. Cluster
buying by insiders is the classic under-the-radar signal, and it was dark for
want of rows, not for want of code.

Deliberately NOT in ``dynamo.cli.backfill``: that module is the ingestion
CLI. These are one-shot history sweeps, they are chunked and resumable, and
they will stop being interesting once the nightly assets have run for a
quarter.

Scope, stated rather than discovered later
------------------------------------------
* **insider** — EDGAR is free but rate-limited to 10 req/s, and a Form 4 costs
  one request for its ownership XML on top of the index page. Market-wide that
  is ~1,000 filings a day. So the sweep is restricted to a symbol list
  (default ``config/training_universe_wide.txt``, ~1,315 liquid names) whose
  CIKs are filtered BEFORE any XML is fetched: about 44% of daily Form 4s
  touch that universe, i.e. ~430 XML fetches per session day, ~1 minute of
  wall clock per day of history. Chunked one day at a time and committed per
  day, so a kill costs at most one day.
* **filings** — restricted to the forms ``filing_event`` actually acts on
  (8-K and the material forms in ``dynamo.data.detectors.MATERIAL_FORMS``).
  Sweeping every form type would be ~10x the volume for rows no detector
  reads. One index request per form per day; no per-filing fetch.
* **earnings** — Finnhub's free ``/calendar/earnings`` serves **from
  2026-07-01 onward only** (earlier windows return an empty calendar; the
  first day that actually carries events is 2026-07-15) and truncates a
  response at 1,500 events without paginating or saying so, hence
  ``--chunk-days`` and the INCOMPLETE report. Historical EPS *actuals* before
  that floor are not obtainable free at universe scale: Alpha Vantage has
  them per symbol at 25 requests/day, i.e. 53 days of quota for 1,315 names.
  ``earnings_surprise`` therefore fires on the actuals inside the served
  window and on everything that lands from now on; it does not get a
  synthetic history.

Every write is an upsert on the table's natural key
(``filings.accession_no``; ``insider_tx`` on
``(accession_no, symbol, insider_name, tx_type)``; ``earnings_events`` on
``(symbol, event_date)``), so re-running any command converges rather than
duplicating.

A multi-hour sweep WILL meet an EDGAR 500. Aborting throws away the hours
already spent; swallowing it leaves a hole in the table that reads as a quiet
day. Both EDGAR sweeps therefore skip the day, keep going, and **exit
non-zero naming every day to re-run** — the same discipline as the truncated
Finnhub window. A zero exit means the window is complete.

ZERO LLM calls; real EDGAR/Finnhub network by design.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_UNIVERSE_FILE = Path("config/training_universe_wide.txt")
DEFAULT_INSIDER_DAYS = 90
DEFAULT_FILINGS_DAYS = 90
# Finnhub's free calendar starts here; earlier windows come back empty.
FINNHUB_CALENDAR_FLOOR = date(2026, 7, 1)
FINNHUB_CHUNK_DAYS = 7  # keeps most responses under the truncation cap
# Finnhub silently truncates a calendar response here — it does not paginate
# and does not say it truncated, so the caller has to detect the boundary.
FINNHUB_RESPONSE_CAP = 1500
EARNINGS_LOOKAHEAD_DAYS = 120

# The forms ``filing_event`` reacts to. 8-K carries the item codes; the rest
# are material by form alone (see dynamo.data.detectors.MATERIAL_FORMS).
#
# These are EDGAR full-text-index ``root_forms`` values, which are NOT the SEC
# form codes: the index says "SCHEDULE 13D" where the detector's MATERIAL_FORMS
# says "SC 13D", and `"SCHEDULE 13D".startswith("SC 13D")` is False — so a
# beneficial-ownership filing stored under the index's own name is a row the
# detector reads and silently ignores. Query by the index's name, store the
# form code.
DEFAULT_FILING_FORMS = ("8-K", "S-1", "SCHEDULE 13D", "SCHEDULE 13G")
EDGAR_ROOT_FORM_TO_FORM_TYPE = {
    "SCHEDULE 13D": "SC 13D",
    "SCHEDULE 13G": "SC 13G",
}


def normalized_form_type(form_type: str) -> str:
    """EDGAR's human-readable root form -> the SEC form code detectors key on."""
    return EDGAR_ROOT_FORM_TO_FORM_TYPE.get(form_type.upper(), form_type)


# -- injectable composition seams (tests monkeypatch these) -----------------


def _settings() -> Any:
    from dynamo.config.settings import Settings

    return Settings()


def _repo(database_url: str | None) -> tuple[Any, Any]:
    from dynamo.persistence.db import make_engine
    from dynamo.persistence.repositories.market import MarketDataRepository

    engine = make_engine(database_url or _settings().database_url)
    return engine, MarketDataRepository(engine)


def _edgar(settings: Any) -> Any:
    from dynamo.data.edgar import EdgarClient

    return EdgarClient(settings.edgar_user_agent)


def _finnhub(settings: Any) -> Any:
    from dynamo.data.finnhub import FinnhubClient

    return FinnhubClient(settings.finnhub_api_key)


# -- shared helpers ---------------------------------------------------------


def read_symbol_file(path: Path) -> list[str]:
    """Symbols from a universe list: one per line, ``#`` comments ignored."""
    out: list[str] = []
    for line in path.read_text().splitlines():
        symbol = line.split("#", 1)[0].strip()
        if symbol:
            out.append(symbol.upper())
    return sorted(set(out))


def cik_map(engine: Any, symbols: Sequence[str] | None = None) -> dict[str, str]:
    """symbol -> CIK for the given symbols (all instruments when ``None``)."""
    from sqlalchemy import select

    from dynamo.persistence import schema

    t = schema.instruments
    stmt = select(t.c.symbol, t.c.cik)
    if symbols is not None:
        stmt = stmt.where(t.c.symbol.in_(list(symbols)))
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return {str(sym): str(cik) for sym, cik in rows if cik}


def table_counts(engine: Any) -> dict[str, int]:
    from sqlalchemy import func, select

    from dynamo.persistence import schema

    tables = {
        "insider_tx": schema.insider_tx,
        "earnings_events": schema.earnings_events,
        "filings": schema.filings,
    }
    with engine.connect() as conn:
        return {
            name: int(conn.execute(select(func.count()).select_from(t)).scalar_one())
            for name, t in tables.items()
        }


def day_range(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def week_chunks(start: date, end: date, size: int) -> Iterable[tuple[date, date]]:
    lo = start
    while lo <= end:
        hi = min(lo + timedelta(days=size - 1), end)
        yield lo, hi
        lo = hi + timedelta(days=1)


def _window(args: argparse.Namespace, default_days: int) -> tuple[date, date]:
    end: date = args.end or datetime.now(UTC).date()
    start: date = args.start or end - timedelta(days=args.days or default_days)
    return start, end


def _print_counts(label: str, counts: dict[str, int]) -> None:
    body = " | ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    print(f"{label}: {body}")


def _report_gaps(what: str, gaps: Sequence[date]) -> int:
    """Exit code for a sweep that skipped days, naming every one of them.

    EDGAR 500s and rate-limits transiently, and a multi-hour sweep WILL hit
    one. Aborting throws away the hours already spent; swallowing the error
    leaves a hole in the table that looks exactly like a quiet day. So: keep
    going, then refuse to exit 0 and print the days to re-run.
    """
    if not gaps:
        return 0
    listed = ", ".join(d.isoformat() for d in gaps[:20])
    more = "" if len(gaps) <= 20 else f" (+{len(gaps) - 20} more)"
    print(
        f"INCOMPLETE: {what} skipped {len(gaps)} day(s) after provider errors: "
        f"{listed}{more} — re-run those windows; the upserts converge",
        file=sys.stderr,
    )
    return 1


# -- insider transactions ---------------------------------------------------


def _cmd_insider(args: argparse.Namespace) -> int:
    """Form 4 buys/sells for a symbol list, one session day at a time."""
    engine, repo = _repo(args.database_url)
    start, end = _window(args, DEFAULT_INSIDER_DAYS)

    symbols = None if args.all_symbols else read_symbol_file(args.universe_file)
    ciks = cik_map(engine, symbols)
    if symbols is not None and not ciks:
        print(
            "ERROR: none of the requested symbols carry a CIK — run "
            "`python -m dynamo.cli.reference ciks` first",
            file=sys.stderr,
        )
        return 1
    scope = "the whole market" if symbols is None else f"{len(ciks)} symbols with a CIK"
    print(f"insider backfill {start} -> {end} over {scope}")
    _print_counts("before", table_counts(engine))

    from dynamo.data.errors import DataGap, ProviderDown, RateLimited

    edgar = _edgar(_settings())
    started = time.monotonic()
    total = 0
    days = 0
    gaps: list[date] = []
    for day in day_range(start, end):
        if day.weekday() >= 5:  # EDGAR does not accept filings at the weekend
            continue
        try:
            txs = list(
                edgar.insider_transactions_in_range(
                    day, day, ciks=None if symbols is None else ciks.values()
                )
            )
        except (ProviderDown, RateLimited, DataGap) as exc:
            gaps.append(day)
            print(f"  {day}: SKIPPED — {type(exc).__name__}: {exc}", flush=True)
            continue
        written = repo.upsert_insider_tx(txs) if txs else 0
        total += written
        days += 1
        print(
            f"  {day}: {len(txs)} transactions, {written} rows "
            f"({time.monotonic() - started:.0f}s elapsed)",
            flush=True,
        )
    print(f"insider backfill: {total} rows over {days} session days")
    _print_counts("after", table_counts(engine))
    return _report_gaps("insider backfill", gaps)


# -- filings ----------------------------------------------------------------


def _cmd_filings(args: argparse.Namespace) -> int:
    """Material filings (8-K + S-1/13D/13G) day by day, form by form."""
    engine, repo = _repo(args.database_url)
    start, end = _window(args, DEFAULT_FILINGS_DAYS)
    forms = list(args.forms)
    print(f"filings backfill {start} -> {end} for forms {forms}")
    _print_counts("before", table_counts(engine))

    # The index names a symbol only when a display_name carries one; fill the
    # rest from instruments, because filing_event skips a symbol-less row.
    from dynamo.data.errors import DataGap, ProviderDown, RateLimited

    by_cik = {cik.lstrip("0"): sym for sym, cik in cik_map(engine).items()}
    edgar = _edgar(_settings())
    started = time.monotonic()
    total = 0
    resolved = 0
    gaps: list[date] = []
    for day in day_range(start, end):
        if day.weekday() >= 5:
            continue
        batch: list[Any] = []
        try:
            for form in forms:
                for filing in edgar.filings_in_range(form, day, day):
                    update: dict[str, Any] = {}
                    if filing.symbol is None:
                        symbol = by_cik.get(filing.cik.lstrip("0"))
                        if symbol is not None:
                            resolved += 1
                            update["symbol"] = symbol
                    normalized = normalized_form_type(filing.form_type)
                    if normalized != filing.form_type:
                        update["form_type"] = normalized
                    batch.append(filing.model_copy(update=update) if update else filing)
        except (ProviderDown, RateLimited, DataGap) as exc:
            gaps.append(day)
            print(f"  {day}: SKIPPED — {type(exc).__name__}: {exc}", flush=True)
            continue
        written = repo.upsert_filings(batch) if batch else 0
        total += written
        print(
            f"  {day}: {len(batch)} filings, {written} rows "
            f"({time.monotonic() - started:.0f}s elapsed)",
            flush=True,
        )
    print(f"filings backfill: {total} rows ({resolved} symbols filled in from instruments)")
    _print_counts("after", table_counts(engine))
    return _report_gaps("filings backfill", gaps)


# -- earnings ---------------------------------------------------------------


def _cmd_earnings(args: argparse.Namespace) -> int:
    """Finnhub earnings calendar over everything the free tier serves."""
    settings = _settings()
    if not settings.finnhub_api_key:
        print("ERROR: FINNHUB_API_KEY is unset — earnings cannot run", file=sys.stderr)
        return 1
    engine, repo = _repo(args.database_url)
    start: date = args.start or FINNHUB_CALENDAR_FLOOR
    end: date = args.end or datetime.now(UTC).date() + timedelta(days=EARNINGS_LOOKAHEAD_DAYS)
    if start < FINNHUB_CALENDAR_FLOOR:
        print(
            f"note: Finnhub's free calendar starts {FINNHUB_CALENDAR_FLOOR}; "
            f"clamping --start {start}"
        )
        start = FINNHUB_CALENDAR_FLOOR
    if start > end:
        print(f"ERROR: --start {start} is after --end {end}", file=sys.stderr)
        return 1
    chunk = max(int(args.chunk_days or FINNHUB_CHUNK_DAYS), 1)
    print(f"earnings backfill {start} -> {end} in {chunk}-day chunks")
    _print_counts("before", table_counts(engine))

    finnhub = _finnhub(settings)
    total = 0
    vintages = 0
    with_actuals = 0
    truncated: list[str] = []
    for lo, hi in week_chunks(start, end, chunk):
        events = list(finnhub.earnings_calendar(lo, hi))
        if not events:
            print(f"  {lo}..{hi}: no events")
            continue
        actuals = sum(1 for e in events if e.eps_act is not None)
        with_actuals += actuals
        total += repo.upsert_earnings(events)
        vintages += repo.append_estimate_vintages(events)
        print(f"  {lo}..{hi}: {len(events)} events ({actuals} with eps_act)", flush=True)
        if len(events) >= FINNHUB_RESPONSE_CAP:
            truncated.append(f"{lo}..{hi}")
            print(
                f"    WARNING: {lo}..{hi} hit Finnhub's {FINNHUB_RESPONSE_CAP}-event "
                "response cap — this window is TRUNCATED"
            )
    print(
        f"earnings backfill: {total} rows, {with_actuals} carrying an EPS actual "
        f"(what earnings_surprise needs), {vintages} estimate vintages appended"
    )
    _print_counts("after", table_counts(engine))
    if truncated:
        # Loud and non-zero: a truncated window looks exactly like a complete
        # one in the table, and silently missing earnings is worse than none.
        print(
            f"INCOMPLETE: {len(truncated)} window(s) truncated at the response "
            f"cap ({', '.join(truncated)}) — re-run those with a smaller "
            f"--chunk-days (current: {chunk})",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    _print_counts("event tables", table_counts(_repo(args.database_url)[0]))
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    return _cmd_earnings(args) or _cmd_filings(args) or _cmd_insider(args)


# -- parser -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo.cli.backfill_events",
        description="Backfill insider_tx / earnings_events / filings history",
    )
    parser.add_argument("--database-url", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    def _window_args(p: argparse.ArgumentParser, default_days: int) -> None:
        p.add_argument("--start", type=date.fromisoformat, default=None)
        p.add_argument("--end", type=date.fromisoformat, default=None, help="default: today")
        p.add_argument(
            "--days",
            type=int,
            default=default_days,
            help=f"days back from --end when --start is omitted (default {default_days})",
        )

    ins = sub.add_parser("insider", help="Form 4 transactions for a symbol list")
    _window_args(ins, DEFAULT_INSIDER_DAYS)
    ins.add_argument(
        "--universe-file",
        type=Path,
        default=DEFAULT_UNIVERSE_FILE,
        help=f"symbol list to restrict the sweep to (default {DEFAULT_UNIVERSE_FILE})",
    )
    ins.add_argument(
        "--all-symbols",
        action="store_true",
        help="sweep every Form 4 in the window instead (~1,000 XML fetches per "
        "session day — hours, not minutes)",
    )

    fil = sub.add_parser("filings", help="material filings (8-K, S-1, 13D/G)")
    _window_args(fil, DEFAULT_FILINGS_DAYS)
    fil.add_argument("--forms", nargs="+", default=list(DEFAULT_FILING_FORMS))

    ear = sub.add_parser("earnings", help="Finnhub earnings calendar + estimate vintages")
    ear.add_argument(
        "--start",
        type=date.fromisoformat,
        default=None,
        help=f"default (and floor): {FINNHUB_CALENDAR_FLOOR}",
    )
    ear.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help=f"default: today + {EARNINGS_LOOKAHEAD_DAYS}d",
    )
    ear.add_argument(
        "--chunk-days",
        type=int,
        default=FINNHUB_CHUNK_DAYS,
        help=f"days per request (default {FINNHUB_CHUNK_DAYS}); shrink it for "
        f"busy reporting weeks that hit the {FINNHUB_RESPONSE_CAP}-event cap",
    )

    sub.add_parser("status", help="row counts for the three tables; writes nothing")
    all_p = sub.add_parser("all", help="earnings, then filings, then insider")
    _window_args(all_p, DEFAULT_FILINGS_DAYS)
    all_p.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    all_p.add_argument("--all-symbols", action="store_true")
    all_p.add_argument("--forms", nargs="+", default=list(DEFAULT_FILING_FORMS))
    return parser


_COMMANDS = {
    "insider": _cmd_insider,
    "filings": _cmd_filings,
    "earnings": _cmd_earnings,
    "status": _cmd_status,
    "all": _cmd_all,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for name, default in (
        ("start", None),
        ("end", None),
        ("days", None),
        ("forms", list(DEFAULT_FILING_FORMS)),
        ("universe_file", DEFAULT_UNIVERSE_FILE),
        ("all_symbols", False),
        ("chunk_days", FINNHUB_CHUNK_DAYS),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
