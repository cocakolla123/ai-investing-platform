"""Three-tier universe manager. Design: doc 02 §2.

- Tier 1 (core): seeded from ``config/universe_seed.txt`` when present, else a
  built-in liquid starter list (~40 mega/large caps + ~10 ETFs). The starter
  list is deliberately small — expand it by editing the seed file; permanent
  changes always require human approval (``approved_by``).
- Tier 3 (promoted): temporary full treatment for a configurable window
  (default 30 days), written with an explicit expiry.
- Recommendations (add to core / demote) are *ackable alert rows* — the
  platform never changes Tier 1 by itself; :meth:`approve_core_addition` /
  :meth:`approve_core_removal` are the only write paths and both record
  ``approved_by``.

All membership transitions write ``universe_membership`` rows with reasons
(point-in-time backbone for survivorship-safe backtests).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import structlog

from dynamo.domain.models.market import Instrument, Tier, UniverseMembership

log = structlog.get_logger(__name__)

DEFAULT_SEED_PATH = "config/universe_seed.txt"

# Built-in starter list: ~40 liquid large caps. The real core universe
# (~500-600 names, doc 02 §2) is grown from here via the seed file and
# approved recommendations.
STARTER_STOCKS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "BRK.B",
    "JPM",
    "V",
    "UNH",
    "JNJ",
    "XOM",
    "PG",
    "MA",
    "HD",
    "COST",
    "ABBV",
    "MRK",
    "AVGO",
    "PEP",
    "KO",
    "WMT",
    "BAC",
    "CRM",
    "AMD",
    "ORCL",
    "NFLX",
    "LIN",
    "DIS",
    "ADBE",
    "TMO",
    "CSCO",
    "ACN",
    "MCD",
    "ABT",
    "INTU",
    "CAT",
    "IBM",
    "TXN",
)

# ~10 ETFs: broad index, sector SPDRs sample, treasuries (doc 02 §2 curated set).
STARTER_ETFS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "TLT",
)

# Crypto sleeve starter pairs (ADR-0013): Alpaca-supported, canonical
# "BASE/QUOTE" form. Joined into the Tier-1 seed when the sleeve is enabled
# (settings.crypto_enabled -> UniverseManager(include_crypto=...)). Mirrors
# the settings.crypto_universe default.
CRYPTO_STARTERS: tuple[str, ...] = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "AVAX/USD",
    "LINK/USD",
)


def seed_symbol_origins(
    *,
    include_crypto: bool,
    seed_path: str | Path = DEFAULT_SEED_PATH,
) -> list[tuple[str, str]]:
    """(symbol, origin) pairs from the seed file, else the built-in list.

    Crypto starter pairs (ADR-0013) are appended in BOTH branches when
    enabled — the equity seed file does not govern the crypto sleeve.
    Shared by :meth:`UniverseManager.seed_core` and the backfill CLI's
    ``universe --as-of`` backdate (which must target the same symbol set the
    seed wrote).
    """
    path = Path(seed_path)
    if path.exists():
        symbols = [
            line.strip().upper()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        out = [(s, f"seed file {path}") for s in symbols]
    else:
        out = [(s, "built-in starter list") for s in (*STARTER_STOCKS, *STARTER_ETFS)]
    if include_crypto:
        seen = {s for s, _ in out}
        out.extend((s, "crypto starter list") for s in CRYPTO_STARTERS if s not in seen)
    return out


class UniverseStore(Protocol):
    """The slice of ``MarketDataRepository`` the manager reads/writes through."""

    def universe_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]: ...

    def upsert_universe_membership(self, memberships: Any) -> int: ...

    def close_membership(
        self, symbol: str, valid_from: date, valid_to: date, *, approved_by: str | None = None
    ) -> int: ...

    def add_alert(
        self,
        *,
        severity: str,
        name: str,
        message: str,
        ackable: bool = True,
        created_at: datetime | None = None,
    ) -> None: ...


class UniverseManager:
    """Point-in-time three-tier universe operations (doc 02 §2)."""

    def __init__(
        self,
        store: UniverseStore,
        *,
        seed_path: str = DEFAULT_SEED_PATH,
        default_promotion_days: int = 30,
        include_crypto: bool = True,
    ) -> None:
        self._store = store
        self._seed_path = Path(seed_path)
        self._default_promotion_days = default_promotion_days
        # ADR-0013: callers pass settings.crypto_enabled; the default matches
        # the settings default so existing wiring seeds the crypto pairs.
        self._include_crypto = include_crypto

    # -- seeding -----------------------------------------------------------

    def _seed_symbols(self) -> list[tuple[str, str]]:
        """See :func:`seed_symbol_origins` (shared with the backfill CLI)."""
        return seed_symbol_origins(include_crypto=self._include_crypto, seed_path=self._seed_path)

    def seed_core(self, as_of: date) -> int:
        """Seed Tier 1 once; idempotent (no-op when Tier 1 already populated)."""
        if self._store.universe_at(as_of, Tier.CORE):
            log.info("universe_seed_skipped_nonempty")
            return 0
        rows = [
            UniverseMembership(
                symbol=symbol,
                tier=Tier.CORE,
                valid_from=as_of,
                valid_to=None,
                reason=f"core seed: {origin}",
                approved_by="seed",
            )
            for symbol, origin in self._seed_symbols()
        ]
        count = self._store.upsert_universe_membership(rows)
        log.info("universe_seeded", symbols=len(rows))
        return count

    # -- point-in-time queries ---------------------------------------------

    def members_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]:
        return self._store.universe_at(as_of, tier)

    def active_symbols(self, as_of: date) -> set[str]:
        """Symbols receiving full treatment: Tier 1 + live Tier-3 promotions."""
        core = {m.symbol for m in self.members_at(as_of, Tier.CORE)}
        promoted = {m.symbol for m in self.members_at(as_of, Tier.PROMOTED)}
        return core | promoted

    # -- Tier-3 promotion lifecycle ----------------------------------------

    def promote(
        self,
        symbol: str,
        *,
        as_of: date,
        reason: str,
        window_days: int | None = None,
    ) -> UniverseMembership:
        """Temporary Tier-1 treatment for ``window_days`` (default 30, doc 02 §2).

        Re-promoting an already-promoted symbol extends the window (continued
        flags extend, per the design).
        """
        window = window_days or self._default_promotion_days
        expiry = as_of + timedelta(days=window)
        existing = [m for m in self.members_at(as_of, Tier.PROMOTED) if m.symbol == symbol]
        if existing:
            current = existing[0]
            extended = UniverseMembership(
                symbol=symbol,
                tier=Tier.PROMOTED,
                valid_from=current.valid_from,
                valid_to=max(expiry, current.valid_to or expiry),
                reason=f"{current.reason}; extended: {reason}",
                approved_by=None,
            )
            self._store.upsert_universe_membership([extended])
            log.info("promotion_extended", symbol=symbol, until=str(extended.valid_to))
            return extended
        membership = UniverseMembership(
            symbol=symbol,
            tier=Tier.PROMOTED,
            valid_from=as_of,
            valid_to=expiry,
            reason=reason,
            approved_by=None,  # temporary — no human approval needed
        )
        self._store.upsert_universe_membership([membership])
        log.info("symbol_promoted", symbol=symbol, until=str(expiry), reason=reason)
        return membership

    def expire_promotions(self, as_of: date) -> list[str]:
        """Close any Tier-3 rows past their window; returns lapsed symbols.

        Exit path (a) in doc 02 §2: the window lapses quietly → back to Tier 2.
        """
        lapsed: list[str] = []
        for m in self._store.universe_at(as_of, Tier.PROMOTED):
            if m.valid_to is None:  # defensive: promotions always carry an expiry
                self._store.close_membership(m.symbol, m.valid_from, as_of)
                lapsed.append(m.symbol)
        # Rows whose valid_to already passed are excluded by universe_at and
        # need no write — they lapse naturally by interval semantics.
        if lapsed:
            log.info("promotions_expired", symbols=lapsed)
        return lapsed

    # -- recommendations (human-in-the-loop, doc 02 §2 exit paths b/c) -----

    def recommend_addition(self, symbol: str, rationale: str, *, now: datetime) -> None:
        """Sustained interest → ackable universe-change recommendation row."""
        self._store.add_alert(
            severity="info",
            name="universe_change_recommendation",
            message=f"ADD {symbol} to core universe: {rationale}",
            ackable=True,
            created_at=now,
        )
        log.info("universe_addition_recommended", symbol=symbol)

    def recommend_demotion(
        self,
        symbol: str,
        rationale: str,
        *,
        now: datetime,
        suggested_replacement: str | None = None,
    ) -> None:
        """Tier-1 name failing floors for 60d → demotion recommendation."""
        suffix = (
            f" (suggested replacement: {suggested_replacement})" if suggested_replacement else ""
        )
        self._store.add_alert(
            severity="info",
            name="universe_change_recommendation",
            message=f"DEMOTE {symbol} from core universe: {rationale}{suffix}",
            ackable=True,
            created_at=now,
        )
        log.info("universe_demotion_recommended", symbol=symbol)

    # -- approved permanent changes (the ONLY tier-1 write paths) ----------

    def approve_core_addition(
        self, symbol: str, *, as_of: date, reason: str, approved_by: str
    ) -> UniverseMembership:
        """Permanent Tier-1 add; requires a human ``approved_by`` (doc 02 §2)."""
        if not approved_by:
            raise ValueError("permanent tier-1 changes require approved_by")
        membership = UniverseMembership(
            symbol=symbol,
            tier=Tier.CORE,
            valid_from=as_of,
            valid_to=None,
            reason=reason,
            approved_by=approved_by,
        )
        self._store.upsert_universe_membership([membership])
        log.info("core_addition_approved", symbol=symbol, approved_by=approved_by)
        return membership

    def approve_core_removal(
        self, symbol: str, *, as_of: date, reason: str, approved_by: str
    ) -> int:
        """Permanent Tier-1 removal: closes the current interval with approval."""
        if not approved_by:
            raise ValueError("permanent tier-1 changes require approved_by")
        closed = 0
        for m in self._store.universe_at(as_of, Tier.CORE):
            if m.symbol == symbol:
                closed += self._store.close_membership(
                    symbol, m.valid_from, as_of, approved_by=approved_by
                )
        log.info("core_removal_approved", symbol=symbol, approved_by=approved_by, rows=closed)
        return closed


# GICS sectors for the built-in equity starters (post-2023 GICS: V/MA are
# Financials; GOOGL/META/NFLX/DIS are Communication Services). Keys must be
# GICS names recognized by :func:`default_sector_etf_map`. Seeded into
# ``instruments.sector`` by ``backfill universe`` so sector-relative labels,
# dossier sector context, and the risk engine's sector caps all work.
STARTER_SECTORS: dict[str, str] = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AMD": "Information Technology",
    "AVGO": "Information Technology",
    "ADBE": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "CSCO": "Information Technology",
    "IBM": "Information Technology",
    "INTU": "Information Technology",
    "ACN": "Information Technology",
    "TXN": "Information Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "JPM": "Financials",
    "BAC": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BRK.B": "Financials",
    "XOM": "Energy",
    "CAT": "Industrials",
    "LIN": "Materials",
}


_GICS_TO_SPDR: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
#: Every sector SPDR a symbol can be benchmarked against.
SECTOR_SPDRS: tuple[str, ...] = tuple(sorted(set(_GICS_TO_SPDR.values())))
#: The market benchmark for beta / residual features and the label fallback.
MARKET_BENCHMARK_SYMBOL = "SPY"


def default_sector_etf_map(sectors: Mapping[str, str]) -> dict[str, str]:
    """GICS sector → sector-SPDR map for the ``rel_strength`` detector.

    ``sectors`` maps symbol → GICS sector name (from ``instruments``).
    """
    return {
        symbol: _GICS_TO_SPDR[sector]
        for symbol, sector in sectors.items()
        if sector in _GICS_TO_SPDR
    }


def feature_input_symbols(universe: Sequence[str]) -> list[str]:
    """The symbols whose BARS the feature builder must see for ``universe``.

    Relative features (``rs_20d_sector``, beta, residual momentum) read a
    benchmark's prices, not just the scored names'. Training read
    ``universe + sector ETFs + SPY``; the live nightly builder and the replay
    runner read only the universe, so any benchmark outside it was NaN at
    serving time while fully populated in training — a train/serve skew that
    no test caught because each path was consistent with itself. All three
    paths now load exactly this set; features are still computed only for
    ``universe``.
    """
    return sorted({*universe, *SECTOR_SPDRS, MARKET_BENCHMARK_SYMBOL})


def seed_symbols_documented() -> Sequence[str]:
    """The built-in starter universe (user-expandable via the seed file)."""
    return (*STARTER_STOCKS, *STARTER_ETFS, *CRYPTO_STARTERS)


def crypto_instruments(symbols: Sequence[str]) -> list[Instrument]:
    """Instrument rows for crypto pairs (ADR-0013).

    Crypto has no GICS sector; committee dossiers and the risk engine's
    sector caps key off ``instruments.sector``, so crypto pairs get the
    literal sector ``"crypto"`` — they share one sector bucket, which is the
    conservative reading (highly correlated assets share a cap). Upserted by
    the nightly ``crypto_bars`` asset (idempotent).
    """
    return [
        Instrument(
            symbol=symbol,
            name=symbol,
            exchange="alpaca_crypto",
            sector="crypto",
        )
        for symbol in symbols
    ]


# --- Listing-date backdating (D3) ------------------------------------------
#
# The Tier-2 surveillance universe is built from Alpaca's tradable-asset list,
# which is a TODAY snapshot with no listing dates on it, so every row was
# written with `valid_from` = the day the build ran (2026-08-17). That makes
# the whole 11,455-name universe INVISIBLE to any point-in-time read before
# that date: `universe_at(2026-03-03, tier=2)` returns zero names, and a
# replay over the campaign window discovers nothing.
#
# The tempting fix — backdate everything to a fixed early date — injects
# look-ahead: a company that IPO'd in 2025-06 would appear in a 2024
# surveillance universe, and the run would "discover" a name that did not
# exist. The honest proxy available in this database is the symbol's FIRST
# BAR: the vendor has no history before a symbol starts trading, so
# `min(bars_1d.ts)` bounds the listing date from above without any outside
# knowledge. It is imperfect in one direction only (a symbol listed before
# the backfill's start date looks younger than it is), which is the
# conservative direction: it under-states, never over-states, membership.


@dataclass(frozen=True)
class BackdateRebase:
    """One membership row moving from ``old_valid_from`` to ``new_valid_from``."""

    symbol: str
    tier: int
    old_valid_from: date
    new_valid_from: date


@dataclass(frozen=True)
class BackdatePlan:
    """What :func:`plan_listing_date_backdate` would change, and what it won't.

    Nothing here touches the database; the plan is the reviewable artifact and
    the applier is dumb on purpose.
    """

    rebases: tuple[BackdateRebase, ...]
    already_backdated: tuple[str, ...]
    """Rows whose ``valid_from`` already equals the first-bar date — this is
    what makes a re-run a no-op."""
    no_bars: tuple[str, ...]
    """Symbols with no bars at all: their listing date is unknowable here, so
    they keep the conservative late ``valid_from`` and stay invisible."""
    first_bar_after_valid_from: tuple[str, ...]
    """Symbols whose first bar is LATER than their current ``valid_from``.
    Moving them forward would remove them from a universe they were already
    in, so they are left alone — backdating only ever moves dates earlier."""

    def summary(self) -> str:
        return (
            f"{len(self.rebases)} rows to backdate, "
            f"{len(self.already_backdated)} already at their first bar, "
            f"{len(self.no_bars)} with no bars (left late), "
            f"{len(self.first_bar_after_valid_from)} already earlier than their first bar"
        )


def plan_listing_date_backdate(
    memberships: Sequence[UniverseMembership],
    first_bar: Mapping[str, date],
    *,
    tier: Tier = Tier.SURVEILLANCE,
) -> BackdatePlan:
    """Rebase each ``tier`` membership's ``valid_from`` onto its first bar.

    ``memberships`` may contain every tier; rows of any other tier are
    ignored entirely — that is the guarantee that the 55 hand-approved Tier-1
    CORE rows survive this untouched.

    Idempotent by construction: a row whose ``valid_from`` already equals its
    first-bar date produces no rebase, so applying the plan twice changes
    nothing the second time.
    """
    rebases: list[BackdateRebase] = []
    already: list[str] = []
    missing: list[str] = []
    later: list[str] = []
    for member in memberships:
        if int(member.tier) != int(tier):
            continue
        listed = first_bar.get(member.symbol)
        if listed is None:
            missing.append(member.symbol)
            continue
        if listed == member.valid_from:
            already.append(member.symbol)
        elif listed > member.valid_from:
            later.append(member.symbol)
        else:
            rebases.append(
                BackdateRebase(
                    symbol=member.symbol,
                    tier=int(member.tier),
                    old_valid_from=member.valid_from,
                    new_valid_from=listed,
                )
            )
    return BackdatePlan(
        rebases=tuple(rebases),
        already_backdated=tuple(sorted(already)),
        no_bars=tuple(sorted(missing)),
        first_bar_after_valid_from=tuple(sorted(later)),
    )


def backdated_memberships(
    memberships: Sequence[UniverseMembership], plan: BackdatePlan
) -> list[UniverseMembership]:
    """The rebased rows to write, carrying every other field unchanged."""
    moved = {(r.symbol, r.old_valid_from): r.new_valid_from for r in plan.rebases}
    out: list[UniverseMembership] = []
    for member in memberships:
        new_from = moved.get((member.symbol, member.valid_from))
        if new_from is None:
            continue
        out.append(
            UniverseMembership(
                symbol=member.symbol,
                tier=member.tier,
                valid_from=new_from,
                valid_to=member.valid_to,
                reason=member.reason,
                approved_by=member.approved_by,
            )
        )
    return out
