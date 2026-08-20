"""How much price history the Tier-2 detector catalog actually needs.

Why this module exists
----------------------
``tier2_scan`` reads one bars window and hands the SAME frame to every
bar-based detector in :mod:`dynamo.data.detectors`. If that window is
shorter than the longest lookback any detector declares, the detectors do
not fail — they quietly compute the wrong thing, or silently never fire:

- ``range_break`` takes ``prior_high = highs[:-1].tail(lookback).max()``
  with ``lookback=252``. Given only ~167 sessions it reports a *3-month*
  high as a "52-week breakout". Measured against the live 11,258-name
  Tier-2 universe, 61% of the ``range_break`` flags produced off a 100
  calendar-day window were false 52-week breaks.
- ``range_bound`` gates on ``len(grp) >= max(vr_window + vr_lag + 1,
  touch_window + band_window, range_window)`` = 131 sessions and simply
  ``continue``s below it. Off a 100 calendar-day window it produced
  **zero** flags market-wide — structurally unable to fire, with no error
  and no log line.

Both failures are silent, so the fix cannot be "set a bigger number and
hope". This module derives the requirement *from the detectors themselves*
so that adding a longer-lookback detector breaks a test instead of
noiselessly no-op'ing.

The arithmetic
--------------
Session requirement per detector = the largest trailing-window parameter it
declares, plus :data:`COMBINATION_ALLOWANCE_SESSIONS`. The allowance covers
the small additive overheads the detectors layer on top of their headline
window (``range_break`` needs ``lookback`` *prior* bars plus the current
one; ``range_bound`` needs ``vr_window + vr_lag + 1`` = 11 sessions beyond
its largest single window). 16 sessions clears the worst of those with
room, and is itself asserted against the real gates in the guard test.

Today that maximum is ``range_break``: 252 + 16 = 268 sessions.

Converting sessions to the calendar days ``bars_window`` takes:
US equities trade ~252 sessions per 365.25-day year. :data:`SESSIONS_PER_
CALENDAR_DAY` deliberately uses 250, not 252, so the conversion *under*
estimates the sessions a window yields and errs toward asking for more
history. 268 sessions / 0.68446 = 392 calendar days minimum.

:data:`TIER2_SCAN_LOOKBACK_DAYS` is 420, not 392, for margin: 420 calendar
days yields 288 sessions on the live bar table (measured on AAPL/MSFT/JPM/
XOM through 2026-08-18), i.e. 20 sessions of slack over the 268 required —
about four trading weeks of holiday clustering and ingestion gaps. It also
matches the 420 the feature panel already uses (``_feature_inputs``), so
the two nightly windows stay in step.

Cost of the wider window, measured: ~300MB peak RSS and ~2s of extra query
time for the full Tier-2 universe. Nothing about that is load-bearing
enough to justify running the detectors on history they cannot use.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from dynamo.data import detectors

#: A parameter declares a trailing window in SESSIONS when its name ends in
#: one of these AND its default is an ``int``. Deliberately narrow: event
#: detectors spell their windows ``window_days`` / ``baseline_days`` and
#: measure calendar days over event tables, not bars — but those detectors
#: are excluded anyway by the "first parameter is ``bars``" rule below.
_WINDOW_PARAM_SUFFIXES: tuple[str, ...] = ("window", "lookback", "lag")

#: Sessions added to each detector's largest declared window to cover the
#: additive overhead it layers on top (see module docstring). The guard
#: test checks this against every detector's real minimum-bars gate.
COMBINATION_ALLOWANCE_SESSIONS = 16

#: Sessions per calendar day. 250/365.25, not 252/365.25: rounding the
#: numerator DOWN makes every conversion conservative.
SESSIONS_PER_CALENDAR_DAY = 250.0 / 365.25

#: The Tier-2 scan window, in calendar days. See the module docstring for
#: the full derivation; the short version is 268 required sessions ->
#: 392 calendar days minimum -> 420 with ~20 sessions of margin.
TIER2_SCAN_LOOKBACK_DAYS = 420


def bar_detectors() -> dict[str, Callable[..., Any]]:
    """Every bar-based detector in the catalog, discovered by signature.

    A detector is bar-based when it is a public function defined in
    :mod:`dynamo.data.detectors` whose first parameter is named ``bars``.
    Discovery is by introspection on purpose: a new detector added to the
    module is picked up automatically, so a longer lookback than the scan
    window fails the guard test loudly instead of silently never firing.
    """
    out: dict[str, Callable[..., Any]] = {}
    for name, obj in vars(detectors).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if getattr(obj, "__module__", None) != detectors.__name__:
            continue
        params = list(inspect.signature(obj).parameters)
        if params and params[0] == "bars":
            out[name] = obj
    return out


def declared_windows(fn: Callable[..., Any]) -> dict[str, int]:
    """Trailing-window parameters (in sessions) a detector declares."""
    windows: dict[str, int] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if not name.endswith(_WINDOW_PARAM_SUFFIXES):
            continue
        default = param.default
        if isinstance(default, int) and not isinstance(default, bool):
            windows[name] = default
    return windows


def detector_session_requirements() -> dict[str, int]:
    """Sessions of history each bar-based detector needs, by detector name."""
    out: dict[str, int] = {}
    for name, fn in bar_detectors().items():
        windows = declared_windows(fn)
        largest = max(windows.values(), default=0)
        out[name] = largest + COMBINATION_ALLOWANCE_SESSIONS
    return out


def max_detector_sessions() -> int:
    """The longest lookback any registered bar detector needs, in sessions."""
    return max(detector_session_requirements().values(), default=0)


def sessions_in_calendar_days(days: int) -> int:
    """Sessions a ``days``-long calendar window yields (conservative)."""
    return int(days * SESSIONS_PER_CALENDAR_DAY)


def calendar_days_for_sessions(sessions: int) -> int:
    """Smallest calendar window that yields at least ``sessions`` sessions."""
    days = 0
    while sessions_in_calendar_days(days) < sessions:
        days += 1
    return days


def scan_window_sessions(days: int = TIER2_SCAN_LOOKBACK_DAYS) -> int:
    """Sessions the Tier-2 scan window is expected to yield."""
    return sessions_in_calendar_days(days)
