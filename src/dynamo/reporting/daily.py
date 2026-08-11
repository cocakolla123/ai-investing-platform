"""Daily activity report: pure builder + plain-language markdown renderer.

Everything the platform did in a day is already in the database; this module
presents it for a non-quant reader. ``build_daily_report`` pulls rows through
a small ``ReportSource`` Protocol (SQL implementation in
``dynamo.reporting.sql_source``; tests use in-memory fakes) and returns a
``DailyReport`` pydantic model; ``render_markdown`` turns that into tidy,
deterministic markdown — same inputs, identical text, no raw JSON dumps.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from dynamo.llm.pricing import (
    BATCH_DISCOUNT,
    CACHE_READ_MULTIPLIER,
    MTOK,
    UnknownModelError,
    pricing_for,
)

Row = dict[str, Any]

# Actions that mean "the committee did something" (vs holds / no-action).
_ACTED = ("buy", "add", "trim", "sell", "avoid")
_BUY_SIDE = ("buy", "add")
_SELL_SIDE = ("trim", "sell", "avoid")


class ReportSource(Protocol):
    """The data slice a daily report needs, as plain lists of dicts.

    Implemented by ``SqlReportSource`` (sibling module); tests supply fakes.
    Every method is scoped to a single report day unless noted.
    """

    # --- at a glance ---
    def equity_points(self, day: date) -> list[Row]: ...  # last 2 rows <= day
    def open_position_counts(self) -> dict[str, int]: ...  # equities/options/kalshi
    def regime_on(self, day: date) -> Row | None: ...
    def ladder_on(self, day: date) -> Row | None: ...

    # --- decisions & watchlist ---
    def verdicts_on(self, day: date) -> list[Row]: ...
    def watchlist_on(self, day: date) -> list[Row]: ...
    def watchlist_prior(self, day: date) -> list[Row]: ...

    # --- trades ---
    def equity_trades_on(self, day: date) -> list[Row]: ...
    def options_trades_on(self, day: date) -> list[Row]: ...
    def pm_trades_on(self, day: date) -> list[Row]: ...

    # --- Claude usage ---
    def cost_entries_on(self, day: date) -> list[Row]: ...
    def cost_month_to_date(self, day: date) -> float: ...
    def monthly_budget(self) -> float: ...

    # --- surveillance & ops ---
    def flags_on(self, day: date) -> list[Row]: ...
    def universe_changes_on(self, day: date) -> list[Row]: ...
    def pending_universe_recommendations(self) -> list[str]: ...
    def unacked_alerts(self) -> list[Row]: ...
    def open_quarantines(self) -> list[Row]: ...
    def batch_failures_on(self, day: date) -> list[Row]: ...
    def reconciliation_on(self, day: date) -> list[Row]: ...

    # --- learning ---
    def reviews_on(self, day: date) -> list[Row]: ...
    def calibration_on(self, day: date) -> list[Row]: ...
    def proposals_on(self, day: date) -> list[Row]: ...
    # Advisory meta-model scores for the day (plan Phase 3): rows with
    # symbol, conviction (raw), effective_probability, model_version.
    def meta_scores_on(self, day: date) -> list[Row]: ...
    # Headline learning.* metrics (plan Phase 6): rows with metric, value —
    # the latest weekly value at or before the report day per metric.
    def learning_metrics_on(self, day: date) -> list[Row]: ...
    # Open improvement proposals awaiting human review (clustering + drift).
    def open_proposal_count(self) -> int: ...


# --- Report model -----------------------------------------------------------


class Glance(BaseModel):
    equity: float | None = None
    day_change_usd: float | None = None
    day_change_pct: float | None = None
    cash_pct: float | None = None
    open_equities: int = 0
    open_options: int = 0
    open_kalshi: int = 0
    open_polymarket_sim: int = 0  # SIMULATED sleeve (ADR-0011) — no real orders
    regime: str | None = None
    ladder_rung: str | None = None
    drawdown_pct: float | None = None
    claude_today_usd: float = 0.0
    claude_mtd_usd: float = 0.0
    claude_budget_usd: float = 0.0
    names_reviewed: int = 0
    names_acted_on: int = 0


class DecisionLine(BaseModel):
    symbol: str
    action: str
    conviction: float
    disagreement: float
    thesis: str = ""
    top_risk: str = ""
    expected_upside_pct: float = 0.0
    expected_downside_pct: float = 0.0
    badges: tuple[str, ...] = ()


class Decisions(BaseModel):
    buys: list[DecisionLine] = []
    sells: list[DecisionLine] = []
    holds: int = 0


class TradeLine(BaseModel):
    filled_at: datetime
    market: str  # stocks | options | kalshi
    instrument: str
    side: str
    quantity: float
    quantity_unit: str  # shares | contracts
    fill_price: float
    price_unit: str  # $ | cents
    slippage_bps: float | None = None
    sleeve: str | None = None


class CostTierRow(BaseModel):
    model_id: str
    label: str
    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_read_share_pct: float  # % of the prompt served from cache
    cost_usd: float


class CostKindRow(BaseModel):
    call_kind: str
    calls: int
    cost_usd: float


class CostSummary(BaseModel):
    tiers: list[CostTierRow] = []
    kinds: list[CostKindRow] = []
    total_today_usd: float = 0.0
    mtd_usd: float = 0.0
    budget_usd: float = 0.0
    cache_savings_usd: float = 0.0
    batch_savings_usd: float = 0.0


class MoverLine(BaseModel):
    sleeve: str
    symbol: str
    rank: int
    prior_rank: int | None = None
    blend_score: float


class DetectorSummary(BaseModel):
    detector: str
    count: int
    symbols: tuple[str, ...] = ()  # notable (up to 3)


class Surveillance(BaseModel):
    detectors: list[DetectorSummary] = []
    triage_outcomes: dict[str, int] = {}
    universe_changes: list[Row] = []
    pending_recommendations: list[str] = []


class AlertsIssues(BaseModel):
    unacked_alerts: list[Row] = []
    quarantines: list[Row] = []
    batch_failures: list[Row] = []
    reconciliation: list[Row] = []


class Learning(BaseModel):
    reviews: list[Row] = []
    calibration: list[Row] = []
    proposals: list[Row] = []
    # Advisory meta-model scores (plan Phase 3) — reported, never sizing
    # input while meta_sizing_enabled is False.
    meta_scores: list[Row] = []
    # Headline learning.* metrics from the weekly suite (plan Phase 6):
    # {metric name: latest value at or before the report day}.
    metrics: dict[str, float] = {}
    # Improvement proposals currently OPEN (awaiting human review).
    open_proposals: int = 0


class DailyReport(BaseModel):
    day: date
    glance: Glance
    decisions: Decisions
    trades: list[TradeLine]
    cost: CostSummary
    movers: list[MoverLine]
    surveillance: Surveillance
    alerts: AlertsIssues
    learning: Learning


# --- Builder ----------------------------------------------------------------


def _f(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _glance(source: ReportSource, day: date, verdicts: list[Row], cost: CostSummary) -> Glance:
    points = sorted(source.equity_points(day), key=lambda p: p["as_of"])
    equity = day_change_usd = day_change_pct = cash_pct = None
    if points:
        latest = points[-1]
        equity = _f(latest["equity"])
        if equity > 0:
            cash_pct = 100.0 * _f(latest["cash"]) / equity
        if len(points) >= 2:
            prior = _f(points[-2]["equity"])
            day_change_usd = equity - prior
            if prior > 0:
                day_change_pct = 100.0 * day_change_usd / prior
    counts = source.open_position_counts()
    regime = source.regime_on(day)
    ladder = source.ladder_on(day)
    return Glance(
        equity=equity,
        day_change_usd=day_change_usd,
        day_change_pct=day_change_pct,
        cash_pct=cash_pct,
        open_equities=int(counts.get("equities", 0)),
        open_options=int(counts.get("options", 0)),
        open_kalshi=int(counts.get("kalshi", 0)),
        open_polymarket_sim=int(counts.get("polymarket_sim", 0)),
        regime=str(regime["regime"]) if regime else None,
        ladder_rung=str(ladder["rung"]) if ladder else None,
        drawdown_pct=100.0 * _f(ladder["drawdown"]) if ladder else None,
        claude_today_usd=cost.total_today_usd,
        claude_mtd_usd=cost.mtd_usd,
        claude_budget_usd=cost.budget_usd,
        names_reviewed=len(verdicts),
        names_acted_on=sum(1 for v in verdicts if str(v["action"]) in _ACTED),
    )


def _decision_line(verdict: Row, badges_by_symbol: dict[str, tuple[str, ...]]) -> DecisionLine:
    payload = verdict.get("payload") or {}
    risks = payload.get("risks") or ()
    return DecisionLine(
        symbol=str(verdict["symbol"]),
        action=str(verdict["action"]),
        conviction=_f(verdict["conviction"]),
        disagreement=_f(verdict.get("disagreement_score")),
        thesis=str(payload.get("thesis") or ""),
        top_risk=str(risks[0]) if risks else "",
        expected_upside_pct=_f(payload.get("expected_upside_pct")),
        expected_downside_pct=_f(payload.get("expected_downside_pct")),
        badges=badges_by_symbol.get(str(verdict["symbol"]), ()),
    )


def _decisions(verdicts: list[Row], watchlist: list[Row]) -> Decisions:
    badges = {
        str(w["symbol"]): tuple(w.get("badges") or ()) for w in watchlist if w.get("badges")
    }
    lines = sorted(
        (_decision_line(v, badges) for v in verdicts),
        key=lambda line: (-line.conviction, line.symbol),
    )
    return Decisions(
        buys=[ln for ln in lines if ln.action in _BUY_SIDE],
        sells=[ln for ln in lines if ln.action in _SELL_SIDE],
        holds=sum(1 for ln in lines if ln.action not in _ACTED),
    )


def _trades(source: ReportSource, day: date) -> list[TradeLine]:
    lines: list[TradeLine] = []
    for t in source.equity_trades_on(day):
        lines.append(
            TradeLine(
                filled_at=t["filled_at"],
                market="stocks",
                instrument=str(t["symbol"]),
                side=str(t["side"]),
                quantity=_f(t["qty"]),
                quantity_unit="shares",
                fill_price=_f(t["fill_price"]),
                price_unit="$",
                slippage_bps=None if t.get("slippage_bps") is None else _f(t["slippage_bps"]),
                sleeve=t.get("sleeve"),
            )
        )
    for t in source.options_trades_on(day):
        lines.append(
            TradeLine(
                filled_at=t["filled_at"],
                market="options",
                instrument=str(t.get("occ_symbol") or t.get("underlying") or ""),
                side=str(t["side"]),
                quantity=_f(t["contracts"]),
                quantity_unit="contracts",
                fill_price=_f(t["price"]),
                price_unit="$",
                sleeve=t.get("sleeve", "options"),
            )
        )
    for t in source.pm_trades_on(day):
        venue = str(t.get("venue") or "kalshi")
        simulated = venue != "kalshi"  # ADR-0011: polymarket rows are SIMULATED
        lines.append(
            TradeLine(
                filled_at=t["filled_at"],
                market="polymarket-sim" if simulated else "kalshi",
                instrument=str(t["market_id"]),
                side=str(t["side"]),
                quantity=_f(t["contracts"]),
                quantity_unit="contracts",
                fill_price=_f(t["price_cents"]),
                price_unit="cents",
                sleeve="polymarket (SIMULATED)" if simulated else t.get("sleeve", "kalshi"),
            )
        )
    return sorted(lines, key=lambda ln: (ln.filled_at, ln.market, ln.instrument, ln.side))


_TIER_LABELS = (("haiku", "Haiku (fast screening)"), ("sonnet", "Sonnet (analysts)"),
                ("opus", "Opus (final arbiter)"))


def _tier_label(model_id: str) -> str:
    for needle, label in _TIER_LABELS:
        if needle in model_id:
            return label
    return model_id


def cache_savings_usd(entries: list[Row]) -> float:
    """Dollars saved by prompt caching: cache reads bill at 0.1x the input
    rate, so every cache-read token saved 0.9x (halved again for batches)."""
    saved = 0.0
    for e in entries:
        tokens = int(e.get("cache_read_tokens") or 0)
        if not tokens:
            continue
        try:
            rate = pricing_for(str(e["model_id"])).input_per_mtok
        except UnknownModelError:
            continue
        full = tokens * rate * (1.0 - CACHE_READ_MULTIPLIER) / MTOK
        saved += full * (BATCH_DISCOUNT if e.get("batch") else 1.0)
    return saved


def _cost(source: ReportSource, day: date) -> CostSummary:
    entries = source.cost_entries_on(day)
    tiers: dict[str, CostTierRow] = {}
    kinds: dict[str, CostKindRow] = {}
    for e in entries:
        model_id = str(e["model_id"])
        tier = tiers.get(model_id)
        if tier is None:
            tier = tiers[model_id] = CostTierRow(
                model_id=model_id,
                label=_tier_label(model_id),
                calls=0,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_read_share_pct=0.0,
                cost_usd=0.0,
            )
        tier.calls += 1
        tier.input_tokens += int(e.get("input_tokens") or 0)
        tier.output_tokens += int(e.get("output_tokens") or 0)
        tier.cache_read_tokens += int(e.get("cache_read_tokens") or 0)
        tier.cost_usd += _f(e.get("cost_usd"))
        kind_key = str(e.get("call_kind") or "other")
        kind = kinds.get(kind_key)
        if kind is None:
            kind = kinds[kind_key] = CostKindRow(call_kind=kind_key, calls=0, cost_usd=0.0)
        kind.calls += 1
        kind.cost_usd += _f(e.get("cost_usd"))
    for tier in tiers.values():
        prompt_tokens = tier.input_tokens + tier.cache_read_tokens
        if prompt_tokens:
            tier.cache_read_share_pct = 100.0 * tier.cache_read_tokens / prompt_tokens
    return CostSummary(
        tiers=sorted(tiers.values(), key=lambda t: (-t.cost_usd, t.model_id)),
        kinds=sorted(kinds.values(), key=lambda k: (-k.cost_usd, k.call_kind)),
        total_today_usd=sum(_f(e.get("cost_usd")) for e in entries),
        mtd_usd=source.cost_month_to_date(day),
        budget_usd=source.monthly_budget(),
        cache_savings_usd=cache_savings_usd(entries),
        # Batches bill at 50%: the discount saved exactly what was charged.
        batch_savings_usd=sum(_f(e.get("cost_usd")) for e in entries if e.get("batch")),
    )


def _movers(source: ReportSource, day: date, top_n: int = 5) -> list[MoverLine]:
    current = source.watchlist_on(day)
    prior_rank = {
        (str(w["sleeve"]), str(w["symbol"])): int(w["rank"]) for w in source.watchlist_prior(day)
    }
    movers = [
        MoverLine(
            sleeve=str(w["sleeve"]),
            symbol=str(w["symbol"]),
            rank=int(w["rank"]),
            prior_rank=prior_rank.get((str(w["sleeve"]), str(w["symbol"]))),
            blend_score=_f(w["blend_score"]),
        )
        for w in current
    ]
    movers.sort(key=lambda m: (m.sleeve, m.rank, m.symbol))
    out: list[MoverLine] = []
    per_sleeve: dict[str, int] = {}
    for m in movers:
        if per_sleeve.get(m.sleeve, 0) < top_n:
            out.append(m)
            per_sleeve[m.sleeve] = per_sleeve.get(m.sleeve, 0) + 1
    return out


def _surveillance(source: ReportSource, day: date) -> Surveillance:
    detectors: dict[str, list[str]] = {}
    triage: dict[str, int] = {}
    for flag in source.flags_on(day):
        detectors.setdefault(str(flag["detector"]), []).append(str(flag["symbol"]))
        verdict = flag.get("triage_verdict")
        if verdict:
            triage[str(verdict)] = triage.get(str(verdict), 0) + 1
    return Surveillance(
        detectors=[
            DetectorSummary(detector=name, count=len(syms), symbols=tuple(sorted(set(syms))[:3]))
            for name, syms in sorted(detectors.items())
        ],
        triage_outcomes=dict(sorted(triage.items())),
        universe_changes=source.universe_changes_on(day),
        pending_recommendations=source.pending_universe_recommendations(),
    )


def build_daily_report(source: ReportSource, day: date) -> DailyReport:
    """Assemble the full report for ``day`` from a ``ReportSource``."""
    verdicts = sorted(source.verdicts_on(day), key=lambda v: str(v["symbol"]))
    watchlist = source.watchlist_on(day)
    cost = _cost(source, day)
    return DailyReport(
        day=day,
        glance=_glance(source, day, verdicts, cost),
        decisions=_decisions(verdicts, watchlist),
        trades=_trades(source, day),
        cost=cost,
        movers=_movers(source, day),
        surveillance=_surveillance(source, day),
        alerts=AlertsIssues(
            unacked_alerts=source.unacked_alerts(),
            quarantines=source.open_quarantines(),
            batch_failures=source.batch_failures_on(day),
            reconciliation=source.reconciliation_on(day),
        ),
        learning=Learning(
            reviews=source.reviews_on(day),
            calibration=source.calibration_on(day),
            proposals=source.proposals_on(day),
            meta_scores=source.meta_scores_on(day),
            metrics={
                str(m["metric"]): _f(m.get("value"))
                for m in source.learning_metrics_on(day)
                if m.get("value") is not None
            },
            open_proposals=int(source.open_proposal_count()),
        ),
    )


# --- Markdown renderer ------------------------------------------------------


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _progress(fraction: float, width: int = 10) -> str:
    filled = min(width, max(0, round(fraction * width)))
    return "`[" + "#" * filled + "-" * (width - filled) + "]`"


def _none_today(lines: list[str]) -> None:
    lines.extend(["None today.", ""])


def _render_glance(g: Glance, lines: list[str]) -> None:
    lines.append("## At a glance")
    if g.equity is None:
        lines.append("- **Portfolio value:** no equity snapshot recorded yet.")
    else:
        change = ""
        if g.day_change_usd is not None and g.day_change_pct is not None:
            direction = "up" if g.day_change_usd >= 0 else "down"
            change = (
                f" ({direction} {_usd(abs(g.day_change_usd))}, "
                f"{g.day_change_pct:+.2f}% vs the prior day)"
            )
        lines.append(f"- **Portfolio value:** {_usd(g.equity)}{change}")
    if g.cash_pct is not None:
        lines.append(f"- **Cash on hand:** {g.cash_pct:.1f}% of the portfolio")
    positions_line = (
        f"- **Open positions:** {g.open_equities} stocks, {g.open_options} options, "
        f"{g.open_kalshi} Kalshi event markets"
    )
    if g.open_polymarket_sim:
        positions_line += (
            f", {g.open_polymarket_sim} Polymarket shadow "
            "(SIMULATED — tracked internally, no real orders sent)"
        )
    lines.append(positions_line)
    if g.regime is not None:
        lines.append(f"- **Market regime (overall market conditions):** {g.regime}")
    if g.ladder_rung is not None:
        dd = f", drawdown {abs(g.drawdown_pct or 0.0):.1f}% below the peak"
        lines.append(
            f"- **Drawdown ladder (risk throttle after losses):** rung '{g.ladder_rung}'{dd}"
        )
    if g.claude_budget_usd > 0:
        frac = g.claude_mtd_usd / g.claude_budget_usd
        lines.append(
            f"- **Claude spend:** {_usd(g.claude_today_usd)} today; "
            f"{_usd(g.claude_mtd_usd)} of {_usd(g.claude_budget_usd)} used this month "
            f"({frac:.0%}) {_progress(frac)}"
        )
    else:
        lines.append(f"- **Claude spend:** {_usd(g.claude_today_usd)} today")
    lines.append(
        f"- **Committee activity:** reviewed {g.names_reviewed} names, acted on "
        f"{g.names_acted_on} — deliberately sitting tight on the rest."
    )
    lines.append("")


def _render_decision(line: DecisionLine, out: list[str]) -> None:
    badges = f" · badges: {', '.join(line.badges)}" if line.badges else ""
    out.append(
        f"**{line.symbol} — {line.action}** (conviction {line.conviction:.2f} of 1, "
        f"disagreement {line.disagreement:.2f} of 1){badges}"
    )
    if line.expected_upside_pct or line.expected_downside_pct:
        out.append(
            f"- Expected move: {line.expected_upside_pct:+.1f}% upside vs "
            f"{-abs(line.expected_downside_pct):.1f}% downside"
        )
    why = line.thesis.rstrip(". ")
    if why:
        risk = f" The main risk: {line.top_risk.rstrip('. ')}." if line.top_risk else ""
        out.append(f"- Why: {why}.{risk}")
    out.append("")


def _render_decisions(d: Decisions, lines: list[str]) -> None:
    lines.append("## Decisions")
    if not d.buys and not d.sells and not d.holds:
        _none_today(lines)
        return
    if d.buys:
        lines.extend(["### Buys & adds", ""])
        for line in d.buys:
            _render_decision(line, lines)
    if d.sells:
        lines.extend(["### Trims, sells & avoids", ""])
        for line in d.sells:
            _render_decision(line, lines)
    lines.extend([f"Holds / no action: {d.holds} names.", ""])


def _render_trades(trades: list[TradeLine], lines: list[str]) -> None:
    lines.append("## Trades executed")
    if not trades:
        _none_today(lines)
        return
    lines.append(
        "Slippage compares the fill to the price when the decision was made "
        "(in basis points; 100 bps = 1%). Prediction-market prices are in cents "
        "per contract."
    )
    if any(t.market == "polymarket-sim" for t in trades):
        lines.append(
            "Polymarket rows are SIMULATED — tracked internally, no real orders sent."
        )
    lines.append("")
    rows = [
        [
            t.filled_at.strftime("%H:%M"),
            t.market,
            t.instrument,
            t.side,
            f"{t.quantity:g} {t.quantity_unit}",
            f"${t.fill_price:,.2f}" if t.price_unit == "$" else f"{t.fill_price:,.2f} cents",
            "—" if t.slippage_bps is None else f"{t.slippage_bps:+.1f} bps",
            t.sleeve or "—",
        ]
        for t in trades
    ]
    lines.extend(
        _table(
            ["Time (UTC)", "Market", "Symbol", "Side", "Quantity", "Fill price", "Slippage",
             "Sleeve"],
            rows,
        )
    )
    lines.append("")


def _render_cost(c: CostSummary, lines: list[str]) -> None:
    lines.append("## Claude usage & cost")
    if not c.tiers:
        _none_today(lines)
        return
    lines.extend(
        _table(
            ["Model tier", "Calls", "Input tokens", "Output tokens", "Cache reads",
             "Prompt from cache", "Cost"],
            [
                [
                    t.label,
                    str(t.calls),
                    f"{t.input_tokens:,}",
                    f"{t.output_tokens:,}",
                    f"{t.cache_read_tokens:,}",
                    f"{t.cache_read_share_pct:.0f}%",
                    _usd(t.cost_usd),
                ]
                for t in c.tiers
            ],
        )
    )
    lines.append("")
    lines.extend(
        _table(
            ["Stage", "Calls", "Cost"],
            [[k.call_kind, str(k.calls), _usd(k.cost_usd)] for k in c.kinds],
        )
    )
    lines.append("")
    lines.append(f"- **Total today:** {_usd(c.total_today_usd)}")
    if c.budget_usd > 0:
        frac = c.mtd_usd / c.budget_usd
        lines.append(
            f"- **Month to date:** {_usd(c.mtd_usd)} of the {_usd(c.budget_usd)} budget "
            f"({frac:.0%}) {_progress(frac)}"
        )
    if c.cache_savings_usd > 0:
        lines.append(
            f"- Reusing cached prompts saved about {_usd(c.cache_savings_usd)} today "
            "(cached text bills at a tenth of the normal rate)."
        )
    if c.batch_savings_usd > 0:
        lines.append(
            f"- Overnight batch processing saved another {_usd(c.batch_savings_usd)} "
            "(batches bill at half price)."
        )
    lines.append("")


def _render_movers(movers: list[MoverLine], lines: list[str]) -> None:
    lines.append("## Watchlist movers")
    if not movers:
        _none_today(lines)
        return
    def _change(m: MoverLine) -> str:
        if m.prior_rank is None:
            return "new"
        delta = m.prior_rank - m.rank
        if delta == 0:
            return "="
        arrow = "up" if delta > 0 else "down"
        return f"{arrow} {abs(delta)}"

    lines.extend(
        _table(
            ["Sleeve", "Rank", "Symbol", "Change vs prior day", "Blend score"],
            [
                [m.sleeve, str(m.rank), m.symbol, _change(m), f"{m.blend_score:.2f}"]
                for m in movers
            ],
        )
    )
    lines.append("")


def _render_surveillance(s: Surveillance, lines: list[str]) -> None:
    lines.append("## Surveillance & promotions")
    if (
        not s.detectors
        and not s.triage_outcomes
        and not s.universe_changes
        and not s.pending_recommendations
    ):
        _none_today(lines)
        return
    if s.detectors:
        lines.extend(
            _table(
                ["Detector", "Flags", "Notable symbols"],
                [[d.detector, str(d.count), ", ".join(d.symbols) or "—"] for d in s.detectors],
            )
        )
        lines.append("")
    if s.triage_outcomes:
        outcomes = ", ".join(f"{k}: {v}" for k, v in s.triage_outcomes.items())
        lines.extend([f"Triage outcomes (what the screening pass decided): {outcomes}.", ""])
    for change in s.universe_changes:
        lines.append(
            f"- {change.get('symbol')}: {change.get('kind')} "
            f"(tier {change.get('tier')}) — {change.get('reason') or 'no reason recorded'}"
        )
    if s.universe_changes:
        lines.append("")
    if s.pending_recommendations:
        lines.append("Universe changes awaiting your approval:")
        lines.extend(f"- {rec}" for rec in s.pending_recommendations)
        lines.append("")


def _render_alerts(a: AlertsIssues, lines: list[str]) -> None:
    lines.append("## Alerts & issues")
    if (
        not a.unacked_alerts
        and not a.quarantines
        and not a.batch_failures
        and not a.reconciliation
    ):
        _none_today(lines)
        return
    if a.unacked_alerts:
        lines.append("Unacknowledged alerts:")
        for alert in a.unacked_alerts:
            lines.append(f"- [{alert.get('severity')}] {alert.get('name')}: {alert.get('message')}")
        lines.append("")
    if a.quarantines:
        lines.append("Data held back in quarantine (excluded until resolved):")
        for q in a.quarantines:
            lines.append(f"- {q.get('scope')} `{q.get('key')}` — {q.get('reason')}")
        lines.append("")
    if a.batch_failures:
        lines.append("Overnight batch jobs that did not finish cleanly:")
        for b in a.batch_failures:
            lines.append(f"- {b.get('stage')} batch `{b.get('batch_id')}` — {b.get('status')}")
        lines.append("")
    for r in a.reconciliation:
        n = len(r.get("divergences") or [])
        status = "clean — our books match the broker" if r.get("result") == "clean" else (
            f"{n} difference(s) found between our books and the broker"
        )
        lines.append(f"- Reconciliation ({r.get('scope')}): {status}.")
    if a.reconciliation:
        lines.append("")


# Headline learning.* metrics (plan Phase 6): (metric, label, format).
_LEARNING_METRIC_LINES: tuple[tuple[str, str, str], ...] = (
    ("learning.brier", "Brier score (confidence vs reality; lower is better)", "{:.3f}"),
    ("learning.success_rate", "Success rate (scored decisions that worked out)", "{:.0%}"),
    ("learning.abstention_rate", "Abstention rate (committee chose to sit out)", "{:.0%}"),
    ("learning.ev_per_decision", "EV per decision (avg realized pnl)", "{:+.2f}%"),
)


def _render_learning(learning: Learning, lines: list[str]) -> None:
    lines.append("## Learning")
    if (
        not learning.reviews
        and not learning.calibration
        and not learning.proposals
        and not learning.meta_scores
        and not learning.metrics
        and not learning.open_proposals
    ):
        _none_today(lines)
        return
    if learning.reviews:
        lines.append("Post-trade reviews completed (what we took away):")
        for r in learning.reviews:
            lesson = r.get("lesson") or "no lesson recorded"
            lines.append(f"- {r.get('symbol')}: {lesson}")
        lines.append("")
    if learning.calibration:
        lines.append(
            "Calibration updates (Brier score: how well confidence matched reality; "
            "lower is better):"
        )
        for c in learning.calibration:
            lines.append(
                f"- {c.get('scope_id')}: Brier {_f(c.get('brier')):.3f} over {c.get('n')} trades"
            )
        lines.append("")
    if learning.proposals:
        lines.append("Improvement proposals opened (awaiting your review):")
        for p in learning.proposals:
            effect = p.get("expected_effect") or "no expected effect stated"
            lines.append(f"- {p.get('kind')} on `{p.get('target')}` — {effect}")
        lines.append("")
    if learning.meta_scores:
        lines.append(
            "Meta-model advisory scores (a second opinion on each decision's "
            "win probability — ADVISORY only, not used for sizing):"
        )
        lines.extend(
            _table(
                ["Symbol", "Raw conviction", "Meta win probability", "Model"],
                [
                    [
                        str(m.get("symbol")),
                        "—" if m.get("conviction") is None else f"{_f(m.get('conviction')):.2f}",
                        f"{_f(m.get('effective_probability')):.2f}",
                        str(m.get("model_version") or ""),
                    ]
                    for m in learning.meta_scores
                ],
            )
        )
        lines.append("")
    rendered_metrics = [
        (label, fmt.format(learning.metrics[name]))
        for name, label, fmt in _LEARNING_METRIC_LINES
        if name in learning.metrics
    ]
    if rendered_metrics:
        n_scored = learning.metrics.get("learning.n_scored")
        suffix = f" (over {n_scored:.0f} scored decisions)" if n_scored is not None else ""
        lines.append(f"Learning health — weekly metrics on resolved decisions{suffix}:")
        lines.extend(f"- {label}: {value}" for label, value in rendered_metrics)
        lines.append("")
    if learning.open_proposals:
        lines.extend(
            [
                f"Improvement proposals open (awaiting your review, nothing "
                f"auto-applies): {learning.open_proposals}",
                "",
            ]
        )


def render_markdown(report: DailyReport) -> str:
    """Deterministic plain-language markdown for one day's activity."""
    lines: list[str] = [
        f"# Dynamo Daily Report — {report.day.isoformat()} ({report.day.strftime('%A')})",
        "",
    ]
    _render_glance(report.glance, lines)
    _render_decisions(report.decisions, lines)
    _render_trades(report.trades, lines)
    _render_cost(report.cost, lines)
    _render_movers(report.movers, lines)
    _render_surveillance(report.surveillance, lines)
    _render_alerts(report.alerts, lines)
    _render_learning(report.learning, lines)
    return "\n".join(lines).rstrip() + "\n"
