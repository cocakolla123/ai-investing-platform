"""The LIVE dossier builder carries the same evidence as the replay one.

``build_deep_dive_contexts`` (live) and ``build_replay_contexts`` (replay)
passed an identical seven-of-eight arguments before this change; the campaign
fix adds fundamentals, peers, extended price stats and the symbol-filtered
news digest to both, so the two paths must not drift. No engine is touched:
every SQL read is monkeypatched to a frame.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import orchestration.assets.research as research_mod
import pandas as pd
import pytest
from orchestration.assets.research import build_deep_dive_contexts

AS_OF = date(2026, 6, 16)
SECTORS = {"ACME": "Industrials", "PEER": "Industrials"}


def _bars(symbols: list[str], sessions: int = 300) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = datetime(2025, 1, 2, tzinfo=UTC)
    for i, sym in enumerate(sorted(symbols)):
        for session in range(sessions):
            close = 100.0 + i * 5 + session * 0.1
            rows.append(
                {
                    "symbol": sym,
                    "ts": start + timedelta(days=session),
                    "close": close,
                    "volume": 1_000.0,
                }
            )
    return pd.DataFrame(rows)


def _facts() -> pd.DataFrame:
    quarters = (
        ("2025-03-31", 100.0, "2025-04-20"),
        ("2025-06-30", 110.0, "2025-07-20"),
        ("2025-09-30", 120.0, "2025-10-20"),
        ("2025-12-31", 460.0, "2026-02-20"),  # fiscal-year row
        ("2026-03-31", 130.0, "2026-04-20"),
    )
    rows = [
        {
            "symbol": symbol,
            "tag": tag,
            "period_end": pd.Timestamp(period),
            "value": value * scale,
            "filed_at": datetime.fromisoformat(filed).replace(tzinfo=UTC),
        }
        for symbol in ("ACME", "PEER")
        for tag, scale in (("Revenues", 1.0), ("NetIncomeLoss", 0.1))
        for period, value, filed in quarters
    ]
    rows += [
        {
            "symbol": symbol,
            "tag": "CommonStockSharesOutstanding",
            "period_end": pd.Timestamp("2026-03-31"),
            "value": 1_000_000.0,
            "filed_at": datetime(2026, 4, 20, tzinfo=UTC),
        }
        for symbol in ("ACME", "PEER")
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A DynamoResource stub plus monkeypatched reads for the live builder."""
    news_rows = [
        {
            "id": "n-1",
            "symbols": ["ACME"],
            "headline": "Acme wins a contract",
            "summary": "",
            "source": "test",
            "published_at": datetime(2026, 6, 15, 12, tzinfo=UTC),
        },
        {
            "id": "n-2",
            "symbols": ["ACME"],
            "headline": "Rival ships a competing widget",
            "summary": "",
            "source": "test",
            "published_at": datetime(2026, 6, 14, 12, tzinfo=UTC),
        },
    ]
    instruments = pd.DataFrame(
        [
            {
                "symbol": "ACME",
                "name": "Acme Corp",
                "sector": "Industrials",
                "is_etf": False,
                "status": "active",
            }
        ]
    )

    def fake_read_df(dynamo: Any, stmt: Any) -> pd.DataFrame:
        cols = [c.key for c in stmt.selected_columns]
        if "headline" in cols:
            return pd.DataFrame(news_rows)
        if "name" in cols and "sector" in cols:
            return instruments
        if cols == ["symbol", "sector"]:
            return pd.DataFrame([{"symbol": s, "sector": sec} for s, sec in SECTORS.items()])
        if "content" in cols:
            return pd.DataFrame(columns=pd.Index(["symbol", "event_date", "source", "content"]))
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    class _Repo:
        def latest_macro_digest(self) -> dict[str, Any]:
            return {"narrative": "Regime stable."}

    class _Dynamo:
        def market_repo(self) -> Any:
            return _Repo()

        def bars_window(self, symbols: Any, as_of: date, days: int) -> pd.DataFrame:
            frame = _bars(list(symbols))
            return cast(pd.DataFrame, frame[frame["symbol"].isin(list(symbols))])

    monkeypatch.setattr(research_mod, "_read_df", fake_read_df)
    monkeypatch.setattr(research_mod, "_dossier_fundamentals", lambda *a, **k: _facts())
    monkeypatch.setattr(research_mod, "_journal_hits_for", lambda *a, **k: {"ACME": []})
    monkeypatch.setattr(research_mod, "_quant_score_for", lambda *a, **k: None)
    return _Dynamo()


def test_live_dossier_carries_fundamentals_peers_and_filtered_news(wired: Any) -> None:
    contexts = build_deep_dive_contexts(wired, ["ACME"], AS_OF)
    context = contexts["ACME"]

    # Fundamentals: the fiscal-year row is differenced, not summed.
    assert "## Fundamentals" in context
    assert "revenue_ttm: 0.0M USD [xbrl:ACME:Revenues:2025-06-30..2026-03-31" in context
    assert "basis: point-in-time XBRL facts filed <= 2026-06-16" in context

    # Peers: sector ETF + benchmark relative strength.
    assert "## Peer / relative strength" in context
    assert "sector: Industrials | sector_etf: XLI | benchmark: SPY" in context
    assert "excess_vs_SPY_60d_pct:" in context

    # Extended price stats alongside the original four.
    for key in ("close:", "ret_20d_pct:", "vol_20d_ann_pct:", "dollar_volume:"):
        assert key in context
    for key in ("high_52w:", "dist_from_52w_high_pct:", "ret_250d_pct:"):
        assert key in context

    # News: the ticker's own item leads, the unrelated one is labelled.
    digest = context.split("## News digest", 1)[1].split("## Market news", 1)[0]
    assert "[n-1]" in digest and "[n-2]" not in digest
    assert "[n-2]" in context.split("## Market news", 1)[1]


def test_live_dossier_is_byte_identical_across_runs(wired: Any) -> None:
    first = build_deep_dive_contexts(wired, ["ACME"], AS_OF)
    second = build_deep_dive_contexts(wired, ["ACME"], AS_OF)
    assert first == second


def test_live_section_signature_is_the_parity_contract(wired: Any) -> None:
    """Section headings are the parity contract between the two call sites —
    ``tests/unit/replay/test_committee.py`` asserts the same list."""
    live = build_deep_dive_contexts(wired, ["ACME"], AS_OF)["ACME"]
    signature = [line for line in live.splitlines() if line.startswith("## ")]
    assert signature == [
        "## Instrument profile",
        "## Fundamentals",
        "## Quant score",
        "## Price/volume stats",
        "## Peer / relative strength",
        "## Filing excerpts",
        "## News digest",
        "## Market news",
        "## Macro digest",
        "## Journal hits",
    ]


# --------------------------------------------------------------------------
# argument-set parity between the two call sites
# --------------------------------------------------------------------------
#
# The section-signature test above compares the RENDERED output, which only
# catches a divergence that changes a heading. It does not catch the class of
# bug that has now bitten twice in one day: one caller passing an argument the
# other does not, where the missing one degrades to its default and renders as
# a plausible "none". So compare the calls themselves.


def _dossier_call_kwargs(func: object) -> set[str]:
    """Keyword names of the ``assemble_dossier_context(...)`` call in ``func``."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(func).lstrip())  # type: ignore[arg-type]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assemble_dossier_context"
        ):
            assert not node.args, "the dossier context takes keyword arguments only"
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no assemble_dossier_context(...) call found in {func!r}")


def test_live_and_replay_pass_the_same_dossier_arguments() -> None:
    """``build_deep_dive_contexts`` (live) and ``build_replay_contexts``
    (replay) must hand the dossier the same argument set.

    A backtest that debates on evidence the nightly run never assembles is
    not a backtest of the nightly run. The divergence is invisible in output
    diffs because every dossier section has a default that renders as "none".
    """
    from dynamo.replay.committee import build_replay_contexts

    live = _dossier_call_kwargs(build_deep_dive_contexts)
    replay = _dossier_call_kwargs(build_replay_contexts)
    assert live == replay, (
        f"dossier argument sets diverged: live-only={sorted(live - replay)}, "
        f"replay-only={sorted(replay - live)}"
    )


def test_the_shared_argument_set_is_the_one_the_dossier_offers() -> None:
    """Pin the set itself, so ADDING a dossier parameter is a deliberate act
    on both sides rather than something one caller quietly picks up."""
    from dynamo.research.dossier import assemble_dossier_context

    live = _dossier_call_kwargs(build_deep_dive_contexts)
    assert live == {
        "instrument",
        "quant_score",
        "price_stats",
        "fundamentals",
        "peers",
        "news_items",
        "market_news_items",
        "macro_digest_text",
        "journal_hits",
        "earnings_call_excerpt",
    }
    # Every name passed must be a real parameter — a typo would silently
    # become an unexpected-keyword TypeError only on a code path that runs.
    accepted = set(assemble_dossier_context.__annotations__) - {"return"}
    assert live <= accepted
    # `filing_excerpts` is offered and passed by NEITHER caller. That is a
    # known, shared gap (nothing extracts filing sections yet), not a
    # divergence — and it stays shared because the test above compares sets.
    assert accepted - live == {"filing_excerpts"}
