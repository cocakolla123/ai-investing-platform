"""Data-integrity repair CLI (D1/D3/D4).

Every adapter and repository is a fake — these tests never touch the network
or a database.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from dynamo.cli import repair
from dynamo.domain.models.market import CorporateAction, Tier, UniverseMembership


class FakeSettings:
    alpaca_api_key = "alpaca-key"
    alpaca_api_secret = "alpaca-secret"


def action(symbol: str, ex_date: date, kind: str = "split", ratio: float = 0.1) -> CorporateAction:
    return CorporateAction(
        symbol=symbol,
        ex_date=ex_date,
        action_type=kind,
        ratio=ratio if kind == "split" else None,
        amount=None,
        source="alpaca",
    )


class FakeCorporateActionAdapter:
    """Returns a fixed announcement list per requested window, and records
    the windows it was asked for."""

    def __init__(self, per_window: list[list[CorporateAction]]) -> None:
        self._per_window = list(per_window)
        self.windows: list[tuple[date, date]] = []

    def corporate_actions(self, start: date, end: date) -> Iterator[CorporateAction]:
        self.windows.append((start, end))
        index = len(self.windows) - 1
        if index < len(self._per_window):
            yield from self._per_window[index]


class FakeCorporateActionRepo:
    """Real natural-key semantics: keyed on (symbol, ex_date, type, source)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, date, str, str], CorporateAction] = {}
        self.upsert_calls = 0

    def upsert_corporate_actions(self, actions: Iterable[CorporateAction]) -> int:
        submitted = list(actions)
        self.upsert_calls += 1
        for a in submitted:
            self.rows[(a.symbol, a.ex_date, a.action_type, a.source)] = a
        return len(submitted)

    def count_corporate_actions(self) -> int:
        return len(self.rows)

    def corporate_actions_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, kind, _ in self.rows:
            counts[kind] = counts.get(kind, 0) + 1
        return counts


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repair, "_settings", lambda: FakeSettings())


# -- D4: corporate actions --------------------------------------------------


def test_corporate_actions_loads_and_is_idempotent(
    patched: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table held 0 rows, so corp_action_sanity had nothing to check
    against. Loading it must be re-runnable: the second pass writes the same
    natural keys and the row count does not move."""
    repo = FakeCorporateActionRepo()
    batch = [
        action("WKHS", date(2025, 3, 17)),
        action("AAPL", date(2025, 5, 9), kind="dividend"),
    ]
    monkeypatch.setattr(repair, "_repo", lambda url: repo)

    for _ in range(2):
        monkeypatch.setattr(repair, "_alpaca_market", lambda s: FakeCorporateActionAdapter([batch]))
        argv = ["corporate-actions", "--start", "2025-01-01", "--end", "2025-12-31"]
        assert repair.main(argv) == 0

    assert repo.count_corporate_actions() == 2
    assert repo.corporate_actions_by_type() == {"split": 1, "dividend": 1}
    out = capsys.readouterr().out
    assert "0 rows before, 2 after" in out
    assert "2 rows before, 2 after" in out  # the re-run is a no-op


def test_corporate_actions_walks_the_range_in_yearly_slabs(
    patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each slab is committed before the next is fetched, so a kill costs at
    most one year."""
    repo = FakeCorporateActionRepo()
    adapter = FakeCorporateActionAdapter(
        [[action("A", date(2024, 6, 1))], [action("B", date(2025, 6, 1))]]
    )
    monkeypatch.setattr(repair, "_repo", lambda url: repo)
    monkeypatch.setattr(repair, "_alpaca_market", lambda s: adapter)

    assert repair.main(["corporate-actions", "--start", "2024-03-01", "--end", "2025-07-01"]) == 0
    assert adapter.windows == [
        (date(2024, 3, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 7, 1)),
    ]
    assert repo.upsert_calls == 2
    assert repo.count_corporate_actions() == 2


def test_corporate_actions_reports_an_empty_table_as_a_failure(
    patched: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Silently "succeeding" with 0 rows is what left the check uncheckable
    in the first place."""
    monkeypatch.setattr(repair, "_repo", lambda url: FakeCorporateActionRepo())
    monkeypatch.setattr(repair, "_alpaca_market", lambda s: FakeCorporateActionAdapter([[]]))
    assert repair.main(["corporate-actions", "--start", "2025-01-01", "--end", "2025-06-01"]) == 1
    assert "still empty" in capsys.readouterr().out


def test_corporate_actions_refuses_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class NoKeys:
        alpaca_api_key = ""
        alpaca_api_secret = ""

    monkeypatch.setattr(repair, "_settings", lambda: NoKeys())
    assert repair.main(["corporate-actions"]) == 1
    assert "no network calls made" in capsys.readouterr().out


def test_corporate_actions_rejects_an_inverted_range(
    patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repair, "_repo", lambda url: FakeCorporateActionRepo())
    assert repair.main(["corporate-actions", "--start", "2026-01-01", "--end", "2025-01-01"]) == 1


# -- D3: backdate-universe --------------------------------------------------


class FakeUniverseRepo:
    def __init__(self, members: list[UniverseMembership], first_bar: dict[str, date]) -> None:
        self.members = list(members)
        self._first_bar = dict(first_bar)
        self.rebase_calls: list[list[tuple[str, date, date]]] = []

    def universe_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]:
        return [
            m
            for m in self.members
            if m.valid_from <= as_of
            and (m.valid_to is None or m.valid_to >= as_of)
            and (tier is None or int(m.tier) == int(tier))
        ]

    def first_bar_dates(self, *, source: str | None = None) -> dict[str, date]:
        return dict(self._first_bar)

    def rebase_universe_valid_from(self, rebases: list[tuple[str, date, date]]) -> tuple[int, int]:
        self.rebase_calls.append(list(rebases))
        moved = {(s, old): new for s, old, new in rebases}
        for i, m in enumerate(self.members):
            new = moved.get((m.symbol, m.valid_from))
            if new is not None:
                self.members[i] = UniverseMembership(
                    symbol=m.symbol,
                    tier=m.tier,
                    valid_from=new,
                    valid_to=m.valid_to,
                    reason=m.reason,
                    approved_by=m.approved_by,
                )
        return (len(rebases), len(rebases))


BUILD_DAY = date(2026, 8, 17)


def wide(symbol: str) -> UniverseMembership:
    return UniverseMembership(
        symbol=symbol,
        tier=Tier.SURVEILLANCE,
        valid_from=BUILD_DAY,
        valid_to=None,
        reason="alpaca tradable universe (NASDAQ)",
        approved_by="universe-wide",
    )


CORE = UniverseMembership(
    symbol="AAPL",
    tier=Tier.CORE,
    valid_from=date(2024, 12, 1),
    valid_to=None,
    reason="starter",
    approved_by="seed",
)


def test_backdate_universe_rebases_then_no_ops(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = FakeUniverseRepo(
        [CORE, wide("OLD"), wide("IPO")],
        {"AAPL": date(2024, 1, 2), "OLD": date(2024, 1, 2), "IPO": date(2025, 6, 16)},
    )
    monkeypatch.setattr(repair, "_repo", lambda url: repo)

    assert repair.main(["backdate-universe", "--tier", "2"]) == 0
    assert sorted(s for s, _, _ in repo.rebase_calls[0]) == ["IPO", "OLD"]
    assert {m.symbol: m.valid_from for m in repo.members if int(m.tier) == 2} == {
        "OLD": date(2024, 1, 2),
        "IPO": date(2025, 6, 16),
    }
    # The CORE row is not in the plan and its date does not move.
    assert [m.valid_from for m in repo.members if int(m.tier) == 1] == [date(2024, 12, 1)]

    assert repair.main(["backdate-universe", "--tier", "2"]) == 0
    assert len(repo.rebase_calls) == 1  # second run wrote nothing
    assert "idempotent no-op" in capsys.readouterr().out


def test_backdate_universe_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = FakeUniverseRepo([wide("OLD")], {"OLD": date(2024, 1, 2)})
    monkeypatch.setattr(repair, "_repo", lambda url: repo)
    assert repair.main(["backdate-universe", "--tier", "2", "--dry-run"]) == 0
    assert repo.rebase_calls == []
    out = capsys.readouterr().out
    assert "no writes performed" in out
    assert "projected valid_from distribution" in out


def test_backdate_universe_reports_visibility_at_the_replay_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The defect in one line: zero names visible at the campaign start."""
    repo = FakeUniverseRepo(
        [wide("OLD"), wide("IPO")],
        {"OLD": date(2024, 1, 2), "IPO": date(2026, 7, 1)},
    )
    monkeypatch.setattr(repair, "_repo", lambda url: repo)
    assert repair.main(["backdate-universe", "--tier", "2", "--report-as-of", "2026-03-03"]) == 0
    assert "1 tier-2 symbols visible at 2026-03-03" in capsys.readouterr().out


# -- D1: census -------------------------------------------------------------


class FakeCensusRepo:
    def __init__(self, frame: pd.DataFrame, splits: set[tuple[str, date]]) -> None:
        self._frame = frame
        self._splits = set(splits)

    def bars_close_frame(self, **kwargs: Any) -> pd.DataFrame:
        return self._frame

    def split_ex_dates(self) -> set[tuple[str, date]]:
        return set(self._splits)


def series(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": pd.date_range("2026-01-05", periods=len(closes), freq="D", tz=UTC),
            "close": closes,
        }
    )


SPLIT_DAY = datetime(2026, 1, 7, tzinfo=UTC).date()


def test_census_passes_on_an_adjusted_panel(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        repair, "_repo", lambda url: FakeCensusRepo(series("ACME", [50.0] * 5), set())
    )
    assert repair.main(["census"]) == 0
    assert "reads as adjusted" in capsys.readouterr().out


def test_census_cross_check_separates_splits_from_price_moves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ratio that LOOKS like a split is not a split. SPLIT halves on a real
    split ex-date; MOVER halves on a day with no corporate action at all —
    an ordinary -50% crash, which must not be blamed on adjustment."""
    frame = pd.concat(
        [series("SPLIT", [100.0, 100.0, 50.0, 50.0]), series("MOVER", [80.0, 80.0, 40.0, 40.0])]
    )
    monkeypatch.setattr(repair, "_repo", lambda url: FakeCensusRepo(frame, {("SPLIT", SPLIT_DAY)}))
    assert repair.main(["census", "--corporate-actions"]) == 1
    out = capsys.readouterr().out
    assert "1 of 2 candidates fall on a real split ex-date" in out
    assert "UNADJUSTED SPLIT" in out
    assert "UNADJUSTED MOVER" not in out


def test_census_cross_check_is_clean_when_no_candidate_is_a_split(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        repair,
        "_repo",
        lambda url: FakeCensusRepo(
            series("MOVER", [80.0, 80.0, 40.0, 40.0]),
            # A witness exists, it just does not name MOVER on that day.
            {("SOMEONEELSE", SPLIT_DAY)},
        ),
    )
    assert repair.main(["census", "--corporate-actions"]) == 0
    assert "0 of 1 candidates fall on a real split ex-date" in capsys.readouterr().out


def test_census_says_so_when_the_cross_check_has_no_witness(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        repair,
        "_repo",
        lambda url: FakeCensusRepo(series("SPLIT", [100.0, 100.0, 50.0, 50.0]), set()),
    )
    assert repair.main(["census", "--corporate-actions"]) == 0
    assert "cross-check unavailable" in capsys.readouterr().out
