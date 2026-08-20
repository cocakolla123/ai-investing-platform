"""Triage reads the flag TABLE, and writes back to exactly the rows it read.

Two defects, both found by the first whole-market sweep (1,155 flags written,
0 ever triaged):

D3 — `_triage_flags` only ever saw the flags computed in its own process, so
     a row written by any other path was written and forgotten.
D4 — the verdict UPDATE matched `symbol AND triage_verdict IS NULL` with no
     time bound, so one night's verdict overwrote every historical un-triaged
     row for that symbol.

No network, no database, no LLM: the model is a stub that records its calls.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import pytest
from orchestration.assets import research as research_mod

from dynamo.domain.models.market import SurveillanceFlag
from tests.unit.orchestration.conftest import FakeRepo, offline_settings

AS_OF = date(2026, 8, 18)
TODAY = datetime(2026, 8, 18, 21, 0, tzinfo=UTC)
YESTERDAY = datetime(2026, 8, 17, 21, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_instrument_sectors` / `_triage_history` read SQL we do not need here."""

    def fake_read_df(dynamo: Any, stmt: Any) -> pd.DataFrame:
        cols = [c.key for c in stmt.selected_columns]
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    monkeypatch.setattr(research_mod, "_read_df", fake_read_df)


class StubModel:
    """Records every triage call and returns a canned verdict."""

    def __init__(self, verdict: str = "noise") -> None:
        self.verdict = verdict
        self.symbols: list[str] = []

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        context = str(kwargs.get("context", ""))
        self.symbols.append(context.split("for ", 1)[1].split("\n", 1)[0].strip())
        return {"verdict": self.verdict, "reason": "stub"}


class StubDynamo:
    """The `_triage_flags` surface only — no engine, no Dagster."""

    def __init__(self, repo: FakeRepo, model: StubModel | None) -> None:
        self._repo = repo
        self._model = model
        self.log = _Log()

    def market_repo(self) -> Any:
        return self._repo

    def settings(self) -> Any:
        return offline_settings()

    def research_model(self) -> Any:
        return self._model

    def engine(self) -> Any:
        raise AssertionError("triage must not reach for a raw engine")


class _Log:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str, *args: Any) -> None:
        self.messages.append(msg % args if args else msg)

    def warning(self, msg: str, *args: Any) -> None:
        self.messages.append(msg % args if args else msg)


def flag(symbol: str, ts: datetime, *, detector: str = "price_zscore", sev: float = 0.8) -> Any:
    return SurveillanceFlag(
        symbol=symbol, detector=detector, severity=sev, details={"z": 5.0}, ts=ts
    )


def repo_with(*flags: Any) -> FakeRepo:
    repo = FakeRepo(bars=pd.DataFrame(), members=[])
    repo.insert_surveillance_flags(list(flags))
    return repo


def triage(repo: FakeRepo, model: StubModel | None, **kwargs: Any) -> int:
    dynamo = StubDynamo(repo, model)
    return research_mod._triage_flags(
        cast(Any, dynamo), cast(Any, dynamo), AS_OF, **kwargs
    )


class TestD3QueueIsRead:
    def test_flags_written_by_another_process_are_picked_up_later(self) -> None:
        """The defect: 1,155 rows were written by the surveillance-sweep CLI
        and no later run would ever have looked at them. Triage is handed NO
        flags here — everything it acts on comes from the table."""
        repo = repo_with(flag("AAPL", TODAY), flag("MSFT", TODAY))
        model = StubModel()

        assert triage(repo, model) == 2
        assert sorted(model.symbols) == ["AAPL", "MSFT"]
        assert all(r["triage_verdict"] == "noise" for r in repo.flag_rows)

    def test_a_second_run_does_not_re_triage_what_it_already_judged(self) -> None:
        """Rows leave the queue once stamped — the funnel drains rather than
        re-billing the same flag every night."""
        repo = repo_with(flag("AAPL", TODAY))
        first = StubModel()
        assert triage(repo, first) == 1

        second = StubModel()
        assert triage(repo, second) == 0
        assert second.symbols == []

    def test_a_backlog_and_tonights_batch_are_triaged_together(self) -> None:
        repo = repo_with(flag("OLD", YESTERDAY), flag("NEW", TODAY))
        model = StubModel()
        assert triage(repo, model) == 2
        assert set(model.symbols) == {"OLD", "NEW"}

    def test_the_queue_is_bounded_so_stale_flags_expire_unread(self) -> None:
        """A flag old enough to be stale should not be answered against
        today's context. It expires; it is not silently judged."""
        stale = TODAY - timedelta(days=research_mod.TRIAGE_QUEUE_LOOKBACK_DAYS + 5)
        repo = repo_with(flag("STALE", stale), flag("FRESH", TODAY))
        model = StubModel()

        assert triage(repo, model) == 1
        assert model.symbols == ["FRESH"]
        stale_row = next(r for r in repo.flag_rows if r["symbol"] == "STALE")
        assert stale_row["triage_verdict"] is None  # untouched, not judged

    def test_no_model_leaves_the_queue_intact_for_the_next_run(self) -> None:
        """The nightly asset used to drop its flags on the floor when the key
        was unset. Now they persist and the next run picks them up."""
        repo = repo_with(flag("AAPL", TODAY))
        assert triage(repo, None) == 0
        assert repo.flag_rows[0]["triage_verdict"] is None

        model = StubModel()
        assert triage(repo, model) == 1


class TestD4VerdictIsBounded:
    def test_todays_verdict_does_not_stamp_yesterdays_untriaged_row(self) -> None:
        """The data-corrupting one. `symbol AND triage_verdict IS NULL` with
        no bound meant judging AAPL tonight also stamped every AAPL flag ever
        left un-triaged with tonight's verdict."""
        repo = FakeRepo(bars=pd.DataFrame(), members=[])
        repo.insert_surveillance_flags([flag("AAPL", YESTERDAY)])
        yesterday_id = repo.flag_rows[0]["id"]
        repo.insert_surveillance_flags([flag("AAPL", TODAY)])
        today_id = repo.flag_rows[1]["id"]

        # Only today's flag is in range.
        model = StubModel(verdict="material")
        triage(repo, model, lookback_days=0)

        rows = {r["id"]: r for r in repo.flag_rows}
        assert rows[today_id]["triage_verdict"] == "material"
        assert rows[yesterday_id]["triage_verdict"] is None
        assert rows[yesterday_id]["triaged_at"] is None

    def test_the_write_targets_ids_not_a_symbol_predicate(self) -> None:
        """Same symbol, two flags in range, one already judged: the earlier
        verdict must survive rather than be overwritten by the later one."""
        repo = FakeRepo(bars=pd.DataFrame(), members=[])
        repo.insert_surveillance_flags([flag("AAPL", YESTERDAY)])
        repo.flag_rows[0]["triage_verdict"] = "watch"
        repo.insert_surveillance_flags([flag("AAPL", TODAY)])

        triage(repo, StubModel(verdict="noise"))

        assert repo.flag_rows[0]["triage_verdict"] == "watch"
        assert repo.flag_rows[1]["triage_verdict"] == "noise"

    def test_set_triage_verdicts_with_no_ids_is_a_no_op(self) -> None:
        repo = repo_with(flag("AAPL", TODAY))
        assert repo.set_triage_verdicts([], "noise") == 0
        assert repo.flag_rows[0]["triage_verdict"] is None


class TestDataQualityFlagsStayOutOfTheFunnel:
    def test_data_quality_flags_never_reach_the_model(self) -> None:
        """A data_quality flag marks a bar the detectors could not trust (an
        unadjusted split). Sending it to triage would spend an API call
        asking whether a broken price is a trading opportunity."""
        repo = repo_with(
            flag("RCON", TODAY, detector="data_quality"),
            flag("AAPL", TODAY),
        )
        model = StubModel()
        triage(repo, model)
        assert model.symbols == ["AAPL"]

    def test_data_quality_flags_are_retired_not_left_to_clog_the_queue(self) -> None:
        repo = repo_with(flag("RCON", TODAY, detector="data_quality"))
        triage(repo, StubModel())
        row = repo.flag_rows[0]
        assert row["triage_verdict"] == research_mod.DATA_QUALITY_VERDICT
        assert row["triaged_at"] is not None

    def test_retiring_them_costs_no_llm_call(self) -> None:
        repo = repo_with(flag("RCON", TODAY, detector="data_quality"))
        model = StubModel()
        assert triage(repo, model) == 0
        assert model.symbols == []
