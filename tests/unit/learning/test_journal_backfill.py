"""Journal metadata backfill plan: point in time, fill-only, idempotent."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.learning.journal_backfill import (
    EntryMeta,
    SleeveSource,
    Sources,
    applied,
    counts,
    plan,
)

T = datetime(2026, 9, 20, tzinfo=UTC)


def entry(**kw: object) -> EntryMeta:
    base: dict[str, object] = dict(
        id="e1",
        kind="thesis",
        symbols=("KO",),
        created_at=T,
        refs={},
        strategy=None,
        regime_at_write=None,
        sectors=(),
    )
    base.update(kw)
    return EntryMeta(**base)  # type: ignore[arg-type]


SRC = Sources(
    live_regimes=[(date(2026, 9, 19), "risk_on"), (date(2026, 9, 22), "neutral")],
    replay_regimes=[(date(2026, 9, 18), "risk_off")],
    sector_by_symbol={"KO": "Consumer Staples"},
)


def test_live_entry_gets_live_regime_on_or_before_write_date_and_sectors() -> None:
    (f,), _ = plan([entry()], SRC)
    assert (f.regime_at_write, f.sectors) == ("risk_on", ("Consumer Staples",))


def test_replay_entry_uses_the_replay_regime() -> None:
    (f,), _ = plan([entry(refs={"source": "replay"})], SRC)
    assert f.regime_at_write == "risk_off"


def test_sleeve_assigned_after_the_entry_is_not_used() -> None:
    later = SleeveSource("tactical", datetime(2026, 9, 21, tzinfo=UTC))
    earlier = SleeveSource("tactical", datetime(2026, 9, 19, tzinfo=UTC))
    for src, want in ((later, None), (earlier, "tactical")):
        s = Sources(decision_context_by_verdict={"v1": src}, live_regimes=SRC.live_regimes)
        (f,), why = plan([entry(refs={"verdict_id": "v1"})], s)
        assert f.strategy == want
        if want is None:
            assert "thesis.strategy: sleeve assigned after the entry was written" in why


def test_existing_values_are_never_overwritten_and_rerun_is_a_noop() -> None:
    e = entry(strategy="long_term", regime_at_write="neutral", sectors=("X",))
    fills, _ = plan([e], SRC)
    assert fills == []
    once, _ = plan([entry()], SRC)
    after = applied([entry()], once)
    assert plan(after, SRC)[0] == []
    assert counts(after)["thesis"] == {
        "total": 1,
        "strategy": 0,
        "regime_at_write": 1,
        "sectors": 1,
    }


def test_unfillable_fields_are_counted_with_a_reason() -> None:
    _, why = plan([entry(symbols=("ZZZ",), created_at=datetime(2026, 9, 1, tzinfo=UTC))], SRC)
    assert why["thesis.regime_at_write: no regime_states session on or before write date"] == 1
    assert why["thesis.sectors: no instruments.sector for ['ZZZ']"] == 1
    assert why["thesis.strategy: no linked trade record"] == 1
