"""Envelope shapes, error mapping, and /health for the read surface."""

from __future__ import annotations

import pytest


def test_watchlist_asof_envelope(client) -> None:
    body = client.get("/watchlist").json()
    assert set(body) == {"data", "as_of"}
    assert body["as_of"] == "2026-08-03"
    row = body["data"][0]
    assert row["symbol"] == "AAPL"
    assert row["badges"] == ["conflict"]


def test_verdicts_page_envelope_includes_dissents_and_self_critique(client) -> None:
    body = client.get("/verdicts/aapl").json()  # case-insensitive symbol
    assert set(body) == {"items", "next_cursor", "total"}
    assert body["total"] == 1
    verdict = body["items"][0]
    assert verdict["dissents"][0]["agent"] == "bear"  # never trimmed (doc 03 §2)
    assert verdict["self_critique"]


def test_verdicts_unknown_symbol_is_problem_json_404(client) -> None:
    resp = client.get("/verdicts/ZZZZ")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["status"] == 404
    assert "ZZZZ" in body["detail"]


def test_dossiers_and_reports(client) -> None:
    assert client.get("/dossiers/AAPL").json()["total"] == 1
    dates = client.get("/reports").json()
    assert dates["items"][0]["verdict_count"] == 1
    report = client.get("/reports/2026-08-03").json()
    assert set(report["data"]) == {"verdicts", "watchlist", "gaps"}
    assert report["data"]["gaps"][0]["reason"] == "schema_invalid"
    assert client.get("/reports/2020-01-01").status_code == 404


def test_positions_page(client) -> None:
    body = client.get("/portfolio/positions").json()
    assert body["items"][0]["decision_context_id"] == "dc-1"


def test_performance_computes_sharpe_and_drawdown(client) -> None:
    data = client.get("/portfolio/performance").json()["data"]
    assert len(data["series"]) == 4
    assert data["sharpe"] is not None
    # equity dips 101000 -> 99500 = -1.485%
    assert data["max_drawdown"] == pytest.approx(-0.014851, abs=1e-4)


def test_risk_endpoints(client) -> None:
    limits = client.get("/risk/limits").json()["data"]
    assert limits["verdict"]["cash_floor"] == 0.05
    assert limits["config"]["per_position_cap"] == 0.08
    metrics = client.get("/risk/metrics").json()["data"]
    assert metrics["metrics"]["var95_1d"] == 0.012
    assert metrics["regime"]["regime"] == "neutral"
    assert client.get("/risk/ladder").json()["items"][0]["rung"] == "normal"
    stress = client.get("/risk/stress").json()["data"]
    assert stress["results"][0]["scenario"] == "2008"


def test_journal_search(client, deps) -> None:
    body = client.get("/journal/search", params={"q": "AAPL", "symbol": "aapl"}).json()
    assert body["total"] == 1
    assert body["items"][0]["kind"] == "thesis"
    assert deps.journal.calls[0]["symbol"] == "AAPL"


def test_journal_search_rejects_unknown_kind(client) -> None:
    resp = client.get("/journal/search", params={"kind": "poem"})
    assert resp.status_code == 400
    assert "poem" in resp.json()["detail"]


def test_ops_summary_and_queues(client) -> None:
    data = client.get("/ops/summary").json()["data"]
    assert data["claude_spend"]["budget_utilization"] == pytest.approx(0.5)
    assert data["batch_jobs"][0]["batch_id"] == "b-1"
    assert client.get("/alerts").json()["total"] == 1
    assert client.get("/universe/queue").json()["items"] == []
    assert client.get("/improvements/queue").json()["items"] == []


def test_health_ok_and_unavailable() -> None:
    from fastapi.testclient import TestClient

    from dynamo.api.app import create_app

    from .conftest import make_deps

    assert TestClient(create_app(deps=make_deps())).get("/health").json() == {"status": "ok"}
    down = TestClient(create_app(deps=make_deps(ping=False))).get("/health")
    assert down.status_code == 503
    assert down.headers["content-type"].startswith("application/problem+json")


def test_metrics_endpoint_mounted(client) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "python_info" in resp.text
