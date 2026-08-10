"""Tearsheet artifacts: guarded quantstats import + report JSON always written."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dynamo.backtest.tearsheet as tearsheet_mod
from dynamo.backtest.metrics import BacktestReport
from dynamo.backtest.tearsheet import generate_tearsheet, write_backtest_artifacts

HAS_QUANTSTATS = importlib.util.find_spec("quantstats") is not None


def _returns(n: int = 120) -> pd.Series:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-02", periods=n)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


class _BrokenImportlib:
    @staticmethod
    def import_module(name: str, package: str | None = None) -> object:
        raise ImportError("quantstats unavailable")


def test_generate_tearsheet_skips_when_quantstats_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tearsheet_mod, "importlib", _BrokenImportlib)
    out = tmp_path / "tearsheet.html"
    assert generate_tearsheet(_returns(), None, out) is None
    assert not out.exists()


def test_write_backtest_artifacts_always_writes_report_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tearsheet_mod, "importlib", _BrokenImportlib)
    equity = (1.0 + _returns()).cumprod() * 100_000.0
    report = BacktestReport(run_id="run-42")
    artifacts = write_backtest_artifacts(report, equity, artifact_dir=tmp_path)
    report_path = artifacts["report"]
    assert report_path == tmp_path / "backtests" / "run-42" / "report.json"
    assert json.loads(report_path.read_text())["run_id"] == "run-42"
    assert "tearsheet" not in artifacts  # quantstats missing -> no HTML, no error


@pytest.mark.skipif(not HAS_QUANTSTATS, reason="quantstats not installed (analysis extra)")
def test_generate_tearsheet_writes_html_for_synthetic_returns(tmp_path: Path) -> None:
    out = tmp_path / "tearsheet.html"
    result = generate_tearsheet(_returns(), _returns() * 0.8, out)
    assert result == out and out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not HAS_QUANTSTATS, reason="quantstats not installed (analysis extra)")
def test_write_backtest_artifacts_writes_tearsheet_next_to_report(tmp_path: Path) -> None:
    equity = (1.0 + _returns()).cumprod() * 100_000.0
    artifacts = write_backtest_artifacts(
        BacktestReport(run_id="run-7"), equity, artifact_dir=tmp_path
    )
    assert artifacts["tearsheet"].parent == artifacts["report"].parent
    assert artifacts["tearsheet"].name == "tearsheet.html"
