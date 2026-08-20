"""--symbols-file parsing (universe-breadth prep): one symbol per line,
'#' comments, blank lines, dedupe — shared by backfill bars and quant_train."""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamo.cli import quant_train as qt
from dynamo.cli.backfill import _build_parser as backfill_parser
from dynamo.cli.backfill import read_symbols_file


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "universe.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_symbols_file_parses_symbols_comments_and_blanks(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
# Training universe (large caps) — full-line comment
AAPL
MSFT   # trailing comment
  NVDA

BRK-B
BTC/USD  # crypto pairs keep their slash and case
#GOOGL entirely commented out
""",
    )
    assert read_symbols_file(path) == ["AAPL", "MSFT", "NVDA", "BRK-B", "BTC/USD"]


def test_read_symbols_file_dedupes_keeping_first_occurrence(tmp_path: Path) -> None:
    path = _write(tmp_path, "AAPL\nMSFT\nAAPL\nMSFT # again\nAMZN\n")
    assert read_symbols_file(path) == ["AAPL", "MSFT", "AMZN"]


def test_read_symbols_file_empty_and_comment_only_files_yield_nothing(
    tmp_path: Path,
) -> None:
    assert read_symbols_file(_write(tmp_path, "")) == []
    assert read_symbols_file(_write(tmp_path, "# only comments\n\n  # more\n")) == []


def test_example_training_universe_file_parses() -> None:
    """The shipped example file is valid input for --symbols-file."""
    root = Path(__file__).resolve().parents[3]
    example = root / "config" / "training_universe.txt.example"
    symbols = read_symbols_file(example)
    assert len(symbols) >= 100
    assert "AAPL" in symbols and "BRK-B" in symbols
    assert len(set(symbols)) == len(symbols)  # no duplicates in the example


def test_backfill_bars_accepts_symbols_file_flag(tmp_path: Path) -> None:
    path = _write(tmp_path, "AAPL\n")
    args = backfill_parser().parse_args(
        ["bars", "--start", "2025-01-01", "--end", "2026-01-01", "--symbols-file", str(path)]
    )
    assert args.symbols_file == path
    assert args.symbols is None and args.universe is False
    # Mutually exclusive with --symbols / --universe.
    with pytest.raises(SystemExit):
        backfill_parser().parse_args(
            [
                "bars",
                "--start",
                "2025-01-01",
                "--end",
                "2026-01-01",
                "--symbols-file",
                str(path),
                "--symbols",
                "AAPL",
            ]
        )


def test_backfill_bars_refuses_empty_symbols_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dynamo.cli import backfill

    class Settings:
        tiingo_api_key = "key"

    monkeypatch.setattr(backfill, "_settings", lambda: Settings())
    monkeypatch.setattr(
        backfill, "_repo", lambda url: (_ for _ in ()).throw(AssertionError("no DB"))
    )
    path = _write(tmp_path, "# nothing here\n")
    rc = backfill.main(
        ["bars", "--start", "2025-01-01", "--end", "2026-01-01", "--symbols-file", str(path)]
    )
    assert rc == 1  # refused before any engine/network work


def test_quant_train_accepts_symbols_file_flag(tmp_path: Path) -> None:
    path = _write(tmp_path, "AAPL\n")
    base = ["--train-start", "2024-01-02", "--train-end", "2025-01-02", "--label", "x"]
    args = qt._build_parser().parse_args([*base, "--symbols-file", str(path)])
    assert args.symbols_file == path
    with pytest.raises(SystemExit):  # mutually exclusive with --symbols
        qt._build_parser().parse_args([*base, "--symbols-file", str(path), "--symbols", "AAPL"])
    with pytest.raises(SystemExit):  # and with --universe
        qt._build_parser().parse_args([*base, "--symbols-file", str(path), "--universe"])
