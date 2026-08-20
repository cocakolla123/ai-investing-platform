"""The kill switch is ONE file, whatever the process CWD is (doc 09 §7).

``kill_switch_path`` defaulted to the relative ``"KILL_SWITCH"``, resolved
against each process's working directory: the Dagster daemon, the API and an
operator's shell each got a different file, so "engaging" the switch in one
shell was invisible to the process actually placing orders.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dynamo.config.settings import Settings
from dynamo.execution.gate import kill_switch_engaged


def _settings(**env: str) -> Settings:
    return Settings(**env)  # type: ignore[arg-type]


def test_relative_path_resolves_identically_from_two_cwds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / "sub").mkdir(parents=True)
    monkeypatch.setenv("DYNAMO_HOME", str(home))

    monkeypatch.chdir(home)
    from_root = _settings().kill_switch_file
    monkeypatch.chdir(home / "sub")
    from_sub = _settings().kill_switch_file

    assert from_root == from_sub == (home / "KILL_SWITCH")


def test_a_switch_engaged_from_one_cwd_is_seen_from_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The behavioural version: write it here, see it there."""
    home = tmp_path / "home"
    (home / "daemon-cwd").mkdir(parents=True)
    monkeypatch.setenv("DYNAMO_HOME", str(home))

    monkeypatch.chdir(home)
    _settings().kill_switch_file.write_text("engaged_by=operator\n")

    monkeypatch.chdir(home / "daemon-cwd")
    assert kill_switch_engaged(_settings().kill_switch_file) is True
    # ...and the pre-fix behaviour would have missed it:
    assert kill_switch_engaged(Path(_settings().kill_switch_path)) is False


def test_absolute_paths_are_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    absolute = tmp_path / "elsewhere" / "SWITCH"
    monkeypatch.setenv("DYNAMO_KILL_SWITCH_PATH", str(absolute))
    monkeypatch.setenv("DYNAMO_HOME", str(tmp_path / "home"))
    assert _settings().kill_switch_file == absolute


def test_home_defaults_to_the_repo_root_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DYNAMO_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    home = _settings().home_dir
    assert home != tmp_path.resolve()
    assert (home / "pyproject.toml").is_file()


def test_the_returned_path_is_always_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYNAMO_HOME", raising=False)
    assert _settings().kill_switch_file.is_absolute()
    assert _settings().approvals_path.is_absolute()


def test_home_expands_a_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYNAMO_HOME", "~/dynamo-runtime")
    expected = Path(os.path.expanduser("~/dynamo-runtime")).resolve()
    assert _settings().kill_switch_file == expected / "KILL_SWITCH"


def test_api_wiring_uses_the_anchored_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: build_deps used to hand FileKillSwitch the raw relative string."""
    from dynamo.api.wiring import FileKillSwitch

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DYNAMO_HOME", str(home))
    settings = _settings()
    monkeypatch.chdir(tmp_path)

    switch = FileKillSwitch(settings.kill_switch_file, market_repo=None)  # type: ignore[arg-type]
    assert switch.engaged() is False
    (home / "KILL_SWITCH").write_text("x")
    assert switch.engaged() is True
