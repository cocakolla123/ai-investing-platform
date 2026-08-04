"""Prompt files load, versions parse, hardening line present in every prompt."""

from __future__ import annotations

import re

import pytest

from dynamo.research.prompt_loader import (
    HARDENING_MARKER,
    PROMPT_NAMES,
    PromptError,
    load_all_prompts,
    load_prompt,
)

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def test_all_prompts_load() -> None:
    prompts = load_all_prompts()
    assert set(prompts) == set(PROMPT_NAMES)
    assert len(prompts) == 8


def test_versions_are_semver() -> None:
    for prompt in load_all_prompts().values():
        assert SEMVER.match(prompt.version), prompt.name


def test_every_system_prompt_has_hardening_line() -> None:
    for prompt in load_all_prompts().values():
        assert HARDENING_MARKER in prompt.system, prompt.name


def test_system_text_has_no_frontmatter_and_is_nonempty() -> None:
    for prompt in load_all_prompts().values():
        assert prompt.system
        assert not prompt.system.startswith("---")
        assert "version:" not in prompt.system.splitlines()[0]


def test_dossier_prompt_is_stance_free_and_advocates_are_opposed() -> None:
    assert "STANCE-FREE" in load_prompt("dossier_analyst").system
    assert "bullish case" in load_prompt("bull_advocate").system.lower()
    assert "bearish case" in load_prompt("bear_advocate").system.lower()
    arbiter = load_prompt("arbiter").system
    assert "no_action" in arbiter  # abstention rewarded
    assert "kill case" in arbiter.lower()


def test_unknown_prompt_raises() -> None:
    with pytest.raises(PromptError):
        load_prompt("nonexistent")
