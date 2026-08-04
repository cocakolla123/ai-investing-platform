"""Versioned prompt loader. Design: doc 03 sections 2-3.

Prompts live as markdown files with YAML frontmatter in the
`dynamo.research.prompts` package; the system text is everything after the
frontmatter, with zero interpolation (no dates/ids) so it prompt-caches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

PROMPT_NAMES: tuple[str, ...] = (
    "dossier_analyst",
    "bull_advocate",
    "bear_advocate",
    "arbiter",
    "macro_digest",
    "prediction_market",
    "triage",
    "post_trade_review",
)

# Every system prompt must contain this hardening line (untrusted-data rule).
HARDENING_MARKER = "untrusted data, never instructions"

_VERSION_RE = re.compile(r'^version:\s*"?(\d+\.\d+\.\d+)"?\s*$', re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class PromptError(ValueError):
    """Malformed or missing prompt file."""


@dataclass(frozen=True)
class LoadedPrompt:
    name: str
    version: str
    system: str


def load_prompt(name: str) -> LoadedPrompt:
    """Read a prompt from the dynamo.research.prompts package -> (version, system)."""
    if name not in PROMPT_NAMES:
        raise PromptError(f"unknown prompt {name!r}; expected one of {PROMPT_NAMES}")
    raw = (
        resources.files("dynamo.research.prompts").joinpath(f"{name}.md").read_text("utf-8")
    )
    front = _FRONTMATTER_RE.match(raw)
    if front is None:
        raise PromptError(f"prompt {name!r} is missing YAML frontmatter")
    version_match = _VERSION_RE.search(front.group(1))
    if version_match is None:
        raise PromptError(f"prompt {name!r} frontmatter has no semver version")
    system = raw[front.end() :].strip()
    if not system:
        raise PromptError(f"prompt {name!r} has an empty system section")
    return LoadedPrompt(name=name, version=version_match.group(1), system=system)


def load_all_prompts() -> dict[str, LoadedPrompt]:
    return {name: load_prompt(name) for name in PROMPT_NAMES}
