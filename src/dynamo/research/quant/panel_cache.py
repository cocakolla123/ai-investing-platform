"""Disk cache of per-session feature frames for quant training.

``dynamo.cli.quant_train`` rebuilds every session's feature frame from raw
database reads through the point-in-time gateway (~seconds per session, hours
per multi-year window). The frames are deterministic given (as_of, universe,
feature code), so re-runs — tuning iterations especially — can reuse them
from disk and finish in minutes.

Keying (staleness safety): a cache directory is keyed by BOTH

* ``universe_hash`` — sha256 of the sorted symbol list. Features are
  cross-sectionally normalized, so the same symbol's row differs under a
  different universe; frames are only reusable for the identical symbol set.
* ``catalog_fingerprint`` — sha256 over the ordered ``FEATURE_CATALOG``
  names AND the parsed AST of the catalog + feature-builder modules. Any
  feature added, removed, reordered, or re-implemented changes the
  fingerprint, so stale frames are never silently reused; REFORMATTING
  does not, which hashing raw source text got wrong.

Within a directory there is one parquet file per session date. Writes are
crash-safe (tmp file + atomic rename) and happen after each session build, so
an interrupted training run still leaves every completed session cached.
Corrupted or unreadable files behave as cache misses (and are deleted
best-effort) — the caller rebuilds, never crashes.

Labels are NOT cached: ``build_labels`` slices one preloaded bars frame and
is cheap relative to a gateway ``feature_inputs`` round-trip.

Replay note (documented decision): ``dynamo.replay.runner.quant_scores_for``
builds features only for the session's *missing* symbols — a varying subset
whose cross-sectional normalization differs from the full-universe frame —
so wiring this cache into replay would change replay's scores. It stays a
training-only cache.

This module does file I/O only (stdlib + pandas/pyarrow): no database, no
settings — callers (the CLI composition layer) choose the root directory.
The default root ``artifacts/panel_cache/`` is gitignored (``/artifacts/``
and ``*.parquet`` rules both cover it).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["PanelFeatureCache", "catalog_fingerprint", "universe_hash"]


def universe_hash(universe: Sequence[str]) -> str:
    """sha256 of the sorted symbol set — order-insensitive, set-sensitive."""
    payload = json.dumps(sorted(str(s) for s in universe))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_hash(module: Any) -> str:
    """SHA-256 of a module's parsed AST, not its source text.

    Formatting must not invalidate the cache. It used to: the fingerprint
    hashed raw source, so ``ruff format`` reflowing ``catalog.py`` — a
    whitespace-only change — threw away 100 cached sessions, which at 1,315
    symbols is 4.3 hours of recompute for zero semantic difference. The old
    docstring's "cost of a false rebuild is minutes" was written when the
    panel was 55 names and stopped being true.

    ``ast.dump`` still changes on any rename, reorder, constant edit or
    implementation change, so the property that matters — never reuse a
    cache across a real feature change — is unchanged. Comments and
    whitespace drop out; docstrings do not (they are AST constants), which
    errs in the safe direction.
    """
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):  # pragma: no cover - no readable source
        return hashlib.sha256(repr(module).encode("utf-8")).hexdigest()
    try:
        digest_input = ast.dump(ast.parse(source))
    except SyntaxError:  # pragma: no cover - unparseable source
        digest_input = source
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def _function_hash(module_name: str, function: str) -> str:
    """AST hash of one top-level function, read from source WITHOUT importing
    its module (research code may not import the persistence layer)."""
    spec = importlib.util.find_spec(module_name)
    origin = spec.origin if spec is not None else None
    if origin is None:  # pragma: no cover - module always present in-repo
        return ""
    tree = ast.parse(Path(origin).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return hashlib.sha256(ast.dump(node).encode("utf-8")).hexdigest()
    return ""


def catalog_fingerprint() -> str:
    """Version fingerprint of the feature-building code.

    Hashes the ordered ``FEATURE_CATALOG`` names plus the parsed AST of the
    catalog and feature-builder modules: any rename, reorder, or
    implementation change yields a new fingerprint (and thus a cold cache),
    while reformatting does not. The cost of a stale reuse would be a
    silently wrong model; the cost of a false rebuild is now hours.
    """
    from dynamo.research.quant import catalog, features

    payload = json.dumps(
        {
            "feature_names": list(catalog.FEATURE_CATALOG),
            "catalog_src": _semantic_hash(catalog),
            "features_src": _semantic_hash(features),
            # The bars every feature is built from: a change to how feeds are
            # merged (2026-10-03: Tiingo open/high/low moved onto the adjusted
            # scale) changes features without touching the feature code.
            "canonical_bars_src": _function_hash(
                "dynamo.persistence.repositories.market", "canonical_bars"
            ),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PanelFeatureCache:
    """Per-session feature-frame cache under ``root``/<fingerprint>-<universe>/.

    ``get`` returns the cached frame for a session date or ``None`` (missing
    OR unreadable — corruption is a miss, never an error). ``put`` writes
    atomically (tmp + rename), so a crash mid-write can never leave a
    half-written file behind under the final name.
    """

    def __init__(
        self,
        root: str | Path,
        universe: Sequence[str],
        *,
        fingerprint: str | None = None,
    ) -> None:
        fp = fingerprint if fingerprint is not None else catalog_fingerprint()
        uh = universe_hash(universe)
        self.directory = Path(root) / f"{fp[:16]}-{uh[:16]}"

    def path_for(self, as_of: date) -> Path:
        return self.directory / f"{as_of.isoformat()}.parquet"

    def get(self, as_of: date) -> pd.DataFrame | None:
        path = self.path_for(as_of)
        if not path.is_file():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            # Corrupted / truncated file (e.g. a hard kill mid-copy of the
            # whole directory): treat as a miss and clear it so the rebuilt
            # frame can be re-cached.
            with contextlib.suppress(OSError):
                path.unlink()
            return None

    def put(self, as_of: date, frame: pd.DataFrame) -> None:
        path = self.path_for(as_of)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(tmp)
            os.replace(tmp, path)  # atomic within one filesystem
        finally:
            if tmp.exists():  # failed write — never leave tmp litter
                with contextlib.suppress(OSError):
                    tmp.unlink()
