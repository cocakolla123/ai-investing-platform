"""Backward-compatible alias for :mod:`dynamo.research.shortlist`.

The module moved to ``dynamo.research`` because the nightly discovery
funnel needs the same diversity controls the replay slate uses, and the
import contract "core logic does not import the replay composition root"
(pyproject ``[tool.importlinter]``) rightly forbids ``dynamo.research``
from reaching into ``dynamo.replay``. The logic was always pure — it
imports nothing but ``dynamo.domain.symbols`` — so it was simply living in
the wrong package.

Nothing here is new behavior; ``dynamo.replay.committee`` and the replay
tests keep their existing imports.
"""

from __future__ import annotations

from dynamo.research.shortlist import (
    CRYPTO_BUCKET,
    UNKNOWN_BUCKET,
    SessionHistory,
    bucket_of,
    select_shortlist,
)

__all__ = [
    "CRYPTO_BUCKET",
    "UNKNOWN_BUCKET",
    "SessionHistory",
    "bucket_of",
    "select_shortlist",
]
