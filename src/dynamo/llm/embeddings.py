"""Local journal embeddings — ``EmbeddingModel`` port (doc 10 §6).

Uses sentence-transformers bge-small (384-d, zero API cost) when the optional
``embeddings`` extra is installed; otherwise falls back to deterministic
sha256-seeded pseudo-vectors of the same dimension so the journal pipeline
stays functional (search quality degrades, the contract does not).
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

DIM = 384
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BACKEND_SENTENCE_TRANSFORMERS = "sentence-transformers"
BACKEND_HASH_FALLBACK = "hash-fallback"


def _hash_vector(text: str) -> list[float]:
    """Deterministic normalized pseudo-embedding seeded from sha256(text)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DIM)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:  # pragma: no cover - standard_normal never returns all zeros
        return [0.0] * DIM
    return (vec / norm).astype(float).tolist()


class LocalEmbedder:
    """``EmbeddingModel`` implementation: bge-small when importable, else
    the deterministic hash fallback. ``backend`` reports which one is live."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model: Any = None
        self._backend = BACKEND_HASH_FALLBACK
        try:
            st = importlib.import_module("sentence_transformers")
            self._model = st.SentenceTransformer(model_name)
            self._backend = BACKEND_SENTENCE_TRANSFORMERS
        except Exception:
            self._model = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        return DIM

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is not None:
            vectors = self._model.encode(list(texts), normalize_embeddings=True)
            return [[float(x) for x in row] for row in vectors]
        return [_hash_vector(t) for t in texts]
