"""LocalEmbedder (doc 10 §6): 384-d vectors from either backend; the hash
fallback is deterministic and normalized."""

from __future__ import annotations

import math

from dynamo.domain.ports import EmbeddingModel
from dynamo.llm.embeddings import (
    BACKEND_HASH_FALLBACK,
    BACKEND_SENTENCE_TRANSFORMERS,
    LocalEmbedder,
    _hash_vector,
)


def test_embedder_satisfies_port_and_reports_backend() -> None:
    embedder: EmbeddingModel = LocalEmbedder()  # structural port conformance
    assert embedder.dim == 384
    backend = LocalEmbedder().backend
    assert backend in (BACKEND_SENTENCE_TRANSFORMERS, BACKEND_HASH_FALLBACK)


def test_embed_shapes_and_empty() -> None:
    embedder = LocalEmbedder()
    assert embedder.embed([]) == []
    vectors = embedder.embed(["thesis text", "another entry"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)


def test_hash_fallback_deterministic_and_distinct() -> None:
    a1 = _hash_vector("semis before the correction")
    a2 = _hash_vector("semis before the correction")
    b = _hash_vector("a completely different memory")
    assert a1 == a2
    assert a1 != b


def test_hash_fallback_normalized() -> None:
    vec = _hash_vector("normalize me")
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)
