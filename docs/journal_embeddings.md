# Journal embeddings: the hash fallback, and how to get rid of it

## TL;DR — the fix

```bash
uv sync --extra embeddings
uv run python -m dynamo.cli.journal_seed --reembed
```

The first command installs `sentence-transformers` (bge-small, 384-d, local,
zero API cost). The second re-embeds, **in place**, every `journal_entries`
row stamped `embedding_version = 'hash-fallback'` — touching only the vector,
never the text and never `created_at`, so no point-in-time read changes.

Journal retrieval re-enables itself the moment the entries stop being
fallback-stamped. There is no setting to flip back.

## What is broken

`dynamo.llm.embeddings.LocalEmbedder` tries to import `sentence_transformers`
and, if that fails, falls back to `_hash_vector`:

```python
seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
rng = np.random.default_rng(seed)
vec = rng.standard_normal(DIM)
```

That is **a random Gaussian seeded from the text's hash**. It is
deterministic, correctly normalized and the right dimension, so nothing
downstream errors — and it carries exactly zero semantic signal. Two entries
about the same thesis in the same sector are as far apart as two entries about
nothing in common. Cosine similarity over these vectors is a random ordering
of the journal.

All **113** rows currently in `journal_entries` were written this way.

## Why that is not merely "degraded search"

`journal_retrieval_enabled` defaults to `True`, and the hits go into the
committee's dossier context under `## Journal hits`, introduced as *similar
past situations*. So the failure mode is not "the search box is bad". It is:

> randomly selected past trades are shown to the decision-making committee
> as precedent, and nothing downstream can tell them from real precedent.

That is worse than showing no precedent at all. A missing section is
information; a fabricated one is not.

## What the code does now

`dynamo.learning.journal` holds the decision, pure, so the live and replay
paths cannot drift apart on it:

- **`journal_retrieval_gate(query_backend=..., allow_hash_fallback=...)`** —
  refuses to run vector search at all when the *query* embedder is the
  fallback. The caller logs `gate.reason` at WARNING and skips retrieval. It
  does **not** silently substitute text search: that would change what the
  committee reads without saying so.
- **`drop_hash_fallback_entries(hits, versions, ...)`** — drops candidates
  whose *stored* vector is fallback-stamped. A real query vector ranked
  against stored random ones is just as meaningless, and a partial re-embed
  leaves both kinds in the table at once. An entry with no stamp is assumed
  good: an absent stamp predates the column rather than denying it.

Both are wired into the live path in
`orchestration/assets/research.py::_journal_hits_for`.

The stored-vector rule is *also* enforced one level down, in
`JournalRepository.search`: a vector search excludes fallback-stamped rows in
SQL unless `allow_hash_fallback=True`. That is deliberate placement — the
repository is the one seam **every** caller goes through, live and replay
alike, so a new caller is protected by default rather than by remembering to
ask. The caller-side filter stays as well, because it is the only place that
can *count* what it removed and say so in the log.

Text search is unaffected in both layers: an ILIKE match is real whatever the
row's vector holds.

Structured retrieval is untouched. The prior-similar-case lines that share the
`journal_hits` channel come from `decision_experience` via SQL filters and use
no embedding at all, so they keep flowing. **The abstention removes the noise,
not the evidence.**

## The escape hatch

`DYNAMO_JOURNAL_ALLOW_HASH_FALLBACK=true` (settings:
`journal_allow_hash_fallback`, default `False`) restores the old behaviour —
for reproducing a historical run byte-for-byte, and for nothing else. It still
logs a warning on every run that says the committee is being shown random past
situations as precedent.

## Why this was not just fixed

`uv sync --extra embeddings` rewrites the virtualenv, and a multi-hour
`dynamo.cli.quant_train` campaign was running out of it at the time. Installing
under a running job is how you get a half-finished campaign and an unclear
cause. The guard turns a silent correctness bug into a visible, logged
abstention; the two commands at the top of this file are the actual repair, and
they are safe to run the moment the training job finishes.

## Verifying

```sql
SELECT embedding_version, count(*) FROM journal_entries GROUP BY 1;
```

Before: `('hash-fallback', 113)`. After the re-embed:
`('sentence-transformers', 113)`.
