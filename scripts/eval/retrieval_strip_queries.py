"""Rebuild evals/retrieval/queries_journal_stripped.jsonl from queries.jsonl.

The Phase 2 contamination correction (0e70665) stripped each query text at
its "## Journal hits" section: those lines carried prior-case outcomes. That
file was made without a committed script; this is the rule, reproduced
byte for byte (asserted by --check). Everything from the heading onward is
removed, including the trailing task instruction that followed it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL = Path("evals/retrieval")
MARKER = "## Journal hits"


def strip(query: dict) -> dict:
    text = query["text"]
    cut = text.find(MARKER)
    kept = text if cut < 0 else text[:cut]
    return {**query, "text": kept, "stripped_chars": len(text) - len(kept)}


def main() -> None:
    src = [json.loads(line) for line in (EVAL / "queries.jsonl").open()]
    out = "".join(json.dumps(strip(q)) + "\n" for q in src)
    target = EVAL / "queries_journal_stripped.jsonl"
    if "--check" in sys.argv:
        same = out == target.read_text()
        print("identical" if same else "DIFFERENT")
        raise SystemExit(0 if same else 1)
    target.write_text(out)


if __name__ == "__main__":
    main()
