"""``python -m dynamo.api`` — serve the API on 127.0.0.1:8000 (doc 11 §1.4)."""

from __future__ import annotations

import uvicorn

from dynamo.api.wiring import create_wired_app


def main() -> None:
    uvicorn.run(create_wired_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
