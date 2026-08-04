"""FastAPI application factory (doc 11 §1).

Delivery layer only: bind routers, auth, error handling, health, metrics.
Composition (engine → stores) happens in ``dynamo.api.wiring``; tests inject a
``Deps`` of fakes directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI
from prometheus_client import make_asgi_app

from dynamo.api.deps import Deps, require_auth
from dynamo.api.envelopes import install_error_handlers, problem_response
from dynamo.api.routers import admin, journal, ops, portfolio, reports, risk

if TYPE_CHECKING:
    from sqlalchemy import Engine


def create_app(engine: Engine | None = None, *, deps: Deps | None = None) -> FastAPI:
    """Build the API app. ``deps`` overrides for tests; otherwise the real
    stores are wired from ``engine`` (or the configured database when None).

    CORS stays off by design: single user, same-origin dashboard (doc 11 §1.4).
    """
    if deps is None:
        from dynamo.api.wiring import build_deps  # avoid import cycle

        deps = build_deps(engine)

    app = FastAPI(title="Dynamo", docs_url="/docs", redoc_url=None)
    app.state.deps = deps
    install_error_handlers(app)

    authed = [Depends(require_auth)]
    for router in (
        reports.router,
        portfolio.router,
        risk.router,
        journal.router,
        ops.router,
        admin.router,
    ):
        app.include_router(router, dependencies=authed)

    @app.get("/health", tags=["ops"])
    def health():  # pyright: ignore[reportUnusedFunction]
        if not deps.ping():
            return problem_response(503, "Service Unavailable", "database ping failed")
        return {"status": "ok"}

    # Prometheus /metrics (mounted ASGI app; unauthenticated, localhost-bound).
    app.mount("/metrics", make_asgi_app())

    return app
