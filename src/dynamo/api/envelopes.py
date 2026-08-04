"""Response envelopes + RFC-9457 problem+json error handling (doc 11 §1.3).

``Page[T]`` for lists, ``AsOf[T]`` for point reads — responses reuse domain
models directly inside these envelopes; no parallel DTO hierarchy.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"


class Page[T](BaseModel):
    """List envelope: ``items``, ``next_cursor``, ``total`` (doc 11 §1.3)."""

    items: list[T]
    next_cursor: str | None = None
    total: int

    @classmethod
    def of(cls, items: list[T], next_cursor: str | None = None) -> Page[T]:
        return cls(items=items, next_cursor=next_cursor, total=len(items))


class AsOf[T](BaseModel):
    """Point-read envelope: the as-of discipline visible in every response."""

    data: T
    as_of: datetime | date | None = None


def problem_response(
    status: int,
    title: str,
    detail: str = "",
    type_: str = "about:blank",
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """An RFC 9457 problem+json response. No stack traces cross this boundary."""
    return JSONResponse(
        status_code=status,
        content={"type": type_, "title": title, "status": status, "detail": detail},
        media_type=PROBLEM_CONTENT_TYPE,
        headers=headers,
    )


_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


def install_error_handlers(app: FastAPI) -> None:
    """Map every error to problem+json with stable ``type`` URIs."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        title = _TITLES.get(exc.status_code, "Error")
        return problem_response(
            exc.status_code,
            title,
            detail=str(exc.detail),
            type_=f"https://dynamo.local/problems/{exc.status_code}",
            headers=dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem_response(
            422,
            "Unprocessable Entity",
            detail=str(exc.errors()),
            type_="https://dynamo.local/problems/validation",
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled API error", exc_info=exc)
        return problem_response(
            500,
            "Internal Server Error",
            detail="internal error",  # deliberately opaque: no stack traces
            type_="https://dynamo.local/problems/internal",
        )
