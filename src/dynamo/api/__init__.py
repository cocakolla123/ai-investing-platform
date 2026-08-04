"""Delivery layer: FastAPI app + admin actions (doc 11). May import everything."""

from dynamo.api.app import create_app

__all__ = ["create_app"]
