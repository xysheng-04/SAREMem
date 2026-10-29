"""Deployment-time selection between frozen legacy and independent Evidence Search."""
from __future__ import annotations

import os
from typing import Any

from .config import Settings
from .service import RouteAService


def search_pipeline_from_env() -> str:
    pipeline = os.getenv("AML_SEARCH_PIPELINE", "legacy").strip().lower()
    if pipeline not in {"legacy", "evidence", "simple"}:
        raise ValueError("AML_SEARCH_PIPELINE must be 'legacy', 'evidence' or 'simple'")
    return pipeline


def create_service(settings: Settings, *, pipeline: str | None = None, production: bool = False, **kwargs: Any):
    selected = pipeline or search_pipeline_from_env()
    if selected == "legacy":
        return RouteAService(settings, **kwargs)
    if selected == "simple":
        if not production:
            raise ValueError("offline simple runners use the preserved experiment tree")
        from .production.service import ProductionService
        return ProductionService(settings, **kwargs)
    if selected != "evidence":
        raise ValueError("pipeline must be 'legacy', 'evidence' or 'simple'")
    # Lazy import proves the default legacy deployment does not instantiate Evidence settings.
    raise ValueError("evidence experiments are not included in the competition runtime")
