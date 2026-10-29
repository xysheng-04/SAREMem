"""Domain errors translated to AML-compatible HTTP status codes by app.py."""
from __future__ import annotations


class ServiceError(Exception):
    status_code = 500

    def __init__(self, reason: str, *, retry_after: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class BadRequest(ServiceError):
    status_code = 400


class Unauthorized(ServiceError):
    status_code = 401


class Forbidden(ServiceError):
    status_code = 403


class Conflict(ServiceError):
    status_code = 409


class UpstreamUnavailable(ServiceError):
    status_code = 503
