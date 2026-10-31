"""FastAPI adapter for the AML Cycle 2 contract."""
from __future__ import annotations

import hmac

from fastapi import FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import ServiceError, Unauthorized
from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .service_factory import create_service


def create_app(settings: Settings | None = None, service=None, *, production: bool = False) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or create_service(settings, production=production)
    if production and not settings.memory_system_key:
        raise ValueError("competition API requires a Memory System Key")
    app = FastAPI(title="SAREMem AML API", version="1.0.0")
    if production:
        from .production.request_clock import SearchRequestClock
        app.add_middleware(SearchRequestClock, policy=getattr(service, 'policy', None))

    def authenticate(authorization: str | None, x_api_key: str | None) -> None:
        from .production.diagnostics import stage
        stage('authentication')
        expected = settings.memory_system_key
        if expected is None:
            return
        supplied: list[str] = []
        if x_api_key:
            supplied.append(x_api_key)
        if authorization:
            scheme, separator, credential = authorization.partition(" ")
            if separator and scheme.lower() in {"bearer", "token"}:
                supplied.append(credential)
        if not any(hmac.compare_digest(value.encode('utf-8'), expected.encode('utf-8'))
                   for value in supplied):
            raise Unauthorized("invalid Memory System Key")

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError):
        from .production.diagnostics import active, error
        from .production.policy import classify
        record = active.get()
        if record is not None and record.get('code') == 'http_error':
            error(classify(exc).code)
        from .production.http_errors import reason, retry_headers
        headers = retry_headers(exc) if production else (
            {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"reason": reason(exc.status_code, exc.reason) if production else exc.reason}},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        from .production.http_errors import validation_details
        errors = validation_details(exc.errors()) if production else jsonable_encoder(exc.errors())
        if production:
            from .production.diagnostics import validation_failure
            validation_failure(errors)
        return JSONResponse(status_code=422, content={"detail": errors})

    if production:
        from starlette.exceptions import HTTPException
        from .production.http_errors import reason

        @app.exception_handler(HTTPException)
        async def http_error_handler(_: Request, exc: HTTPException):
            # Includes framework 404/405 and parsing errors, without reflecting URLs or bodies.
            headers = None
            if exc.status_code == 405 and exc.headers:
                allow = exc.headers.get('Allow', '')
                methods = {part.strip() for part in allow.split(',')}
                if methods and methods <= {'GET', 'POST', 'HEAD', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'TRACE', 'CONNECT'}:
                    headers = {'Allow': allow}
            return JSONResponse(status_code=exc.status_code,
                                content={"detail": {"reason": reason(exc.status_code)}}, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, __: Exception):
        from .production.diagnostics import active, error
        error('internal')
        if active.get() is not None: active.get()['status_attempted'] = 500
        # Do not leak keys, provider payloads, Base64 images, or stack traces to AML.
        return JSONResponse(
            status_code=500,
            content={"detail": {"reason": "internal memory service error"}},
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse)
    def add(
        body: AddRequest,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    ):
        authenticate(authorization, x_api_key)
        from .production.diagnostics import active, stage
        import threading
        record = active.get()
        complete = threading.Event()
        if record is not None: record['_completion'] = complete
        try: result = service.add(body)
        finally: complete.set()
        stage('response_validation')
        if production:
            # A valid-looking 200 must never mis-echo a logical write's identity.
            response = AddResponse.model_validate(result)
            if (response.request_id, response.user_id, response.session_id) != (
                    body.request_id, body.user_id, body.session_id):
                raise ServiceError('invalid Add response identity')
            return response
        return result

    @app.post("/search", response_model=SearchResponse)
    def search(
        body: SearchRequest,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    ):
        authenticate(authorization, x_api_key)
        from .production.diagnostics import active
        import threading
        record = active.get(); complete = threading.Event()
        if record is not None: record['_completion'] = complete
        try: return service.search(body)
        finally: complete.set()

    return app
