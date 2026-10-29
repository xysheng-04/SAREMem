"""Production Search arrival clock, without logging or HTTP metadata."""
from contextvars import ContextVar
import time

request_started = ContextVar('saremem_search_started', default=None)


class SearchRequestClock:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope.get('type') == 'http' and scope.get('method') == 'POST' and
                scope.get('path', '').rstrip('/') == '/search'):
            token = request_started.set(time.monotonic())
            try:
                await self.app(scope, receive, send)
            finally:
                request_started.reset(token)
        else:
            await self.app(scope, receive, send)
