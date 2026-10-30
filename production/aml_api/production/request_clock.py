"""One Add/Search arrival deadline through request reads and response sends."""
from contextvars import ContextVar
import asyncio
from contextlib import suppress
import json
import threading
import time

request_started = ContextVar('saremem_search_started', default=None)
request_deadline = ContextVar('saremem_request_deadline', default=None)
request_cancelled = ContextVar('saremem_request_cancelled', default=None)


class SearchRequestClock:
    def __init__(self, app, policy=None):
        self.app = app
        from .policy import Policy
        self.policy = policy or Policy.from_env()

    async def __call__(self, scope, receive, send):
        if (scope.get('type') == 'http' and scope.get('method') == 'POST' and
                scope.get('path', '').rstrip('/') in {'/add','/search'}):
            from .http_transport import CAPABILITY,ARRIVED,drain,abort
            start = scope.get('extensions',{}).get(ARRIVED,time.monotonic())
            transport = scope.get('extensions',{}).get(CAPABILITY)
            end = start+self.policy.total_seconds
            cancelled = threading.Event()
            token = request_started.set(start)
            dt = request_deadline.set(end)
            ct = request_cancelled.set(cancelled)
            headers_sent = False
            body_done = asyncio.Event()
            receiving = asyncio.Lock()
            async def receive_bounded():
                remaining = end-time.monotonic()
                if remaining <= 0: raise asyncio.TimeoutError()
                if cancelled.is_set(): return {'type':'http.disconnect'}
                async with receiving:
                    message = await asyncio.wait_for(receive(),max(0,end-time.monotonic()))
                if message.get('type') == 'http.disconnect': cancelled.set()
                if message.get('type') == 'http.request' and not message.get('more_body',False): body_done.set()
                return message
            async def watch_disconnect():
                try:
                    await body_done.wait()
                    async with receiving:
                        message = await receive()
                    if message.get('type') == 'http.disconnect': cancelled.set()
                except (OSError,RuntimeError):
                    cancelled.set()
            async def send_bounded(message):
                nonlocal headers_sent
                remaining = end-time.monotonic()
                if remaining <= 0: raise asyncio.TimeoutError()
                if message.get('type') == 'http.response.start': headers_sent = True
                if message.get('type') == 'http.response.body' and len(message.get('body',b'')) > 16384:
                    body = message['body']
                    for offset in range(0,len(body),16384):
                        last = offset+16384 >= len(body)
                        await asyncio.wait_for(send({**message,'body':body[offset:offset+16384],
                            'more_body':message.get('more_body',False) if last else True}),max(0,end-time.monotonic()))
                else:
                    await asyncio.wait_for(send(message),remaining)
            watcher = asyncio.create_task(watch_disconnect())
            try:
                await asyncio.wait_for(self.app(scope,receive_bounded,send_bounded),
                                       max(0,end-time.monotonic()))
                await drain(transport,end)
            except asyncio.TimeoutError:
                cancelled.set()
                if headers_sent:
                    # Status cannot be replaced once sent. Abort, never append a second response.
                    if transport is not None: abort(transport)
                    raise RuntimeError('memory response deadline exceeded') from None
                body = json.dumps({'detail':{'reason':'memory request deadline exceeded; retry with backoff'}}).encode()
                # Bounded final error attempt; a disconnected peer may receive no response.
                try:
                    await asyncio.wait_for(send({'type':'http.response.start','status':504,
                        'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]}),1)
                    await asyncio.wait_for(send({'type':'http.response.body','body':body}),1)
                except (asyncio.TimeoutError, OSError): pass
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                watcher.cancel()
                with suppress(asyncio.CancelledError): await watcher
                request_started.reset(token)
                request_deadline.reset(dt)
                request_cancelled.reset(ct)
        else:
            await self.app(scope, receive, send)
