"""One Add/Search arrival deadline through request reads and response sends."""
from contextvars import ContextVar
import asyncio
from contextlib import suppress
import json
import threading
import time
import uuid
from collections import deque
from .diagnostics import active, stage, error, FailureSink

request_started = ContextVar('saremem_search_started', default=None)
request_deadline = ContextVar('saremem_request_deadline', default=None)
request_cancelled = ContextVar('saremem_request_cancelled', default=None)


class SearchRequestClock:
    def __init__(self, app, policy=None):
        self.app = app
        from .policy import Policy
        self.policy = policy or Policy.from_env()
        import os
        from pathlib import Path
        self.capacity = int(os.environ.get('AML_PROD_MAX_INFLIGHT', '32'))
        if not 1 <= self.capacity <= 32: raise ValueError('invalid admission capacity')
        # One executing request keeps existing database fencing and source generation stable.
        self.execution = asyncio.Lock()
        self.inflight = 0
        self.waiters = deque()
        directory = Path(__file__).resolve().parents[3]/'private/diagnostics'
        self.sink = None
        if os.environ.get('AML_FAILURE_DIAGNOSTICS','on') != 'off':
            try: self.sink = FailureSink(directory)
            except Exception: pass

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
            record = {'correlation':uuid.uuid4().hex, 'operation':scope['path'].strip('/').lower(),
                      'stage':'admission','code':'http_error','status_attempted':None,
                      'response_started':False,'response_completed':False,'delivery':'unconfirmed', 'queue_ms':0}
            diagnostic_token = active.set(record)
            headers_sent = False
            admitted = False
            executing = False
            body_done = asyncio.Event()
            receiving = asyncio.Lock()
            async def receive_bounded():
                remaining = end-time.monotonic()
                if remaining <= 0: raise asyncio.TimeoutError()
                if cancelled.is_set(): return {'type':'http.disconnect'}
                async with receiving:
                    message = await asyncio.wait_for(receive(),max(0,end-time.monotonic()))
                if message.get('type') == 'http.disconnect' and not record['response_completed']: cancelled.set()
                if message.get('type') == 'http.request' and not message.get('more_body',False): body_done.set()
                return message
            async def watch_disconnect():
                try:
                    await body_done.wait()
                    async with receiving:
                        message = await receive()
                    if message.get('type') == 'http.disconnect' and not record['response_completed']: cancelled.set()
                except (OSError,RuntimeError):
                    cancelled.set()
            async def send_bounded(message):
                nonlocal headers_sent
                remaining = end-time.monotonic()
                if remaining <= 0: raise asyncio.TimeoutError()
                if message.get('type') == 'http.response.start':
                    headers_sent = True
                    record['status_attempted'] = message['status']
                    if message['status'] < 400: stage('response_send')
                if message.get('type') == 'http.response.body' and len(message.get('body',b'')) > 16384:
                    body = message['body']
                    for offset in range(0,len(body),16384):
                        last = offset+16384 >= len(body)
                        await asyncio.wait_for(send({**message,'body':body[offset:offset+16384],
                            'more_body':message.get('more_body',False) if last else True}),max(0,end-time.monotonic()))
                else:
                    await asyncio.wait_for(send(message),remaining)
                if message.get('type') == 'http.response.start': record['response_started'] = True
                if message.get('type') == 'http.response.body' and not message.get('more_body',False): record['response_completed'] = True
            watcher = asyncio.create_task(watch_disconnect())
            try:
                if self.inflight >= self.capacity:
                    stage('admission'); error('capacity')
                    await send_bounded({'type':'http.response.start','status':429,'headers':[(b'content-type',b'application/json'),(b'retry-after',b'1')]})
                    await send_bounded({'type':'http.response.body','body':b'{"detail":{"reason":"memory service busy; retry with backoff"}}'})
                    return
                self.inflight += 1; admitted = True
                stage('queue')
                queued = time.monotonic()
                ticket = object(); self.waiters.append(ticket)
                # Poll peer only while queued, without consuming a potentially large body.
                transport_obj = getattr(transport, 'transport', transport)
                while self.execution.locked() or self.waiters[0] is not ticket:
                    if transport_obj is not None and hasattr(transport_obj,'is_closing') and transport_obj.is_closing():
                        cancelled.set(); error('cancelled'); return
                    if time.monotonic() >= end: raise asyncio.TimeoutError()
                    await asyncio.sleep(min(.02,max(0,end-time.monotonic())))
                await self.execution.acquire(); executing = True
                self.waiters.popleft()
                record['queue_ms'] = round((time.monotonic()-queued)*1000,3)
                stage('parse_validation')
                await asyncio.wait_for(self.app(scope,receive_bounded,send_bounded),
                                       max(0,end-time.monotonic()))
                await drain(transport,end)
            except asyncio.TimeoutError:
                cancelled.set(); error('deadline')
                if headers_sent:
                    # Status cannot be replaced once sent. Abort, never append a second response.
                    if transport is not None: abort(transport)
                    raise RuntimeError('memory response deadline exceeded') from None
                body = json.dumps({'detail':{'reason':'memory request deadline exceeded; retry with backoff'}}).encode()
                # Bounded final error attempt; a disconnected peer may receive no response.
                try:
                    record['status_attempted'] = 504
                    await asyncio.wait_for(send({'type':'http.response.start','status':504,
                        'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]}),1)
                    record['response_started'] = True
                    await asyncio.wait_for(send({'type':'http.response.body','body':body}),1)
                    record['response_completed'] = True
                except (asyncio.TimeoutError, OSError): pass
            except asyncio.CancelledError:
                cancelled.set(); error('cancelled')
                raise
            except BaseException:
                error('send_failed' if headers_sent else 'internal')
                raise
            finally:
                if admitted and not executing: self.waiters.remove(ticket)
                if record['stage'] == 'queue' and admitted: record['queue_ms'] = round((time.monotonic()-queued)*1000,3)
                if cancelled.is_set() and record['code'] == 'http_error': error('cancelled')
                if executing:
                    # Sync work observes cancellation and owns the execution lease until it ends.
                    # wait_for cancellation abandons the thread in Starlette; retain gate until
                    # its service finally signals completion below.
                    completion = record.get('_completion')
                    if completion is not None and not completion.is_set():
                        async def release_after():
                            while not completion.is_set(): await asyncio.sleep(.02)
                            self.execution.release()
                            self.inflight -= 1
                        asyncio.create_task(release_after())
                        admitted = False
                    else: self.execution.release()
                if admitted: self.inflight -= 1
                if self.sink is not None and (record['status_attempted'] is None or record['status_attempted'] >= 400 or cancelled.is_set() or not record['response_completed'] or record['code'] in {'send_failed','deadline','cancelled'}):
                    record['elapsed_ms'] = round((time.monotonic()-start)*1000,3)
                    try: self.sink.emit(record)
                    except Exception: pass
                active.reset(diagnostic_token)
                watcher.cancel()
                with suppress(asyncio.CancelledError): await watcher
                request_started.reset(token)
                request_deadline.reset(dt)
                request_cancelled.reset(ct)
        else:
            await self.app(scope, receive, send)
