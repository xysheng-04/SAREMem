"""Pinned Uvicorn H11 adapter exposing only a private transport capability."""
import array
import asyncio
import fcntl
import os
import socket
import struct
import sys
import time

from uvicorn.protocols.http.h11_impl import H11Protocol

CAPABILITY = 'saremem.transport'
ARRIVED = 'saremem.arrived'


class DeadlineH11Protocol(H11Protocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._header_since = None
        self._header_timer = None
        application = self.app
        async def bound(scope, receive, send):
            if scope.get('type') == 'http' and scope.get('path','').rstrip('/') in {'/add','/search'}:
                scope = {**scope, 'extensions':{**scope.get('extensions',{}),CAPABILITY:self.transport,
                    ARRIVED:self._header_since or time.monotonic()}}
            self._header_since = None
            if self._header_timer is not None:
                self._header_timer.cancel(); self._header_timer = None
            await application(scope,receive,send)
        self.app = bound

    def data_received(self, data):
        if self._header_since is None and (self.cycle is None or self.cycle.response_complete):
            self._header_since = time.monotonic()
            seconds = float(os.environ.get('AML_PROD_TOTAL_SECONDS','1680'))
            self._header_timer = self.loop.call_later(seconds,abort,self.transport)
        super().data_received(data)

    def connection_lost(self, exc):
        if self._header_timer is not None: self._header_timer.cancel()
        super().connection_lost(exc)


def pending_bytes(transport):
    queued = transport.get_write_buffer_size()
    sock = transport.get_extra_info('socket')
    if sock is not None and sock.fileno() >= 0 and sys.platform.startswith('linux'):
        try:
            # Linux SIOCOUTQ includes TCP bytes still awaiting acknowledgement.
            output = array.array('i',[0])
            fcntl.ioctl(sock.fileno(),0x5411,output,True)
            queued += max(0,output[0])
        except OSError:
            # The peer or socket may already have closed. Never inspect payloads.
            if not transport.is_closing(): raise
    return queued


def abort(transport):
    # Drop unsent TCP bytes on expiry; do not leave a large response flushing
    # after the application has already enforced its final deadline.
    sock = transport.get_extra_info('socket')
    if sock is not None:
        try: sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
        except OSError: pass
    transport.abort()


async def drain(transport, deadline):
    if transport is None: return  # ASGI unit clients have no physical TCP transport.
    while pending_bytes(transport):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            abort(transport)
            raise asyncio.TimeoutError()
        await asyncio.sleep(min(.02,remaining))
