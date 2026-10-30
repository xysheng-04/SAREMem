import asyncio
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from aml_api.production.http_transport import drain,CAPABILITY
from aml_api.production.request_clock import SearchRequestClock
from aml_api.production.policy import Policy


def test_pending_transport_bytes_are_deadline_bounded():
    class Transport:
        aborted=False
        def get_extra_info(self,name):return None
        def get_write_buffer_size(self):return 1000
        def abort(self):self.aborted=True
    transport=Transport()
    with pytest.raises(asyncio.TimeoutError):asyncio.run(drain(transport,time.monotonic()+.05))
    assert transport.aborted


def test_chunked_send_preserves_response_bytes_and_order():
    original=b'a'*50000;messages=[]
    async def app(scope,receive,send):
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':original})
    async def send(message):messages.append(message)
    async def receive():return {'type':'http.request','body':b''}
    asyncio.run(SearchRequestClock(app)({'type':'http','method':'POST','path':'/search'},receive,send))
    body=[m for m in messages if m['type']=='http.response.body']
    assert b''.join(m['body'] for m in body)==original
    assert all(len(m['body'])<=16384 for m in body) and all(m['more_body'] for m in body[:-1]) and not body[-1]['more_body']


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux real TCP and output queue verification')
def test_real_tcp_slow_upload_and_response_are_bounded(tmp_path):
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    production=Path(__file__).resolve().parents[1]/'production'
    source=f'''import sys,asyncio
sys.path.insert(0,{str(production)!r})
from aml_api.production.process import quiet
quiet()
from aml_api.production.request_clock import SearchRequestClock
from aml_api.production.policy import Policy
from aml_api.production.http_transport import DeadlineH11Protocol
import uvicorn
async def application(scope,receive,send):
 if scope['path']=='/health':
  await send({{'type':'http.response.start','status':200,'headers':[]}})
  await send({{'type':'http.response.body','body':b'ok'}});return
 while True:
  message=await receive()
  if not message.get('more_body',False):break
 await send({{'type':'http.response.start','status':200,'headers':[]}})
 body=b'x'*(32*1024*1024) if scope['path']=='/search' else b'{{}}'
 await send({{'type':'http.response.body','body':body}})
policy=Policy(total_seconds=.6,simple_seconds=.1,legacy_seconds=.1,seed_reserve_seconds=.05,cleanup_seconds=.01)
if __name__=='__main__':uvicorn.run(SearchRequestClock(application,policy),host='127.0.0.1',port={port},http=DeadlineH11Protocol,lifespan='off',access_log=False,log_config=None,log_level='critical')
'''
    script=tmp_path/'tcp_deadline_fixture.py';script.write_text(source)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.Popen([sys.executable,'-B',str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(100):
            try:
                with opener.open('http://127.0.0.1:'+str(port)+'/health',timeout=.3) as response:assert response.status==200
                break
            except Exception:time.sleep(.1)
        else:raise AssertionError('TCP fixture did not start')
        with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
            before=time.monotonic();client.sendall(b'POST /add HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{')
            received=b''
            while True:
                part=client.recv(65536)
                if not part:break
                received+=part
            assert b'504 Gateway Timeout' in received and b'memory request deadline exceeded' in received
            assert time.monotonic()-before<2
        with socket.socket() as client:
            client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024);client.settimeout(3);client.connect(('127.0.0.1',port))
            client.sendall(b'POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}')
            time.sleep(1)
            received=0
            while True:
                try:part=client.recv(65536)
                except ConnectionResetError:break
                if not part:break
                received+=len(part)
            assert received<32*1024*1024  # Slow peer sees a terminated response, never a fabricated second 504.
    finally:
        proc.terminate();proc.wait(timeout=5)
        stdout,stderr=proc.communicate()
        assert not stdout and not stderr
