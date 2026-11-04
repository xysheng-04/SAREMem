import asyncio, json, os, threading, time
from pathlib import Path
from dataclasses import replace
import pytest
from aml_api.production.request_clock import SearchRequestClock, request_deadline
from aml_api.production.diagnostics import FailureSink, active, stage, safe_failure
from aml_api.production.policy import Policy

class Sink:
    def __init__(self): self.records=[]
    def emit(self,record): self.records.append(dict(record))

def policy(seconds=10):
    return Policy(total_seconds=seconds,simple_seconds=.1,legacy_seconds=.1,seed_reserve_seconds=.05,cleanup_seconds=.01)

async def call(clock,path='/add',send_override=None,transport=None):
    output=[]; body=False
    async def receive():
        nonlocal body
        if not body: body=True;return {'type':'http.request','body':b'{}'}
        await asyncio.Future()
    async def send(m):
        if send_override: await send_override(m)
        output.append(m)
    scope={'type':'http','method':'POST','path':path}
    if transport is not None: scope['extensions']={'saremem.transport':transport}
    await clock(scope,receive,send)
    return output

@pytest.mark.parametrize('paths',[['/add']*16,['/search']*16,['/add']*16+['/search']*16])
def test_admits_platform_concurrency_without_busy(paths):
    async def run():
        current=peak=0; ends=[]
        async def app(scope,receive,send):
            nonlocal current,peak
            current+=1;peak=max(current,peak);ends.append(request_deadline.get())
            await receive();await asyncio.sleep(.01)
            await send({'type':'http.response.start','status':200})
            await send({'type':'http.response.body','body':b'{}'})
            current-=1
        clock=SearchRequestClock(app,policy());clock.sink=Sink()
        output=await asyncio.gather(*(call(clock,p) for p in paths))
        assert all(v[0]['status']==200 for v in output)
        assert peak==len(set(paths)) and clock.inflight==0 and not clock.execution.locked()
        assert max(ends)-min(ends)<.05 and not clock.sink.records
    asyncio.run(run())

def test_capacity_and_queue_deadline():
    async def run():
        async def app(scope,receive,send): await asyncio.sleep(10)
        clock=SearchRequestClock(app,policy(.3));clock.sink=Sink()
        out=await asyncio.gather(*(call(clock) for _ in range(33)))
        statuses=[v[0]['status'] for v in out]
        assert statuses.count(429)==1 and statuses.count(504)==32
        assert any(r['stage']=='queue' and r['queue_ms']>0 for r in clock.sink.records)
        assert not clock.execution.locked()
    asyncio.run(run())

def test_cancelled_worker_holds_gate_until_completion():
    async def run():
        complete=threading.Event();entered=asyncio.Event()
        async def app(scope,receive,send):
            active.get()['_completion']=complete;entered.set();await asyncio.Future()
        clock=SearchRequestClock(app,policy());clock.sink=Sink()
        task=asyncio.create_task(call(clock));await entered.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert clock.execution.locked()
        complete.set();await asyncio.sleep(.05)
        assert not clock.execution.locked()
    asyncio.run(run())

def test_response_send_failure_does_not_claim_delivery():
    async def run():
        async def app(scope,receive,send):
            await send({'type':'http.response.start','status':200})
            await send({'type':'http.response.body','body':b'private'})
        async def send(m):
            if m['type']=='http.response.body': raise OSError('SECRET')
        clock=SearchRequestClock(app,policy());clock.sink=Sink()
        with pytest.raises(OSError):await call(clock,send_override=send)
        r=clock.sink.records[0]
        assert r['status_attempted']==200 and r['delivery']=='unconfirmed'
        assert r['stage']=='response_send' and r['code']=='send_failed' and not r['response_completed']
        assert 'SECRET' not in json.dumps(r)
    asyncio.run(run())

def test_sink_rotation_and_failure_best_effort(tmp_path):
    sink=FailureSink(tmp_path/'private',maximum=512,backups=3)
    for i in range(80): sink.emit({'correlation':str(i),'operation':'add','stage':'provider','code':'provider','raw':'SECRET'})
    sink.items.join()
    files=list((tmp_path/'private').iterdir())
    assert len(files)<=4
    assert (tmp_path/'private').stat().st_mode&0o777==0o700
    for p in files:
        assert p.stat().st_size<=512 and p.stat().st_mode&0o777==0o600
        assert 'SECRET' not in p.read_text()
    sink._write=lambda raw: (_ for _ in ()).throw(OSError('SECRET'))
    sink.emit({'stage':'provider'});sink.items.join()
    assert sink.thread.is_alive()

def test_failed_sink_cannot_fail_request():
    async def run():
        async def app(scope,receive,send):
            await send({'type':'http.response.start','status':503})
            await send({'type':'http.response.body','body':b'{}'})
        clock=SearchRequestClock(app,policy());clock.sink=FailureSink('/dev/null/invalid')
        result=await call(clock)
        assert result[0]['status']==503
        clock.sink.items.join()
    asyncio.run(run())

@pytest.mark.parametrize('phase', ['parse_validation','authentication','admission','queue','image_processing','provider','compute_process','database_claim','database_commit','response_validation'])
def test_each_failure_stage_safe(phase):
    async def run():
        async def app(scope,receive,send):
            stage(phase)
            await send({'type':'http.response.start','status':503})
            await send({'type':'http.response.body','body':b'{}'})
        clock=SearchRequestClock(app,policy());clock.sink=Sink()
        await call(clock)
        assert clock.sink.records[0]['stage']==phase
        assert clock.sink.records[0]['status_attempted']==503
        assert len(clock.sink.records[0]['correlation'])==32
    asyncio.run(run())

def test_queued_disconnect_does_not_compute():
    async def run():
        count=0
        async def app(scope,receive,send):
            nonlocal count
            count+=1;await asyncio.sleep(.08)
            await send({'type':'http.response.start','status':200});await send({'type':'http.response.body','body':b'{}'})
        class Transport:
            def is_closing(self): return True
        clock=SearchRequestClock(app,policy());clock.sink=Sink()
        task=asyncio.create_task(call(clock));await asyncio.sleep(.01)
        assert await call(clock,transport=Transport())==[]
        await task
        assert count==1 and clock.sink.records[0]['code']=='cancelled'
    asyncio.run(run())

def test_real_http_adapter_add16_search16_database_and_replay(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY","offline-only")
    import httpx, sqlite3
    from aml_api.config import Settings
    from aml_api.store import SQLiteStore
    from aml_api.production.service import ProductionService
    from aml_api.production.add_control import ComputeRunner
    from server import build_app
    from test_add_deadline import fake_compute
    settings=replace(Settings.from_env(),storage_path=tmp_path/'independent.sqlite3',memory_system_key='offline-key-0000000000000000')
    SQLiteStore(settings.storage_path)
    class Search:
        def run(self,stage,settings,request,policy,deadline,checkpoint,publish):
            with sqlite3.connect(settings.storage_path) as db:
                # No visible committed memory can lack all its atomically inserted views.
                assert db.execute('SELECT count(*) FROM memories m WHERE NOT EXISTS (SELECT 1 FROM views v WHERE v.memory_id=m.memory_id)').fetchone()[0]==0
            return {'response':{'data':[]},'stats':{}}
    service=ProductionService(settings,policy=replace(policy(90),max_concurrent_searches=1,worker_memory_mb=1024),runner=Search(),add_runner=ComputeRunner(fake_compute))
    app=build_app(settings=settings,service=service)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://offline',headers={'X-Api-Key':settings.memory_system_key},timeout=90) as client:
            def payload(i):
                return {'request_id':str(i),'user_id':'u','session_id':'s','messages':[{'role':'user','timestamp':1700000000000,'content':[{'type':'text','text':'synthetic'}]}]}
            result=await asyncio.gather(*[client.post('/add',json=payload(i)) for i in range(16)],*[client.post('/search',json={'user_id':'u','query':[{'type':'text','text':'synthetic'}],'top_k':100}) for i in range(16)])
            assert [v.status_code for v in result]==[200]*32
            replay=await asyncio.gather(*[client.post('/add',json=payload(0)) for _ in range(16)])
            assert all(v.status_code==200 for v in replay)
            different=payload(0);different['user_id']='other'
            assert (await client.post('/add',json=different)).status_code==409
            different=payload(0);different['messages'][0]['content'][0]['text']='different'
            assert (await client.post('/add',json=different)).status_code==409
    asyncio.run(run())
    assert len(settings.storage_path.with_suffix('.computations').read_text().splitlines())==16
    with sqlite3.connect(settings.storage_path) as db:
        assert db.execute('SELECT count(*) FROM memories').fetchone()[0]==16
        assert db.execute("SELECT count(*) FROM add_requests WHERE state='complete'").fetchone()[0]==16
        assert not db.execute('PRAGMA foreign_key_check').fetchall()

def provider_failure(settings,request,policy,deadline):
    from aml_api.errors import UpstreamUnavailable
    raise UpstreamUnavailable('SECRET provider body')

def child_exit(settings,request,policy,deadline):
    os._exit(2)

@pytest.mark.parametrize('backend,code,phase',[(provider_failure,'provider','provider'),(child_exit,'worker_exit','compute_process')])
def test_real_child_safe_failure_reaches_parent(backend,code,phase):
    from aml_api.production.add_control import ComputeRunner
    from aml_api.errors import ServiceError
    record={'stage':'compute_process','code':'http_error'}
    token=active.set(record)
    try:
        with pytest.raises(ServiceError) as exc:
            ComputeRunner(backend).run(None,None,policy(),time.monotonic()+8)
        assert record['stage']==phase and record['code']==code
        assert 'SECRET' not in str(exc.value)
    finally: active.reset(token)

def test_invalid_search_image_is_image_stage():
    from aml_api.production.service import ProductionService
    from aml_api.schemas import SearchRequest
    from aml_api.errors import BadRequest
    record={'stage':'authentication'};token=active.set(record)
    try:
        request=SearchRequest(user_id='u',top_k=100,query=[{'type':'image_url','image_url':{'url':'invalid'}}])
        with pytest.raises(BadRequest):ProductionService.__new__(ProductionService).search(request)
        assert record['stage']=='image_processing'
    finally:active.reset(token)
