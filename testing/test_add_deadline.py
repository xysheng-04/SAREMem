import asyncio
import fcntl
import hashlib
import json
import multiprocessing
import os
import sqlite3
import socket
import subprocess
import sys
import time
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from aml_api.config import Settings
from aml_api.errors import Conflict, ServiceError, UpstreamUnavailable
from aml_api.schemas import AddRequest
from aml_api.service import RouteAService
from aml_api.store import SQLiteStore,payload_digest
from aml_api.production.add_control import add,OwnedStore,PreparedStore,ComputeRunner,worker
from aml_api.production.http_errors import AddDeadlineExceeded,CapacityExceeded
from aml_api.production.policy import Policy
from aml_api.production.request_clock import SearchRequestClock,request_started,request_deadline,request_cancelled
from aml_api.production.service import ProductionService
from faults import FakeEncoder,FakeVLM
from server import build_app


def fake_compute(settings,request,policy,deadline):
    marker=settings.storage_path.with_suffix('.computations')
    with marker.open('a') as f:f.write('1\n')
    # No database access is permitted inside the computation child.
    sqlite3.connect=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('child touched database'))
    store=PreparedStore();service=RouteAService(settings,store=store,embedder=FakeEncoder(),vlm=FakeVLM())
    service.add(request)
    return {**store.value,'input_digest':payload_digest(request.model_dump(mode='json'))}


def sleeping(settings,request,policy,deadline):
    time.sleep(20)


def descendants(settings,request,policy,deadline):
    settings.storage_path.with_suffix('.workerpid').write_text(str(os.getpid()))
    proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    settings.storage_path.with_suffix('.childpid').write_text(str(proc.pid))
    time.sleep(30)


def lost_parent(settings,request,policy):
    parent,child=socket.socketpair()
    proc=multiprocessing.get_context('spawn').Process(target=worker,args=(child,settings,request,policy,time.monotonic()+30,descendants,os.getpid()))
    proc.start();child.close()
    end=time.monotonic()+8
    while not settings.storage_path.with_suffix('.childpid').exists() and time.monotonic()<end:time.sleep(.02)
    os._exit(0)


@pytest.fixture
def data(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-only')
    settings=replace(Settings.from_env(),storage_path=tmp_path/'new.sqlite3',memory_system_key='offline-key-0000000000000000')
    SQLiteStore(settings.storage_path)
    request=AddRequest(request_id='r',user_id='u',session_id='s',messages=[{'role':'user','timestamp':1700000000000,'content':[{'type':'text','text':'synthetic original'}]}])
    policy=Policy(total_seconds=8,simple_seconds=3,legacy_seconds=1,seed_reserve_seconds=1,cleanup_seconds=.2,worker_memory_mb=1024,max_concurrent_searches=1)
    return settings,request,policy


def rows(settings,table):
    with sqlite3.connect(settings.storage_path) as db:return db.execute('select * from '+table).fetchall()


def test_original_algorithm_and_cached_replay_no_compute(data):
    settings,request,policy=data
    first=add(settings,request,policy,ComputeRunner(fake_compute))
    assert add(settings,request,policy,ComputeRunner(sleeping))==first
    assert settings.storage_path.with_suffix('.computations').read_text()=='1\n'
    assert len(rows(settings,'memories'))==1 and rows(settings,'add_requests')[0][2]=='complete'
    expected=replace(settings,storage_path=settings.storage_path.with_name('comparison.sqlite3'))
    legacy=RouteAService(expected,embedder=FakeEncoder(),vlm=FakeVLM());legacy.add(request)
    assert rows(settings,'memories')[0]==rows(expected,'memories')[0]
    assert rows(settings,'views')==rows(expected,'views')


def test_different_body_conflict_without_model(data):
    settings,request,policy=data;add(settings,request,policy,ComputeRunner(fake_compute))
    other=request.model_copy(update={'user_id':'other'})
    with pytest.raises(Conflict):add(settings,other,policy,ComputeRunner(sleeping))


def test_compute_timeout_reaps_and_releases_claim(data):
    settings,request,policy=data;policy=replace(policy,total_seconds=1.3,simple_seconds=.4,legacy_seconds=.2,seed_reserve_seconds=.1,cleanup_seconds=.1)
    before=time.monotonic()
    with pytest.raises(AddDeadlineExceeded):add(settings,request,policy,ComputeRunner(sleeping))
    assert time.monotonic()-before<3 and not multiprocessing.active_children()
    assert not rows(settings,'memories') and not rows(settings,'add_requests')
    assert add(settings,request,replace(policy,total_seconds=8),ComputeRunner(fake_compute)).success


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux process group status verification')
def test_compute_timeout_stops_descendants(data):
    settings,request,policy=data
    with pytest.raises(AddDeadlineExceeded):add(settings,request,policy,ComputeRunner(descendants))
    pid=int(settings.storage_path.with_suffix('.childpid').read_text())
    status=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
    assert not status or status.startswith('Z')
    assert not rows(settings,'memories')


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux parent-loss process status verification')
def test_parent_death_stops_computation_and_descendants(data):
    settings,request,policy=data
    broker=multiprocessing.get_context('spawn').Process(target=lost_parent,args=(settings,request,policy));broker.start();broker.join(12)
    assert not broker.is_alive();broker.close()
    pids=[int(settings.storage_path.with_suffix(s).read_text()) for s in ['.workerpid','.childpid']]
    end=time.monotonic()+5
    while True:
        states=[subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip() for pid in pids]
        if all(not s or s.startswith('Z') for s in states):break
        assert time.monotonic()<end
        time.sleep(.05)
    assert not rows(settings,'memories')


def prepared_here(settings,request):
    store=PreparedStore();RouteAService(settings,store=store,embedder=FakeEncoder(),vlm=FakeVLM()).add(request)
    return {**store.value,'input_digest':payload_digest(request.model_dump(mode='json'))}


def test_transaction_rollback_no_partial_memory_or_views(data):
    settings,request,policy=data
    class Broken:
        def run(self,*args):
            value=prepared_here(settings,request);value['views'][-1]['embedding']=[float('nan')];return value
    with pytest.raises(ValueError):add(settings,request,policy,Broken())
    assert not rows(settings,'memories') and not rows(settings,'views') and not rows(settings,'add_requests')


def test_old_owner_cannot_commit_or_remove_new_claim(data):
    settings,request,policy=data
    class Replaced:
        def run(self,*args):
            value=prepared_here(settings,request)
            with sqlite3.connect(settings.storage_path) as db:db.execute('update add_requests set updated_at=updated_at+10')
            return value
    with pytest.raises(Conflict):add(settings,request,policy,Replaced())
    assert not rows(settings,'memories') and len(rows(settings,'add_requests'))==1


def test_stale_claim_recovered_under_kernel_ownership(data):
    settings,request,policy=data
    with sqlite3.connect(settings.storage_path) as db:
        db.execute("insert into add_requests values (?,?,'processing',NULL,?)",(request.request_id,payload_digest(request.model_dump(mode='json')),time.time()-2000))
    assert add(settings,request,policy,ComputeRunner(fake_compute)).success
    assert len(rows(settings,'memories'))==1 and rows(settings,'add_requests')[0][2]=='complete'


def test_recent_supervised_orphan_claim_recovers_without_waiting_1800(data):
    settings,request,policy=data
    store=OwnedStore(settings,time.monotonic()+5,None)
    digest=payload_digest(request.model_dump(mode='json'));store.claim(request.request_id,digest)
    # No live kernel owner remains, simulating a crashed supervised parent.
    assert add(settings,request,policy,ComputeRunner(fake_compute)).success
    assert not store.marker(request.request_id).exists()
    assert len(rows(settings,'memories'))==1


def test_cancelled_computation_does_not_commit(data):
    settings,request,policy=data;event=threading.Event();token=request_cancelled.set(event)
    timer=threading.Timer(.2,event.set);timer.start();before=time.monotonic()
    try:
        with pytest.raises(AddDeadlineExceeded):add(settings,request,policy,ComputeRunner(sleeping))
    finally:request_cancelled.reset(token);timer.join()
    assert time.monotonic()-before<2 and not rows(settings,'memories') and not rows(settings,'add_requests')


def test_prepared_scope_mismatch_rolls_back(data):
    settings,request,policy=data
    class Wrong:
        def run(self,*args):
            value=prepared_here(settings,request);value['memory']['user_id']='other';return value
    with pytest.raises(ValueError):add(settings,request,policy,Wrong())
    assert not rows(settings,'memories') and not rows(settings,'add_requests')


def test_lost_success_after_commit_recovers_cache(data,monkeypatch):
    settings,request,policy=data;original=OwnedStore.complete_owned
    def committed_then_lost(self,*args):original(self,*args);raise OSError('synthetic loss after commit')
    monkeypatch.setattr(OwnedStore,'complete_owned',committed_then_lost)
    assert add(settings,request,policy,ComputeRunner(fake_compute)).success
    assert len(rows(settings,'memories'))==1
    assert add(settings,request,policy,ComputeRunner(sleeping)).success
    assert settings.storage_path.with_suffix('.computations').read_text()=='1\n'


def test_capacity_and_duplicate_processing(data):
    settings,request,policy=data
    name='.add-'+hashlib.sha256(str(settings.storage_path.resolve()).encode()).hexdigest()[:16]+'.lock'
    with (settings.storage_path.parent/name).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with pytest.raises(CapacityExceeded):add(settings,request,policy,ComputeRunner(sleeping))
        store=OwnedStore(settings,time.monotonic()+5,None);store.claim(request.request_id,payload_digest(request.model_dump(mode='json')))
        with pytest.raises(Conflict):add(settings,request,policy,ComputeRunner(sleeping))


def test_database_lock_wait_is_bounded(data):
    settings,request,policy=data;policy=replace(policy,total_seconds=1.3,simple_seconds=.4,legacy_seconds=.2,seed_reserve_seconds=.1,cleanup_seconds=.1)
    db=sqlite3.connect(settings.storage_path);db.execute('BEGIN IMMEDIATE')
    try:
        before=time.monotonic()
        with pytest.raises(UpstreamUnavailable):add(settings,request,policy,ComputeRunner(sleeping))
        assert time.monotonic()-before<2
    finally:db.rollback();db.close()


def test_http_add_success_and_contract(data):
    settings,request,policy=data
    service=ProductionService(settings,policy=policy,add_runner=ComputeRunner(fake_compute))
    with TestClient(build_app(settings=settings,service=service)) as c:
        response=c.post('/add',headers={'X-Api-Key':settings.memory_system_key},json=request.model_dump(mode='json'))
        assert response.status_code==200 and set(response.json())=={'success','request_id','user_id','session_id'}
        assert c.post('/add',headers={'X-Api-Key':settings.memory_system_key},json=request.model_dump(mode='json')).json()==response.json()


def test_http_add_deadline_is_504_without_partial_commit(data):
    settings,request,policy=data;policy=replace(policy,total_seconds=1.3,simple_seconds=.4,legacy_seconds=.2,seed_reserve_seconds=.1,cleanup_seconds=.1)
    service=ProductionService(settings,policy=policy,add_runner=ComputeRunner(sleeping))
    with TestClient(build_app(settings=settings,service=service)) as c:
        response=c.post('/add',headers={'X-Api-Key':settings.memory_system_key},json=request.model_dump(mode='json'))
        assert response.status_code==504 and set(response.json())=={'detail'}
    assert not rows(settings,'memories') and not rows(settings,'add_requests')


@pytest.mark.parametrize('path',['/add','/search'])
def test_slow_upload_and_absolute_deadline(path):
    messages=[];events=[]
    async def receive():await asyncio.sleep(1);return {'type':'http.request','body':b'{}'}
    async def send(message):messages.append(message)
    async def app(scope,receive,send):
        events.append(request_cancelled.get());assert request_started.get() and request_deadline.get()
        await receive()
    policy=Policy(total_seconds=.15,simple_seconds=.04,legacy_seconds=.02,seed_reserve_seconds=.02,cleanup_seconds=.01)
    before=time.monotonic();asyncio.run(SearchRequestClock(app,policy)({'type':'http','method':'POST','path':path},receive,send))
    assert time.monotonic()-before<.6 and messages[0]['status']==504 and events[0].is_set()
    assert request_started.get() is None and request_deadline.get() is None


def test_slow_response_abort_without_second_http_response():
    starts=[]
    async def app(scope,receive,send):
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'{"data":[]}'})
    async def send(message):
        if message['type']=='http.response.start':starts.append(message['status'])
        else:await asyncio.sleep(1)
    async def receive():return {'type':'http.request','body':b''}
    policy=Policy(total_seconds=.15,simple_seconds=.04,legacy_seconds=.02,seed_reserve_seconds=.02,cleanup_seconds=.01)
    with pytest.raises(RuntimeError):asyncio.run(SearchRequestClock(app,policy)({'type':'http','method':'POST','path':'/search'},receive,send))
    assert starts==[200]


def test_disconnect_after_body_cancels_add_and_reaps(data):
    settings,request,policy=data;responses=[];calls=0
    async def receive():
        nonlocal calls
        calls+=1
        if calls==1:return {'type':'http.request','body':b'{}','more_body':False}
        await asyncio.sleep(.2)
        return {'type':'http.disconnect'}
    async def send(message):responses.append(message)
    async def app(scope,receive,send):
        await receive()
        with pytest.raises(AddDeadlineExceeded):
            await asyncio.to_thread(add,settings,request,policy,ComputeRunner(sleeping))
    before=time.monotonic()
    asyncio.run(SearchRequestClock(app,policy)({'type':'http','method':'POST','path':'/add'},receive,send))
    assert time.monotonic()-before<2 and not rows(settings,'memories') and not rows(settings,'add_requests')
