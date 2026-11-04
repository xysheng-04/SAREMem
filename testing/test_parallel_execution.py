import asyncio,threading,time,sqlite3,dataclasses
from concurrent.futures import ThreadPoolExecutor
from aml_api.config import Settings
from aml_api.schemas import AddRequest
from aml_api.store import SQLiteStore,payload_digest
from aml_api.service import RouteAService
from aml_api.production.add_control import PreparedStore
from aml_api.production.service import ProductionService
from aml_api.errors import Conflict
from faults import FakeEncoder,FakeVLM
from test_failure_queue import call,policy
from aml_api.production.request_clock import SearchRequestClock


def test_add16_concurrent_atomic_and_replay(tmp_path,monkeypatch):
 monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('AML_PROD_MAX_CONCURRENT_ADDS','16')
 s=dataclasses.replace(Settings.from_env(),storage_path=tmp_path/'memory.sqlite3');SQLiteStore(s.storage_path)
 barrier=threading.Barrier(16);counter=[]
 class Runner:
  def run(self,settings,request,p,deadline,cancelled):
   counter.append(request.request_id);barrier.wait(timeout=15)
   store=PreparedStore();RouteAService(settings,store=store,embedder=FakeEncoder(),vlm=FakeVLM()).add(request)
   return {**store.value,'input_digest':payload_digest(request.model_dump(mode='json'))}
 svc=ProductionService(s,add_runner=Runner())
 reqs=[AddRequest(request_id='r'+str(i),user_id='u'+str(i),session_id='s',messages=[{'role':'user','content':'synthetic','timestamp':1700000000000+i}]) for i in range(16)]
 with ThreadPoolExecutor(max_workers=16) as pool:assert all(x.success for x in pool.map(svc.add,reqs))
 assert len(counter)==16
 for req in reqs:assert svc.add(req).success
 assert len(counter)==16
 try:svc.add(reqs[0].model_copy(update={'user_id':'other'}))
 except Conflict:pass
 else:raise AssertionError('conflicting duplicate accepted')
 db=sqlite3.connect(s.storage_path)
 assert db.execute('SELECT count(*) FROM memories').fetchone()[0]==16
 assert db.execute("SELECT count(*) FROM add_requests WHERE state='complete'").fetchone()[0]==16
 assert not db.execute('PRAGMA foreign_key_check').fetchall()


def test_http_independent_add16_search8(monkeypatch):
 monkeypatch.setenv('AML_PROD_MAX_CONCURRENT_ADDS','16');monkeypatch.setenv('AML_PROD_MAX_CONCURRENT_SEARCHES','8')
 async def run():
  active={'/add':0,'/search':0};peak=dict(active);entered=asyncio.Event();release=asyncio.Event()
  async def app(scope,receive,send):
   p=scope['path'];active[p]+=1;peak[p]=max(peak[p],active[p])
   if sum(active.values())==24:entered.set()
   await receive();await release.wait()
   await send({'type':'http.response.start','status':200,'headers':[]});await send({'type':'http.response.body','body':b'{}'})
   active[p]-=1
  clock=SearchRequestClock(app,policy(10))
  tasks=[asyncio.create_task(call(clock,path)) for path,n in [('/add',16),('/search',8)] for _ in range(n)]
  await asyncio.wait_for(entered.wait(),3);assert peak=={'/add':16,'/search':8}
  release.set();await asyncio.gather(*tasks);assert clock.inflight==0
 asyncio.run(run())
