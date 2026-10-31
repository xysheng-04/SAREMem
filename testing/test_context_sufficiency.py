import dataclasses,time
import pytest
from aml_api.config import Settings
from aml_api.schemas import SearchRequest,SearchResponse
from aml_api.production.service import ProductionService
from aml_api.production.policy import Policy
from aml_api.production.http_errors import SearchDeadlineExceeded
from aml_api.errors import UpstreamUnavailable
from aml_api.context_sufficiency import decide,assess,SCHEMA
from aml_api.context_sufficiency.provider import parse
from types import SimpleNamespace

@pytest.mark.parametrize('points',range(101))
def test_threshold(points):
 d=decide({'sufficiency_points':points,'reason_code':'missing_fact'},.20)
 assert d['refuse']==(points<20)
 assert 'supported_evidence_ids' not in d

@pytest.mark.parametrize('v',[True,-1,101,15.,float('nan')])
def test_invalid(v):
 with pytest.raises(ValueError):decide({'sufficiency_points':v,'reason_code':'sufficient'},.20)

def test_no_citations():
 assert set(SCHEMA['properties'])=={'sufficiency_points','reason_code'}
 assert not decide({'sufficiency_points':100,'reason_code':'sufficient'},.20)['refuse']
 assert parse('```json\n{"sufficiency_points":100,"reason_code":"sufficient"}\n```')['sufficiency_points']==100

def test_mock_provider():
 req=SearchRequest.model_validate({'user_id':'u','top_k':20,'query':[{'type':'text','text':'Q'}]})
 res=SearchResponse.model_validate({'data':[{'id':'m','content':[{'type':'text','text':'E'}],'score':1}]})
 class Fake:
  def __init__(self,*a):pass
  def close(self):pass
  def call(self,stage,payload,schema):
   assert set(schema['properties'])=={'sufficiency_points','reason_code'}
   return {'sufficiency_points':100,'reason_code':'sufficient'}
 assert not assess(SimpleNamespace(return_limit=20),req,res,Policy(),.20,Fake)['refuse']

class Runner:
 def __init__(self,kind):self.kind=kind;self.calls=[];self.deadlines=[]
 def run(self,stage,settings,request,policy,deadline,checkpoint,publish):
  self.calls.append(stage);self.deadlines.append(deadline)
  data={'data':[{'id':'m','content':[{'type':'text','text':'E'}],'score':1}]}
  if stage=='sufficiency':
   if self.kind=='error':return {'error':'schema'}
   if self.kind=='timeout':return {'error':'deadline'}
   if self.kind=='low':return {'response':{'data':[]}}
   if self.kind=='mutate':data['data'][0]['content'][0]['text']='changed'
  return {'response':data}

@pytest.mark.parametrize('mode,kind,expected',[('off','low',1),('on','low',0),('on','high',1)])
def test_service(mode,kind,expected,monkeypatch,tmp_path):
 monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('AML_SUFFICIENCY_MODE',mode)
 s=dataclasses.replace(Settings.from_env(),storage_path=tmp_path/'unused.sqlite3')
 runner=Runner(kind);svc=ProductionService(s,runner=runner)
 req=SearchRequest.model_validate({'user_id':'u','top_k':20,'query':[{'type':'text','text':'Q'}]})
 begin=time.monotonic();r=svc.search(req)
 assert len(r.data)==expected
 assert runner.calls==(['simple'] if mode=='off' else ['simple','sufficiency'])
 if mode=='on':assert runner.deadlines[0]<=begin+1066 and runner.deadlines[1]<=begin+1680

@pytest.mark.parametrize('kind,exception',[('error',UpstreamUnavailable),('timeout',SearchDeadlineExceeded),('mutate',UpstreamUnavailable)])
def test_gate_error_never_fallback(kind,exception,monkeypatch,tmp_path):
 monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('AML_SUFFICIENCY_MODE','on')
 svc=ProductionService(dataclasses.replace(Settings.from_env(),storage_path=tmp_path/'unused.sqlite3'),runner=Runner(kind))
 req=SearchRequest.model_validate({'user_id':'u','top_k':20,'query':[{'type':'text','text':'Q'}]})
 with pytest.raises(exception):svc.search(req)
 assert svc.runner.calls==['simple','sufficiency']
 assert svc.slots.acquire(blocking=False);svc.slots.release()

class ScoreProvider:
 points=15
 def __init__(self,*a):pass
 def close(self):pass
 def call(self,stage,payload,schema):
  return {'sufficiency_points':self.points,'reason_code':'sufficient'}

class ComponentBackend:
 def __init__(self,points):self.points=points
 def __call__(self,stage,settings,request,policy,deadline,checkpoint,publish,stats):
  if stage=='simple':return {'data':[{'id':'m','content':[{'type':'text','text':'E'}],'score':1}]}
  assert stage=='sufficiency'
  from aml_api.context_sufficiency import execute
  ScoreProvider.points=self.points
  return execute(settings,request,policy,deadline,checkpoint,stats,ScoreProvider)

@pytest.mark.parametrize('points,expected',[(19,0),(20,1),(100,1)])
def test_real_supervised_component(points,expected,monkeypatch,tmp_path):
 import sqlite3
 from aml_api.production.process import StageRunner
 monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('AML_SUFFICIENCY_MODE','on')
 path=tmp_path/'private.sqlite3';db=sqlite3.connect(path)
 db.execute('CREATE TABLE memories(memory_id TEXT,user_id TEXT,content_json TEXT,created_at INTEGER,source_timestamp_ms INTEGER)')
 db.execute('INSERT INTO memories(memory_id,user_id,content_json,created_at) VALUES(?,?,?,?)',('m','u','[{"type":"text","text":"E"}]',1));db.commit();db.close()
 s=dataclasses.replace(Settings.from_env(),storage_path=path)
 svc=ProductionService(s,runner=StageRunner(ComponentBackend(points)))
 req=SearchRequest.model_validate({'user_id':'u','top_k':20,'query':[{'type':'text','text':'Q'}]})
 assert len(svc.search(req).data)==expected

def test_scope_before_provider(monkeypatch,tmp_path):
 import sqlite3
 from aml_api.context_sufficiency import execute
 from aml_api.production.policy import SearchFault
 monkeypatch.setenv('AML_SUFFICIENCY_THRESHOLD','0.20');monkeypatch.setenv('OPENAI_API_KEY','offline')
 path=tmp_path/'private.sqlite3';db=sqlite3.connect(path)
 db.execute('CREATE TABLE memories(memory_id TEXT,user_id TEXT,content_json TEXT,created_at INTEGER,source_timestamp_ms INTEGER)')
 db.execute('INSERT INTO memories(memory_id,user_id,content_json,created_at) VALUES(?,?,?,?)',('m','other','[{"type":"text","text":"E"}]',1));db.commit();db.close()
 s=dataclasses.replace(Settings.from_env(),storage_path=path)
 req=SearchRequest.model_validate({'user_id':'u','top_k':20,'query':[{'type':'text','text':'Q'}]})
 def forbidden(*a):raise AssertionError('provider should never be created')
 with pytest.raises(SearchFault):execute(s,req,Policy(),time.monotonic()+90,{'data':[{'id':'m','content':[{'type':'text','text':'E'}],'score':1}]},{'dropped':0},forbidden)
