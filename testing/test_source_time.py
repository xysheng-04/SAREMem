import dataclasses,json,sqlite3
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from aml_api.schemas import SearchResponse,SearchItem,SearchRequest
from aml_api.source_time import source_created_at
from aml_api.simple.metadata import trusted_metadata
from aml_api.production.validation import finalize,seed_response
from aml_api.production.policy import Policy
from aml_api.postseed.seed_port import digest,content_digest
from aml_api.config import Settings

@pytest.mark.parametrize('timestamp,expected',[(None,None),('123',None),(True,None),(-1,None),(1.5,None),(253402300800000,None),(0,'1970-01-01T00:00:00Z'),(1700000000123,'2023-11-14T22:13:20.123000Z'),(253402300799999,'9999-12-31T23:59:59.999000Z')])
def test_timestamp(timestamp,expected):
 assert source_created_at(timestamp)==expected

@pytest.mark.parametrize('timestamp',[None,0,1700000000123])
def test_http_and_metadata(timestamp):
 md=trusted_metadata({'source_timestamp_ms':timestamp,'created_at':'2099-01-01T00:00:00Z'})
 assert md['created_at']==source_created_at(timestamp)
 app=FastAPI()
 @app.get('/',response_model=SearchResponse)
 def route():return SearchResponse(data=[SearchItem(id='m',content=[{'type':'text','text':'E'}],created_at=md['created_at'])])
 row=TestClient(app).get('/').json()['data'][0]
 assert ('created_at' in row)==(timestamp is not None)
 if timestamp is not None:assert row['created_at']==source_created_at(timestamp)

@pytest.mark.parametrize('timestamp',[None,0,1700000000123])
def test_projection_and_seed(tmp_path,timestamp,monkeypatch):
 monkeypatch.setenv('OPENAI_API_KEY','offline')
 path=tmp_path/'source.sqlite3';content=[{'type':'text','text':'E'}]
 db=sqlite3.connect(path);db.execute('CREATE TABLE memories(memory_id TEXT,user_id TEXT,content_json TEXT,created_at TEXT,source_timestamp_ms INTEGER)');db.execute('INSERT INTO memories VALUES(?,?,?,?,?)',('m','u',json.dumps(content),'2099-01-01T00:00:00Z',timestamp));db.commit();db.close()
 settings=dataclasses.replace(Settings.from_env(),storage_path=path)
 req=SearchRequest(user_id='u',query='Q',top_k=20);policy=Policy()
 def stats():return {'dropped':0}
 good={'id':'m','content':content,'score':1,'created_at':source_created_at(timestamp)}
 out=finalize(settings,req,policy,{'data':[good]},'simple',stats())
 assert ('created_at' in out['data'][0])==(timestamp is not None)
 # Recovery reads authoritative source milliseconds, never the legacy stored string.
 checkpoint={'request':digest(req.model_dump(mode='json')),'hits':[{'id':'m','hash':content_digest(content),'score':1}]}
 recovered=seed_response(settings,req,policy,checkpoint,stats())
 out2=finalize(settings,req,policy,recovered,'seed',stats())
 assert out==out2
 bad={**good,'created_at':'2099-01-01T00:00:00Z'}
 assert finalize(settings,req,policy,{'data':[bad]},'legacy',stats())=={'data':[]}

@pytest.mark.parametrize('timestamps,expected',[([],None),([None],None),([-1],None),([1700000000123,None,0],0),([1700000000123,1700000000000],1700000000000)])
def test_add_storage(tmp_path,monkeypatch,timestamps,expected):
 from aml_api.store import SQLiteStore
 from aml_api.service import RouteAService
 from aml_api.schemas import AddRequest
 from faults import FakeEncoder,FakeVLM
 monkeypatch.setenv('OPENAI_API_KEY','offline')
 settings=dataclasses.replace(Settings.from_env(),storage_path=tmp_path/'add.sqlite3')
 svc=RouteAService(settings,store=SQLiteStore(settings.storage_path),embedder=FakeEncoder(),vlm=FakeVLM())
 req=AddRequest(request_id='r',user_id='u',session_id='s',messages=[{'role':'user','content':'synthetic source time','timestamp':t} for t in (timestamps or [None])])
 svc.add(req)
 db=sqlite3.connect(settings.storage_path);row=db.execute('SELECT source_timestamp_ms,created_at FROM memories').fetchone();db.close()
 assert row==(expected,source_created_at(expected) or '')
