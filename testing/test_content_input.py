import base64
import copy
import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aml_api.schemas import AddRequest, SearchRequest
from aml_api.store import SQLiteStore, payload_digest
from aml_api.production.service import ProductionService
from aml_api.production.add_control import ComputeRunner
from aml_api.production.process import StageRunner
from aml_api.production.policy import Policy
from aml_api.config import Settings
from aml_api.service import _flatten_messages
from faults import FaultBackend
from test_add_deadline import fake_compute
from server import build_app


def blocks(text):return [{'type':'text','text':text}]

def mixed_payload():
    uri='data:image/png;base64,'+base64.b64encode(b'synthetic-image-bytes').decode()
    return {'request_id':'r','user_id':'u','session_id':'s','messages':[
        {'role':'user','content':'synthetic bicycle first','timestamp':1704067200000},
        {'role':'assistant','content':[{'type':'text','text':'middle before image'},{'type':'image_url','image_url':{'url':uri}},{'type':'text','text':'middle after image'}]},
        {'role':'user','content':'最后一条\nsecond line'}]}

def array_form(payload):
    p=copy.deepcopy(payload)
    for m in p['messages']:
        if isinstance(m['content'],str):m['content']=blocks(m['content'])
    return p


def test_mixed_order_and_canonical_idempotency_equivalence():
    p=mixed_payload();original=copy.deepcopy(p)
    string=AddRequest.model_validate(p);array=AddRequest.model_validate(array_form(p))
    assert p==original
    assert string.model_dump(mode='json')==array.model_dump(mode='json')
    assert payload_digest(string.model_dump(mode='json'))==payload_digest(array.model_dump(mode='json'))
    content,transcript,images=_flatten_messages(string)
    assert [v['type'] for v in content]==['text','text','image_url','text','text']
    assert [content[i]['text'] for i in [0,1,3,4]]==['[user] synthetic bicycle first','[assistant] middle before image','middle after image','[user] 最后一条\nsecond line']
    assert images==[p['messages'][1]['content'][1]['image_url']['url']]
    assert string.messages[0].timestamp==1704067200000
    assert _flatten_messages(string)==_flatten_messages(array)


def test_search_string_and_ordered_array_bind_identically():
    one=SearchRequest(user_id='u',query='synthetic bicycle',top_k=100,options=['A','B'])
    two=SearchRequest(user_id='u',query=blocks('synthetic bicycle'),top_k=100,options=['A','B'])
    assert one.model_dump(mode='json')==two.model_dump(mode='json')
    assert payload_digest(one.model_dump(mode='json'))==payload_digest(two.model_dump(mode='json'))
    query=[*blocks('before'),{'type':'image_url','image_url':{'url':'data:image/png;base64,eA=='}},*blocks('after')]
    assert SearchRequest(user_id='u',query=query,top_k=100).model_dump(mode='json')['query']==query


@pytest.mark.parametrize('value',[None,{}, {'text':'not a content part array'}, 1,1.25,True,[], '', ' \n\t', [None], ['text'], [{'type':'text','text':None}]])
def test_invalid_types_and_empty_inputs_remain_rejected(value):
    p=mixed_payload();p['messages'][0]['content']=value
    with pytest.raises(ValidationError):AddRequest.model_validate(p)
    with pytest.raises(ValidationError):SearchRequest(user_id='u',query=value,top_k=100)


def test_schema_advertises_string_and_array_but_serializes_arrays():
    for cls,field in [(AddRequest,'messages'),(SearchRequest,'query')]:
        schema=cls.model_json_schema()
        prop=schema['$defs']['AddMessage']['properties']['content'] if cls is AddRequest else schema['properties'][field]
        assert {v['type'] for v in prop['anyOf']}=={'string','array'}
    assert isinstance(SearchRequest(user_id='u',query='text',top_k=1).model_dump()['query'],list)


def test_real_atomic_add_replay_conflict_search_and_existing_array_store(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-only')
    settings=replace(Settings.from_env(),storage_path=tmp_path/'new.sqlite3',memory_system_key='offline-key-0000000000000000')
    SQLiteStore(settings.storage_path)
    policy=Policy(total_seconds=30,simple_seconds=15,legacy_seconds=5,seed_reserve_seconds=2,cleanup_seconds=.2,max_concurrent_searches=1,worker_memory_mb=1024)
    service=ProductionService(settings,policy=policy,add_runner=ComputeRunner(fake_compute),runner=StageRunner(FaultBackend()))
    app=build_app(settings=settings,service=service)
    p=mixed_payload();old=array_form(p);old['request_id']='existing-array'
    headers={'X-Api-Key':settings.memory_system_key}
    with TestClient(app) as c:
        assert c.post('/add',json=old,headers=headers).status_code==200
        old_string=copy.deepcopy(p);old_string['request_id']='existing-array'
        assert c.post('/add',json=old_string,headers=headers).status_code==200
        new=c.post('/add',json=p,headers=headers)
        assert new.status_code==200 and new.json()=={'success':True,'request_id':'r','user_id':'u','session_id':'s'}
        before=hashlib.sha256(settings.storage_path.read_bytes()).hexdigest()
        assert c.post('/add',json=p,headers=headers).json()==new.json()
        assert c.post('/add',json=array_form(p),headers=headers).json()==new.json()
        assert hashlib.sha256(settings.storage_path.read_bytes()).hexdigest()==before
        changed=copy.deepcopy(p);changed['messages'][2]['content']='different'
        assert c.post('/add',json=changed,headers=headers).status_code==409
        changed=copy.deepcopy(p);changed['user_id']='other'
        assert c.post('/add',json=changed,headers=headers).status_code==409
        first=c.post('/search',json={'user_id':'u','query':'synthetic bicycle','top_k':100},headers=headers)
        second=c.post('/search',json={'user_id':'u','query':blocks('synthetic bicycle'),'top_k':100},headers=headers)
        assert first.status_code==second.status_code==200 and first.json()==second.json()
        expected=_flatten_messages(AddRequest.model_validate(p))[0]
        assert first.json()['data'] and all(v['content']==expected for v in first.json()['data'])
        assert c.post('/search',json={'user_id':'other','query':'synthetic bicycle','top_k':100},headers=headers).json()=={'data':[]}
        assert hashlib.sha256(settings.storage_path.read_bytes()).hexdigest()==before
    assert settings.storage_path.with_suffix('.computations').read_text().splitlines()==['1','1']
    with sqlite3.connect(settings.storage_path) as db:
        assert db.execute('select count(*) from memories').fetchone()[0]==2
        assert db.execute("select count(*) from add_requests where state='complete'").fetchone()[0]==2
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
