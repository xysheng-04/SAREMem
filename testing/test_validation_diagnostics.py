import copy
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aml_api.config import Settings
from aml_api.schemas import AddRequest, AddResponse, SearchResponse
from aml_api.production.diagnostics import FailureSink, active, validation_failure, validation_fields
from aml_api.production.http_errors import validation_details
from server import build_app

SECRET = 'PRIVATE-content-image-key-value'

class Spy:
    calls = 0
    def add(self, body):
        self.calls += 1
        return AddResponse(request_id=body.request_id,user_id=body.user_id,session_id=body.session_id)
    def search(self, body):
        self.calls += 1
        return SearchResponse(data=[])

class Sink:
    def __init__(self): self.rows=[]
    def emit(self, row): self.rows.append(copy.deepcopy(row))

@pytest.fixture
def api(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY','offline-only')
    from aml_api.production import request_clock
    sink=Sink();monkeypatch.setattr(request_clock,'FailureSink',lambda directory:sink)
    settings=replace(Settings.from_env(),storage_path=tmp_path/'unused.sqlite3',memory_system_key='offline-only-key-000000000000')
    spy=Spy();app=build_app(settings=settings,service=spy)
    with TestClient(app) as client:
        yield client,sink,spy,settings


def payload():
    return {'request_id':'r','user_id':'u','session_id':'s','messages':[{'role':'user','timestamp':1704067200000,'content':[{'type':'text','text':SECRET}]}]}

@pytest.mark.parametrize('change,expected_type,expected_loc',[
    ('content_object','list_type',['body','messages',0,'content']),
    ('empty_text','string_too_short',['body','messages',0,'content',0,'text','text']),
    ('bad_role','literal_error',['body','messages',0,'role']),
    ('extra','extra_forbidden',['body','<unknown>']),
    ('missing','missing',['body','request_id']),
])
def test_actual_422_has_private_field_diagnostics_and_unchanged_response(api,change,expected_type,expected_loc):
    client,sink,spy,settings=api;p=payload()
    if change=='content_object':p['messages'][0]['content']={'text':SECRET}
    elif change=='empty_text':p['messages'][0]['content'][0]['text']=''
    elif change=='bad_role':p['messages'][0]['role']=SECRET
    elif change=='extra':p[SECRET]=SECRET
    else:del p['request_id']
    with pytest.raises(ValidationError) as caught:AddRequest.model_validate(p)
    expected=validation_details([{**e,'loc':('body',*e['loc'])} for e in caught.value.errors()])
    response=client.post('/add',json=p)
    assert response.status_code==422 and response.json()=={'detail':expected}
    assert 'validation_errors' not in response.text
    row=sink.rows[-1]
    assert row['stage']=='parse_validation' and row['code']=='schema'
    assert {'type':expected_type,'loc':expected_loc} in row['validation_errors']
    assert SECRET not in json.dumps(row) and spy.calls==0
    assert not settings.storage_path.exists()


def test_malformed_json_and_search_field_diagnostics(api):
    client,sink,spy,settings=api
    response=client.post('/add',content='{"messages":'+SECRET,headers={'Content-Type':'application/json'})
    assert response.status_code==422
    assert sink.rows[-1]['validation_errors'][0]['type']=='json_invalid'
    assert SECRET not in json.dumps(sink.rows[-1])
    response=client.post('/search',json={'user_id':'u','top_k':100,'query':{'text':SECRET}})
    assert response.status_code==422
    assert sink.rows[-1]['validation_errors']==[{'type':'list_type','loc':['body','query']}]
    assert spy.calls==0


def test_success_has_no_diagnostic_and_field_diagnostic_failure_keeps_422(api,monkeypatch):
    client,sink,spy,settings=api
    response=client.post('/add',json=payload(),headers={'X-Api-Key':settings.memory_system_key})
    assert response.status_code==200 and spy.calls==1 and not sink.rows
    from aml_api.production import diagnostics
    def broken(errors):raise OSError(SECRET)
    monkeypatch.setattr(diagnostics,'validation_fields',broken)
    response=client.post('/add',json={})
    assert response.status_code==422 and set(response.json())=={'detail'}
    assert 'validation_errors' not in sink.rows[-1] and SECRET not in json.dumps(sink.rows[-1])


def test_bounds_and_sink_resanitization(tmp_path):
    errors=[{'type':SECRET,'loc':['body',SECRET,*range(30)],'input':SECRET,'ctx':{'secret':SECRET},'msg':SECRET} for _ in range(100)]
    record={'correlation':'a'*32,'operation':'add','status_attempted':422,'stage':'parse_validation','code':'schema'}
    token=active.set(record)
    try:validation_failure(errors)
    finally:active.reset(token)
    assert len(record['validation_errors'])==4 and record['validation_errors_truncated']
    assert all(len(e['loc'])==8 and set(e)=={'loc','type'} for e in record['validation_errors'])
    sink=FailureSink(tmp_path/'diagnostics')
    record['validation_errors'][0]['input']=SECRET
    record['validation_errors'][0]['type']=SECRET
    record['validation_errors'][0]['loc']=[SECRET]*100
    sink.emit(record);sink.items.join()
    raw=(tmp_path/'diagnostics/failures.jsonl').read_text()
    assert len(raw.encode())<=2048 and SECRET not in raw
    value=json.loads(raw);assert value['validation_errors_truncated']
    assert len(value['validation_errors'])==4
    assert value['validation_errors'][0]=={'type':'value_error','loc':['<unknown>']*8}


def test_all_whitelisted_paths_fit_line_limit(tmp_path):
    from aml_api.production.http_errors import FIELDS,VALIDATION_MESSAGES
    field=max(FIELDS,key=len);kind=max(VALIDATION_MESSAGES,key=len)
    sink=FailureSink(tmp_path/'diagnostics')
    record={'correlation':'a'*32,'operation':'search','status_attempted':422,'response_started':True,'response_completed':True,'delivery':'unconfirmed','stage':'parse_validation','code':'schema','queue_ms':1680000.123,'elapsed_ms':1680000.123,'validation_errors':[{'type':kind,'loc':[field]*8} for _ in range(4)]}
    sink.emit(record);sink.items.join()
    raw=(tmp_path/'diagnostics/failures.jsonl').read_bytes()
    assert len(raw)<=2048
    assert len(json.loads(raw)['validation_errors'])==4
