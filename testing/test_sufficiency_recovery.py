import asyncio, copy, dataclasses, json, sqlite3, threading, time
from concurrent.futures import ThreadPoolExecutor
import pytest
from aml_api.config import Settings
from aml_api.schemas import SearchRequest
from aml_api.errors import UpstreamUnavailable
from aml_api.production.service import ProductionService
from aml_api.production.policy import SearchFault
from aml_api.production.http_errors import SearchDeadlineExceeded
from aml_api.production.request_clock import request_cancelled, SearchRequestClock
from aml_api.production.diagnostics import active, FailureSink

IMAGE='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='

@pytest.fixture
def setup(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY','offline')
    monkeypatch.setenv('AML_SUFFICIENCY_MODE','on')
    monkeypatch.setenv('AML_SUFFICIENCY_MAX_SECONDS','90')
    monkeypatch.setenv('AML_FAILURE_DIAGNOSTICS','off')
    path=tmp_path/'synthetic.sqlite3'
    evidence={'data':[{'id':'m','content':[{'type':'text','text':'synthetic evidence'},
        {'type':'image_url','image_url':{'url':IMAGE}}],'score':1.0}]}
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE memories(memory_id TEXT,user_id TEXT,content_json TEXT,created_at INTEGER,source_timestamp_ms INTEGER)')
        db.execute('INSERT INTO memories VALUES(?,?,?,?,?)',('m','u',json.dumps(evidence['data'][0]['content']),1,None))
    settings=dataclasses.replace(Settings.from_env(),storage_path=path)
    request=SearchRequest.model_validate({'user_id':'u','top_k':1,'query':[{'type':'text','text':'synthetic question'}]})
    return settings,request,evidence

class Runner:
    def __init__(self,outcomes,evidence,hook=None):
        self.outcomes=list(outcomes);self.evidence=evidence;self.calls=[];self.deadlines=[];self.hook=hook
    def run(self,stage,settings,request,policy,deadline,checkpoint,publish):
        self.calls.append(stage)
        if stage=='simple':return {'response':copy.deepcopy(self.evidence)}
        assert stage=='sufficiency'
        assert checkpoint==self.evidence
        self.deadlines.append(deadline)
        outcome=self.outcomes.pop(0)
        if self.hook:self.hook(outcome,checkpoint)
        if outcome=='high':return {'response':copy.deepcopy(self.evidence)}
        if outcome=='low':return {'response':{'data':[]}}
        if outcome=='mutate':
            value=copy.deepcopy(self.evidence);value['data'][0]['content'][0]['text']='MUTATED'
            return {'response':value}
        if outcome=='exception':raise SearchFault('worker_exit')
        if outcome=='terminal':return {'error':'internal','terminal':True}
        return {'error':outcome}

@pytest.mark.parametrize('outcomes,expected,attempts,fallback',[
    (['high'],1,1,False),(['low'],0,1,False),
    (['provider','high'],1,2,False),(['provider','schema','high'],1,3,False),
    (['provider','schema','low'],0,3,False),
    (['provider']*3,1,3,True),(['exception']*3,1,3,True),
])
def test_paths(setup,outcomes,expected,attempts,fallback):
    settings,request,evidence=setup;before=settings.storage_path.read_bytes()
    runner=Runner(outcomes,evidence);svc=ProductionService(settings,runner=runner)
    record={};token=active.set(record)
    try:response=svc.search(request)
    finally:active.reset(token)
    assert len(response.data)==expected
    if expected:assert response.model_dump(mode='json')==evidence
    assert runner.calls==['simple']+['sufficiency']*attempts
    assert runner.deadlines==sorted(runner.deadlines)
    assert record['sufficiency_fallback'] is fallback
    assert record['sufficiency_attempts']==attempts
    assert settings.storage_path.read_bytes()==before
    assert svc.slots.acquire(blocking=False);svc.slots.release()

@pytest.mark.parametrize('failure',['deadline','retry_budget','scope_denied','source_integrity',
    'source_changed','source_unavailable','response_validation','provider_auth','terminal','mutate'])
def test_hard_errors_never_enable_fallback(setup,failure):
    settings,request,evidence=setup
    runner=Runner([failure],evidence);svc=ProductionService(settings,runner=runner)
    exception=SearchDeadlineExceeded if failure in {'deadline','retry_budget'} else UpstreamUnavailable
    with pytest.raises(exception):svc.search(request)
    assert runner.calls==['simple','sufficiency']

def test_503_then_timeout_does_not_fallback(setup):
    settings,request,evidence=setup;runner=Runner(['provider','deadline'],evidence)
    with pytest.raises(SearchDeadlineExceeded):ProductionService(settings,runner=runner).search(request)
    assert runner.calls.count('sufficiency')==2

def test_deadline_not_renewed(setup):
    settings,request,evidence=setup
    runner=Runner(['provider']*3,evidence,lambda *_:time.sleep(.03))
    svc=ProductionService(settings,runner=runner);svc.sufficiency_config=(.2,.02)
    with pytest.raises(SearchDeadlineExceeded):svc.search(request)
    assert runner.calls==['simple','sufficiency']

def test_cancel_after_third_failure_prevents_fallback(setup):
    settings,request,evidence=setup;cancel=threading.Event();n=[]
    def hook(*_):
        n.append(1)
        if len(n)==3:cancel.set()
    runner=Runner(['provider']*3,evidence,hook);token=request_cancelled.set(cancel)
    try:
        with pytest.raises(SearchDeadlineExceeded):ProductionService(settings,runner=runner).search(request)
    finally:request_cancelled.reset(token)

def test_changed_store_blocks_fallback(setup):
    settings,request,evidence=setup;n=[]
    def hook(*_):
        n.append(1)
        if len(n)==3:
            with sqlite3.connect(settings.storage_path) as db:db.execute("UPDATE memories SET user_id='other'")
    with pytest.raises(UpstreamUnavailable):
        ProductionService(settings,runner=Runner(['provider']*3,evidence,hook)).search(request)

def test_failed_attempt_cannot_mutate_saved_evidence(setup):
    settings,request,evidence=setup
    def hook(_,checkpoint):checkpoint['data'][0]['content'][0]['text']='MUTATED'
    svc=ProductionService(settings,runner=Runner(['provider']*3,evidence,hook))
    assert svc.search(request).model_dump(mode='json')==evidence

def test_counts_do_not_cross_requests(setup):
    settings,request,evidence=setup
    def one(_):
        runner=Runner(['provider']*3,evidence)
        result=ProductionService(settings,runner=runner).search(request)
        return result.model_dump(mode='json'),runner.calls.count('sufficiency')
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,range(3)))
    assert rows==[(evidence,3)]*3

class Sink:
    def __init__(self):self.records=[]
    def emit(self,row):self.records.append(dict(row))

@pytest.mark.parametrize('outcomes,expected_outcome',[
    (['high'],'pass'),(['low'],'refuse'),(['provider']*3,'fallback')])
def test_real_http_and_audit(setup,outcomes,expected_outcome):
    from aml_api.app import create_app
    from fastapi.testclient import TestClient
    settings,request,evidence=setup
    settings=dataclasses.replace(settings,memory_system_key='synthetic-test-key-at-least-24-characters')
    service=ProductionService(settings,runner=Runner(outcomes,evidence))
    app=create_app(settings,service=service,production=True)
    with TestClient(app) as client:
        layer=app.middleware_stack
        while not isinstance(layer,SearchRequestClock):layer=layer.app
        layer.sufficiency_sink=Sink()
        sink=layer.sufficiency_sink
        response=client.post('/search',json=request.model_dump(mode='json'),headers={'X-Api-Key':settings.memory_system_key})
        assert response.status_code==200
        assert response.json()==({'data':[]} if expected_outcome=='refuse' else evidence)
        row=sink.records[-1]
        assert row['sufficiency_outcome']==expected_outcome and row['utc']
        assert row['status_attempted']==200 and row['response_completed']

def test_disk_audit_contains_only_safe_fields(tmp_path):
    sink=FailureSink(tmp_path,filename='sufficiency.jsonl')
    sink.emit({'utc':'2026-11-04T00:00:00Z','sufficiency_attempts':3,'sufficiency_fallback':True,
        'sufficiency_outcome':'fallback','query':'SECRET_QUERY','header':'SECRET_KEY','content':IMAGE})
    sink.items.join()
    raw=(tmp_path/'sufficiency.jsonl').read_text()
    assert 'SECRET' not in raw and 'base64' not in raw
    assert json.loads(raw)['sufficiency_attempts']==3


class FailingSupervisedBackend:
    def __init__(self,evidence):self.evidence=evidence
    def __call__(self,stage,settings,request,policy,deadline,checkpoint,publish,stats):
        if stage=='simple':return self.evidence
        assert stage=='sufficiency'
        raise SearchFault('provider_retry_exhausted')

def test_real_subprocesses_and_revalidation(setup):
    from aml_api.production.process import StageRunner
    from aml_api.production.policy import Policy
    settings,request,evidence=setup
    before=settings.storage_path.read_bytes()
    service=ProductionService(settings,policy=dataclasses.replace(Policy(),worker_memory_mb=1024),
        runner=StageRunner(FailingSupervisedBackend(evidence)))
    record={};token=active.set(record)
    try:response=service.search(request)
    finally:active.reset(token)
    assert response.model_dump(mode='json')==evidence
    assert record['sufficiency_attempts']==3 and record['sufficiency_fallback'] is True
    assert settings.storage_path.read_bytes()==before

def test_same_service_new_request_resets_count(setup):
    settings,request,evidence=setup
    runner=Runner(['provider']*6,evidence);service=ProductionService(settings,runner=runner)
    for _ in range(2):assert service.search(request).model_dump(mode='json')==evidence
    assert runner.calls==['simple']+['sufficiency']*3+['simple']+['sufficiency']*3

class FakeClock:
    def __init__(self):self.now=100.0
    def monotonic(self):return self.now

@pytest.mark.parametrize('seconds_each',[40.0,60.0,89.0])
def test_three_slow_503_get_full_attempt_budgets(setup,monkeypatch,seconds_each):
    import aml_api.production.service as service_module
    import aml_api.production.sufficiency_fallback as recovery_module
    settings,request,evidence=setup;clock=FakeClock()
    monkeypatch.setattr(service_module,'time',clock);monkeypatch.setattr(recovery_module,'time',clock)
    runner=Runner(['provider']*3,evidence,lambda *_:setattr(clock,'now',clock.now+seconds_each))
    service=ProductionService(settings,runner=runner)
    response=service.search(request)
    assert response.model_dump(mode='json')==evidence
    assert runner.deadlines==[190.0,190.0+seconds_each,190.0+2*seconds_each]
    assert clock.now==100.0+3*seconds_each

def test_original_search_end_caps_recovery(setup,monkeypatch):
    import aml_api.production.service as service_module
    import aml_api.production.sufficiency_fallback as recovery_module
    settings,request,evidence=setup;clock=FakeClock()
    monkeypatch.setattr(service_module,'time',clock);monkeypatch.setattr(recovery_module,'time',clock)
    runner=Runner(['provider']*3,evidence,lambda *_:setattr(clock,'now',clock.now+40.0))
    service=ProductionService(settings,runner=runner)
    response=__import__('aml_api.schemas',fromlist=['SearchResponse']).SearchResponse.model_validate(evidence)
    with pytest.raises(SearchDeadlineExceeded):service._sufficiency(request,response,service.policy,160.0,None)
    assert runner.deadlines==[155.0,155.0]

def test_single_attempt_still_stops_at_90(setup,monkeypatch):
    import aml_api.production.service as service_module
    import aml_api.production.sufficiency_fallback as recovery_module
    settings,request,evidence=setup;clock=FakeClock()
    monkeypatch.setattr(service_module,'time',clock);monkeypatch.setattr(recovery_module,'time',clock)
    runner=Runner(['provider']*3,evidence,lambda *_:setattr(clock,'now',clock.now+91.0))
    record={};token=active.set(record)
    try:
        with pytest.raises(SearchDeadlineExceeded):ProductionService(settings,runner=runner).search(request)
    finally:active.reset(token)
    assert runner.deadlines==[190.0] and record['sufficiency_stop_reason']=='attempt_deadline'

def test_recovery_total_270_is_not_extended(setup,monkeypatch):
    import aml_api.production.service as service_module
    import aml_api.production.sufficiency_fallback as recovery_module
    settings,request,evidence=setup;clock=FakeClock()
    monkeypatch.setattr(service_module,'time',clock);monkeypatch.setattr(recovery_module,'time',clock)
    runner=Runner(['provider']*3,evidence,lambda *_:setattr(clock,'now',clock.now+89.0))
    service=ProductionService(settings,runner=runner)
    response=__import__('aml_api.schemas',fromlist=['SearchResponse']).SearchResponse.model_validate(evidence)
    import aml_api.production.validation as validation
    original=validation.finalize
    def slow_validation(*args,**kwargs):
        clock.now+=4.0
        return original(*args,**kwargs)
    monkeypatch.setattr(validation,'finalize',slow_validation)
    record={};token=active.set(record)
    try:
        with pytest.raises(SearchDeadlineExceeded):service._sufficiency(request,response,service.policy,1780.0,None)
    finally:active.reset(token)
    assert runner.deadlines==[190.0,279.0,368.0]
    assert record['sufficiency_fallback'] is False and record['sufficiency_stop_reason']=='recovery_deadline'
