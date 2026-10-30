"""Offline budget tests: virtual latency and real formal ASGI responses."""
import asyncio
import json
import time

import httpx
from openai import OpenAI
import pytest

from aml_api.schemas import SearchRequest
from aml_api.production.policy import Policy
from aml_api.production.runtime import Controls
from aml_api.production.retry import AttemptBudget
from aml_api.production.transport import DeadlineTransport, network_timeout, bounded_search_client
from aml_api.production.service import ProductionService
from aml_api.production.request_clock import request_started
from server import build_app
from test_faults import setup, request, check_success, HEADERS, QUERY
from test_retries import RetryBackend


def test_default_profile_preserves_official_margin():
    p=Policy()
    assert (p.total_seconds,p.simple_seconds,p.legacy_seconds,p.seed_reserve_seconds,p.cleanup_seconds)==(1680,1140,420,90,5)
    assert p.simple_seconds+p.legacy_seconds+p.seed_reserve_seconds+3*p.cleanup_seconds<=p.total_seconds
    assert p.total_seconds+120<=1800
    assert p.provider_timeout_seconds==300 and p.connect_timeout_seconds==30


@pytest.mark.parametrize('changes', [
 {'provider_timeout_seconds':601}, {'provider_timeout_seconds':float('nan')},
 {'provider_timeout_seconds':0}, {'connect_timeout_seconds':61},
 {'connect_timeout_seconds':0}, {'connect_timeout_seconds':float('inf')},
 {'provider_timeout_seconds':10,'connect_timeout_seconds':11}])
def test_invalid_network_configuration_rejected(changes):
    with pytest.raises(ValueError):Policy(**changes)


def test_provider_timeout_from_env(monkeypatch):
    monkeypatch.setenv('AML_PROD_PROVIDER_TIMEOUT_SECONDS','240')
    monkeypatch.setenv('AML_PROD_CONNECT_TIMEOUT_SECONDS','20')
    p=Policy.from_env()
    assert p.provider_timeout_seconds==240 and p.connect_timeout_seconds==20


def test_fractional_time_budget_from_env(monkeypatch):
    monkeypatch.setenv('AML_PROD_CLEANUP_SECONDS','0.5')
    monkeypatch.setenv('AML_PROD_CONNECT_TIMEOUT_SECONDS','12.5')
    p=Policy.from_env()
    assert p.cleanup_seconds==.5 and p.connect_timeout_seconds==12.5


def test_slow_ninety_second_provider_can_succeed_formal_http(setup):
    response,events=request(setup,RetryBackend(('slow',)))
    check_success(response,setup[0])
    assert events[0]['final']=='simple'
    assert events[0]['stages'][0]['test_sends']==4
    assert events[0]['stages'][0]['test_timeouts'][0]>90
    assert not events[0]['stages'][0]['deadline_exhausted']


def test_attempt_budget_uses_configured_timeout_without_reset(setup,monkeypatch):
    clock=[1000.]
    monkeypatch.setattr('time.monotonic',lambda:clock[0])
    monkeypatch.setattr('time.sleep',lambda duration:clock.__setitem__(0,clock[0]+duration))
    controls=Controls(Policy(),setup[0],1700)
    seen=[]
    def send(timeout):
        seen.append(timeout)
        if len(seen)==1:
            clock[0]+=290
            raise httpx.ReadTimeout('MUST_NOT_LEAK')
        return {}
    budget=AttemptBudget();budget.send(controls,send)
    assert seen==[300,300] and budget.sends==2
    clock[0]=1690
    budget.send(controls,lambda timeout:seen.append(timeout))
    assert seen[-1]==10 and budget.sends==3


def test_transport_clips_each_send_to_remaining_stage_time(monkeypatch):
    clock=[1000.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    seen=[]
    def handle(req):
        seen.append(dict(req.extensions['timeout']))
        return httpx.Response(200,json={})
    p=Policy();transport=DeadlineTransport(p,1700,httpx.MockTransport(handle))
    with httpx.Client(transport=transport,timeout=network_timeout(p,1700)) as client:
        client.get('https://invalid.test')
        clock[0]=1690;client.get('https://invalid.test')
        clock[0]=1701
        with pytest.raises(TimeoutError):client.get('https://invalid.test')
    assert seen[0]=={'connect':30,'read':300,'write':30,'pool':30}
    assert set(seen[1].values())=={10}
    assert len(seen)==2


def test_frozen_sdk_retry_count_preserved_in_bounded_search_client():
    seen=[]
    def reply(req):
        seen.append(req.extensions['timeout'])
        return httpx.Response(200,json={'object':'list','data':[]})
    http_client=httpx.Client(transport=httpx.MockTransport(reply),follow_redirects=False,trust_env=False)
    old=OpenAI(api_key='offline-placeholder',base_url='https://invalid.test/v1',max_retries=2,http_client=http_client)
    new=bounded_search_client(old,Policy(),time.monotonic()+500)
    try:
        assert new is old and new._client is http_client
        assert not new.is_closed() and new.max_retries==2
        assert new._client.follow_redirects is False and new._client.trust_env is False
        new.models.list()
        assert seen[0]['read']==300 and seen[0]['connect']==30
    finally:new.close()


def test_exhausted_simple_and_legacy_leave_seed_reserve(setup,monkeypatch):
    clock=[1000.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    seen=[];p=Policy()
    class Runner:
        def run(self,stage,settings,req,policy,deadline,seed,publish):
            seen.append((stage,clock[0],deadline))
            if stage=='simple':
                from aml_api.postseed.seed_port import digest
                publish({'request':digest(req.model_dump(mode='json')),'hits':[]})
            if stage!='seed':
                clock[0]=deadline+policy.cleanup_seconds
                return {'error':'provider'}
            clock[0]=deadline
            return {'response':{'data':[]}}
    response=ProductionService(setup[0],policy=p,runner=Runner()).search(
        SearchRequest.model_validate(QUERY))
    assert response.model_dump()=={'data':[]}
    assert [item[0] for item in seen]==['simple','legacy','seed']
    assert seen[0][2]-seen[0][1]==1140
    assert seen[1][2]-seen[1][1]==420
    assert seen[2][2]-seen[2][1]>=90
    assert clock[0]<=1000+p.total_seconds


def test_upload_and_thread_entry_share_arrival_deadline_formal_http(setup):
    async def verify():
        records=[]
        class Runner:
            def run(self,stage,settings,req,policy,deadline,seed,publish):
                records.append({'arrival':request_started.get(),'call':time.monotonic(),'deadline':deadline})
                return {'response':{'data':[]}}
        # No worker spawn here: test request-body/ASGI/thread propagation itself.
        p=Policy(total_seconds=.3,simple_seconds=.25,legacy_seconds=.04,seed_reserve_seconds=.04,cleanup_seconds=.005)
        svc=ProductionService(setup[0],policy=p,runner=Runner())
        app=build_app(settings=setup[0],service=svc)
        async def body():
            await asyncio.sleep(.06)
            yield json.dumps(QUERY).encode()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://offline.test') as client:
            r=await client.post('/search',headers={**HEADERS,'content-type':'application/json'},content=body())
        assert r.status_code==200 and r.json()=={'data':[]}
        assert set(r.headers)<={'content-type','content-length'}
        assert records[0]['arrival'] is not None
        assert records[0]['call']-records[0]['arrival']>=.05
        assert records[0]['deadline']-records[0]['arrival']==pytest.approx(.205,abs=.01)
        assert request_started.get() is None
    asyncio.run(verify())


def test_exhausted_arrival_budget_does_not_start_workers(setup):
    class NeverRun:
        def run(self,*a,**k):pytest.fail('request already exhausted its application deadline')
    token=request_started.set(time.monotonic()-1700)
    try:
        from aml_api.schemas import SearchRequest
        from aml_api.errors import UpstreamUnavailable
        with pytest.raises(UpstreamUnavailable):
            ProductionService(setup[0],runner=NeverRun()).search(SearchRequest.model_validate(QUERY))
    finally:request_started.reset(token)
