"""Synthetic real SDK + formal HTTP entry; never calls a remote provider."""
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
import json
import time

import httpx
from openai import OpenAI, APIStatusError
import pytest

from faults import FakeFrozen, FaultBackend
from test_faults import setup, request, check_success
from aml_api.production.policy import Policy, SearchFault
from aml_api.production.retry import AttemptBudget, retry_delay, transient
from aml_api.production.runtime import Controls


@dataclass
class RetryBackend:
    sequence: tuple = ()
    legacy_mode: str = 'success'
    seed_mode: str = 'success'
    retry_after: str | None = None

    def __call__(self, stage, settings, req, policy, deadline, checkpoint, publish, stats):
        if stage != 'simple':
            return FaultBackend(legacy_mode=self.legacy_mode, seed_mode=self.seed_mode)(
                stage, settings, req, policy, deadline, checkpoint, publish, stats)
        import socket
        socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden'))
        from aml_api.production import runtime as rt
        rt.FrozenProviders = FakeFrozen
        provider_type = rt.AssessmentProvider
        sends, repairs, timeouts = [], [], []
        def respond(http_request):
            body = json.loads(http_request.content)
            payload = json.loads(body['messages'][1]['content'][0]['text'])
            repairs.append('validation_feedback' in payload)
            timeouts.append(http_request.extensions['timeout']['read'])
            action = self.sequence[len(sends)] if len(sends) < len(self.sequence) else 'ok'
            sends.append(action)
            if action == 'slow':
                # Virtual remote latency; no sleeping or external request.
                if http_request.extensions['timeout']['read'] < 90:
                    raise httpx.ReadTimeout('MUST_NOT_LEAK', request=http_request)
                original_clock = time.monotonic
                time.monotonic = lambda: original_clock() + 90
            if action == 'hang':
                time.sleep(30)
            if action in {'timeout','connect'}:
                cls = httpx.ReadTimeout if action == 'timeout' else httpx.ConnectError
                raise cls('MUST_NOT_LEAK', request=http_request)
            if action == 'wire_json':
                return httpx.Response(200,content=b'{invalid',headers={'content-type':'application/json'})
            if action == 'wire_shape':
                return httpx.Response(200,json=[])
            if isinstance(action, int):
                headers = {'retry-after': self.retry_after} if self.retry_after is not None else {}
                return httpx.Response(action, headers=headers, json={'error':{'message':'MUST_NOT_LEAK'}})
            content = {'relevance':'context','fragment_ids':['f0'],'possible_conflict':False}
            if action == 'schema': content = {'relevance':'invented'}
            if action == 'id': content['fragment_ids'] = ['outside']
            if action == 'duplicate': content['fragment_ids'] = ['f0','f0']
            if action == 'missing': content['fragment_ids'] = []
            if action == 'uncertain': content.update(relevance='uncertain',fragment_ids=[])
            text = '{invalid' if action == 'json' else json.dumps(content)
            if action == 'duplicate_key': text = '{"relevance":"context","relevance":"direct"}'
            choices = [{'index':0,'message':{'role':'assistant','content':text},
                        'finish_reason':'length' if action == 'length' else 'stop'}]
            return httpx.Response(200, json={'id':'synthetic','object':'chat.completion',
                'created':0,'model':'gpt-4o-mini','choices':None if action=='envelope' else choices})
        def provider(s, controls):
            obj = provider_type.__new__(provider_type)
            obj.controls = controls
            obj.client = OpenAI(api_key='offline-placeholder', base_url='https://invalid.test/v1',
                max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            return obj
        rt.AssessmentProvider = provider
        try:
            return rt.simple(settings, req, policy, deadline, publish, stats)
        finally:
            # Test-only counters; never present in production code or HTTP body.
            stats.update(test_sends=len(sends),test_repairs=repairs,test_timeouts=timeouts)


@pytest.mark.parametrize('failure', ['timeout','connect',408,425,429,500,502,503,504])
def test_transient_recovers_without_replaying_completed_candidates(setup, failure):
    response, events = request(setup, RetryBackend((failure,)))
    check_success(response, setup[0])
    event = events[0]
    assert event['final'] == 'simple' and len(event['stages']) == 1
    assert event['stages'][0]['test_sends'] == 5  # four candidates + one retry
    assert not any(event['stages'][0]['test_repairs'])


@pytest.mark.parametrize('failure', ['json','schema','id','duplicate','missing','length','duplicate_key','envelope','wire_json','wire_shape'])
def test_one_protocol_repair_uses_frozen_strict_assessment(setup, failure):
    response, events = request(setup, RetryBackend((failure,)))
    check_success(response, setup[0])
    stage = events[0]['stages'][0]
    assert events[0]['final'] == 'simple'
    assert stage['test_sends'] == 5
    assert stage['test_repairs'] == [False,True,False,False,False]


@pytest.mark.parametrize('sequence', [('timeout','json'),('json','timeout')])
def test_shared_three_send_budget_across_transport_and_repair(setup, sequence):
    response, events = request(setup, RetryBackend(sequence))
    check_success(response, setup[0])
    assert events[0]['final'] == 'simple'
    assert events[0]['stages'][0]['test_sends'] == 6


@pytest.mark.parametrize('sequence,sends,code', [
    (('timeout','timeout'),2,'provider_retry_exhausted'),
    (('json','json'),2,'schema'),
    (('timeout','json','timeout'),3,'provider_retry_exhausted'),
    (('json','timeout','timeout'),3,'provider_retry_exhausted'),
    ((401,),1,'provider'),((403,),1,'provider'),((400,),1,'provider'),
    ((404,),1,'provider'),((409,),1,'provider'),((422,),1,'provider'),
])
def test_exhaustion_and_permanent_failures_fall_back(setup, sequence, sends, code):
    response, events = request(setup, RetryBackend(sequence))
    check_success(response, setup[0])
    assert events[0]['final'] == 'legacy'
    assert [s['path'] for s in events[0]['stages']] == ['simple','legacy']
    first = events[0]['stages'][0]
    assert first['test_sends'] == sends and first['code'] == code


def test_retry_exhaustion_then_legacy_failure_reuses_seed(setup):
    response, events = request(setup, RetryBackend(('timeout','timeout'),legacy_mode='error'))
    check_success(response, setup[0])
    assert events[0]['final'] == 'seed'
    assert events[0]['stages'][-1]['phase'] == 'final_validation'
    assert events[0]['stages'][-1]['seed_existing']


def test_retry_after_too_long_falls_back_without_waiting(setup):
    started = time.monotonic()
    # Bound the whole HTTP request, including cold spawned-worker imports.
    policy = Policy(total_seconds=30,simple_seconds=8,legacy_seconds=7,
                    seed_reserve_seconds=10,cleanup_seconds=.5)
    response, events = request(setup, RetryBackend((429,),retry_after='1800'),policy=policy)
    check_success(response, setup[0])
    assert events[0]['final'] == 'legacy'
    assert events[0]['stages'][0]['code'] == 'retry_budget'
    assert events[0]['stages'][0]['test_sends'] == 1
    assert time.monotonic()-started < policy.total_seconds


def test_retry_after_zero_is_honored(setup):
    response, events = request(setup, RetryBackend((429,),retry_after='0'))
    check_success(response, setup[0])
    assert events[0]['final'] == 'simple'
    assert events[0]['stages'][0]['test_sends'] == 5


def test_hanging_retry_is_killed_and_seed_time_reserved(setup):
    # Reserve enough wall time for cold spawn on the small Linux host; the
    # injected 30s hang must still be terminated at the simple stage deadline.
    policy = Policy(total_seconds=30,simple_seconds=8,legacy_seconds=7,
                    seed_reserve_seconds=10,cleanup_seconds=.5)
    started = time.monotonic()
    response, events = request(setup, RetryBackend(('timeout','hang'),legacy_mode='error'),policy=policy)
    check_success(response, setup[0])
    assert events[0]['final'] == 'seed'
    assert [s['path'] for s in events[0]['stages']] == ['simple','legacy','seed']
    assert events[0]['stages'][0]['code'] == 'deadline'
    assert time.monotonic()-started < policy.total_seconds
    # request() also verifies that all spawned children have been reaped.


def test_uncertain_needs_no_repair_and_production_has_no_diagnostics(setup,tmp_path):
    before = set(tmp_path.rglob('*'))
    response, events = request(setup, RetryBackend(('uncertain',)),diagnostics=False)
    check_success(response, setup[0])
    assert events == []
    assert not [p for p in set(tmp_path.rglob('*'))-before if p.suffix in {'.jsonl','.log'}]


@pytest.mark.parametrize('seconds', [1680.001,1740,1790,1800,3600,float('nan'),float('inf')])
def test_configuration_cannot_extend_total_deadline(seconds):
    with pytest.raises(ValueError): Policy(total_seconds=seconds)


def status_error(value):
    return APIStatusError('MUST_NOT_LEAK',response=httpx.Response(429,
        headers={'retry-after':value},request=httpx.Request('POST','https://invalid.test')),body=None)


@pytest.mark.parametrize('value', ['bad','-1','nan','inf'])
def test_invalid_retry_after_uses_bounded_default(value):
    assert retry_delay(status_error(value)) == .5


def test_http_date_retry_after(monkeypatch):
    monkeypatch.setattr('aml_api.production.retry.time.time',lambda:1000)
    date = format_datetime(datetime.fromtimestamp(1002,timezone.utc),usegmt=True)
    assert retry_delay(status_error(date)) == 2


def test_no_send_when_attempt_time_is_insufficient(setup):
    controls = Controls(Policy(),setup[0],time.monotonic()+.01)
    with pytest.raises(SearchFault,match='retry_budget'):
        AttemptBudget().send(controls,lambda timeout: pytest.fail('unexpected send'))


def test_no_send_after_retry_sleep_overshoots_deadline(setup,monkeypatch):
    clock = [100.]
    monkeypatch.setattr('aml_api.production.retry.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('aml_api.production.retry.time.sleep',lambda _:clock.__setitem__(0,200.))
    controls = Controls(Policy(),setup[0],103)
    sends = []
    def send(timeout):
        sends.append(timeout)
        raise httpx.ReadTimeout('MUST_NOT_LEAK')
    with pytest.raises(TimeoutError): AttemptBudget().send(controls,send)
    assert sends == [3.]


def test_sdk_disabled_and_timeout_clamped(setup):
    controls = Controls(Policy(),setup[0],time.monotonic()+2)
    values = []
    AttemptBudget().send(controls,lambda timeout:values.append(timeout))
    assert 1 <= values[0] <= 2
    assert not transient(httpx.UnsupportedProtocol('MUST_NOT_LEAK'))


def test_actual_provider_disables_hidden_sdk_transport_retries_and_redirects(setup):
    from aml_api.production.runtime import AssessmentProvider
    controls = Controls(Policy(),setup[0],time.monotonic()+30)
    provider = AssessmentProvider(setup[0],controls)
    try:
        assert provider.client.max_retries == 0
        assert provider.client._client.follow_redirects is False
        assert provider.client._client._transport.delegate._pool._retries == 0
    finally:
        provider.close()


def test_shared_cap_never_sends_fourth_request(setup):
    controls = Controls(Policy(),setup[0],time.monotonic()+30)
    calls = []
    budget = AttemptBudget()
    for _ in range(3): budget.send(controls,lambda timeout:calls.append(timeout))
    with pytest.raises(SearchFault,match='retry_limit'):
        budget.send(controls,lambda timeout:calls.append(timeout))
    assert len(calls) == 3


def test_second_protocol_repair_context_is_rejected(setup):
    controls = Controls(Policy(),setup[0],time.monotonic()+30)
    with controls.call_context('candidate_0',0): pass
    with controls.call_context('candidate_0',1): pass
    with pytest.raises(SearchFault,match='retry_limit'):
        with controls.call_context('candidate_0',1): pass
