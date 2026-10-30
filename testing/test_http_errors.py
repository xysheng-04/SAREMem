"""Official error table over the real production HTTP adapter, without providers."""
import dataclasses
import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from aml_api.config import Settings
from aml_api.errors import BadRequest, Conflict, Forbidden, ServiceError
from aml_api.production.http_errors import CapacityExceeded
from aml_api.production.policy import SearchFault
from aml_api.production.service import ProductionService
from server import build_app
from test_faults import setup, HEADERS, QUERY

ADD = {'request_id': 'synthetic-error-audit', 'user_id': 'test-a', 'session_id': 'fixture',
       'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'synthetic memory'}]}]}
SECRET = 'MUST_NOT_LEAK_SECRET_PAYLOAD'


class NoCalls:
    def add(self, request):
        pytest.fail('invalid or unauthenticated input reached Add')

    def search(self, request):
        pytest.fail('invalid or unauthenticated input reached Search')


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-placeholder')
    return dataclasses.replace(Settings.from_env(), storage_path=tmp_path/'unused.sqlite3',
                               memory_system_key=HEADERS['X-Api-Key'])


@pytest.mark.parametrize('path,body', [('/add', ADD), ('/search', QUERY)])
@pytest.mark.parametrize('headers', [{}, {'X-Api-Key': 'incorrect'},
    {'X-Api-Key': b'\xe9'}, {'Authorization': b'Bearer \xff'},
    {'Authorization': 'Basic invalid'}, {'Authorization': 'Bearer'}])
def test_invalid_credentials_are_401_never_500(settings, path, body, headers):
    with TestClient(build_app(settings=settings, service=NoCalls())) as client:
        response = client.post(path, json=body, headers=headers)
    assert response.status_code == 401
    assert response.json() == {'detail': {'reason': 'invalid Memory System Key'}}


@pytest.mark.parametrize('scheme', ['X-Api-Key', 'Bearer', 'Token'])
def test_supported_auth_schemes_reach_the_authorized_service(settings, scheme):
    class Empty:
        def search(self, request):
            return {'data': []}
    headers = HEADERS if scheme == 'X-Api-Key' else {'Authorization': scheme+' '+HEADERS['X-Api-Key']}
    with TestClient(build_app(settings=settings, service=Empty())) as client:
        response = client.post('/search', json=QUERY, headers=headers)
    assert response.status_code == 200 and response.json() == {'data': []}


def test_422_has_precise_missing_fields_and_nested_safe_paths(settings):
    with TestClient(build_app(settings=settings, service=NoCalls())) as client:
        response = client.post('/add', json={}, headers=HEADERS)
        assert response.status_code == 422
        assert {tuple(e['loc']) for e in response.json()['detail']} == {
            ('body', 'request_id'), ('body', 'messages'), ('body', 'user_id'), ('body', 'session_id')}
        body = {**ADD, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {}}]}]}
        response = client.post('/add', json=body, headers=HEADERS)
        assert response.status_code == 422
        assert ['body', 'messages', 0, 'content', 0, 'image_url', 'image_url', 'url'] in [
            e['loc'] for e in response.json()['detail']]


@pytest.mark.parametrize('body', [{**QUERY, 'top_k': 0}, {**QUERY, 'top_k': 101},
    {**QUERY, 'top_k': SECRET}, {**QUERY, 'query': []}, {**QUERY, SECRET: SECRET},
    {**QUERY, 'query': [{'type': SECRET, 'text': SECRET}]},
    {**QUERY, 'query': [{'type': 'text', 'text': {'secret': SECRET}}]}])
def test_422_never_echoes_input_tags_keys_context_or_values(settings, body):
    with TestClient(build_app(settings=settings, service=NoCalls())) as client:
        response = client.post('/search', json=body, headers=HEADERS)
    assert response.status_code == 422 and SECRET not in response.text
    for error in response.json()['detail']:
        assert set(error) == {'type', 'loc', 'msg'} and error['loc'][0] == 'body'


def test_malformed_json_does_not_echo_content(settings):
    with TestClient(build_app(settings=settings, service=NoCalls())) as client:
        response = client.post('/search', content='{"'+SECRET+'":',
                               headers={**HEADERS, 'Content-Type': 'application/json'})
    assert response.status_code == 422 and SECRET not in response.text
    assert response.json()['detail'][0]['type'] == 'json_invalid'


@pytest.mark.parametrize('code', [400, 403, 408, 409, 425, 429, 500, 502, 503, 504, 524])
@pytest.mark.parametrize('path,body', [('/add', ADD), ('/search', QUERY)])
def test_error_statuses_have_safe_envelopes_and_bounded_retry_headers(settings, code, path, body):
    class Error(ServiceError):
        status_code = code
    class Broken:
        def add(self, request):
            raise Error(SECRET, retry_after=1)
        search = add
    with TestClient(build_app(settings=settings, service=Broken()), raise_server_exceptions=False) as client:
        response = client.post(path, json=body, headers=HEADERS)
    assert response.status_code == code and SECRET not in response.text
    assert set(response.json()) == {'detail'} and set(response.json()['detail']) == {'reason'}
    assert ('retry-after' in response.headers) == (code in {408, 425, 429, 502, 503, 504, 524})


@pytest.mark.parametrize('value', [SECRET, -1, 0, 61, True])
def test_invalid_retry_after_is_never_reflected(settings, value):
    class Busy:
        def search(self, request):
            raise CapacityExceeded(SECRET, retry_after=value)
    with TestClient(build_app(settings=settings, service=Busy())) as client:
        response = client.post('/search', json=QUERY, headers=HEADERS)
    assert response.status_code == 429 and 'retry-after' not in response.headers
    assert SECRET not in response.text


def test_routes_return_standard_safe_404_and_405(settings):
    with TestClient(build_app(settings=settings, service=NoCalls())) as client:
        missing = client.get('/'+SECRET)
        method = client.get('/add')
    assert missing.status_code == 404 and method.status_code == 405
    assert method.headers['Allow'] == 'POST'
    assert SECRET not in missing.text
    assert all(isinstance(r.json()['detail']['reason'], str) for r in [missing, method])


def test_capacity_is_429_before_workers_or_provider(settings):
    class NeverRun:
        def run(self, *args):
            pytest.fail('capacity rejection started a worker')
    service = ProductionService(settings, runner=NeverRun())
    for _ in range(service.policy.max_concurrent_searches):
        assert service.slots.acquire(blocking=False)
    with TestClient(build_app(settings=settings, service=service)) as client:
        response = client.post('/search', json=QUERY, headers=HEADERS)
    assert response.status_code == 429 and response.headers['Retry-After'] == '1'
    assert not settings.storage_path.exists()


@pytest.mark.parametrize('fault,status', [('deadline', 504), ('provider', 503)])
def test_exhausted_search_has_correct_error_and_releases_capacity(settings, fault, status):
    class FailedStages:
        calls = 0
        def run(self, *args):
            self.calls += 1
            raise SearchFault(fault)
    runner = FailedStages()
    service = ProductionService(settings, runner=runner)
    with TestClient(build_app(settings=settings, service=service)) as client:
        response = client.post('/search', json=QUERY, headers=HEADERS)
    assert response.status_code == status and runner.calls == 1
    assert service.slots.acquire(blocking=False)
    assert not settings.storage_path.exists()


def test_add_replay_conflict_and_processing_use_same_real_http_add(setup):
    settings, add = setup
    with TestClient(build_app(settings=settings, service=ProductionService(settings, add_service=add))) as client:
        first = client.post('/add', json=ADD, headers=HEADERS)
        replay = client.post('/add', json=ADD, headers=HEADERS)
        assert first.status_code == 200 and first.json() == replay.json()
        assert first.json() == {'success': True, 'request_id': ADD['request_id'],
                                'user_id': ADD['user_id'], 'session_id': ADD['session_id']}
        changed = {**ADD, 'session_id': 'changed'}
        response = client.post('/add', json=changed, headers=HEADERS)
        assert response.status_code == 409 and 'different payload' in response.json()['detail']['reason']
        pending = {**ADD, 'request_id': 'synthetic-pending'}
        from aml_api.schemas import AddRequest
        from aml_api.store import payload_digest
        digest = payload_digest(AddRequest.model_validate(pending).model_dump(mode='json'))
        add.store.claim_add(pending['request_id'], digest)
        response = client.post('/add', json=pending, headers=HEADERS)
        assert response.status_code == 409 and 'still processing' in response.json()['detail']['reason']


@pytest.mark.parametrize('kind', ['exception', 'bad_add', 'bad_search', 'wrong_id'])
def test_internal_errors_and_invalid_success_bodies_cannot_be_http_200(settings, kind):
    class Broken:
        def add(self, request):
            if kind == 'exception':
                raise RuntimeError(SECRET)
            if kind == 'wrong_id':
                return {'success': True, 'request_id': SECRET,
                        'user_id': request.user_id, 'session_id': request.session_id}
            return {'success': False, 'request_id': SECRET}
        def search(self, request):
            return {'data': [{'id': SECRET}]}
    with TestClient(build_app(settings=settings, service=Broken()), raise_server_exceptions=False) as client:
        response = client.post('/search' if kind == 'bad_search' else '/add',
                               json=QUERY if kind == 'bad_search' else ADD, headers=HEADERS)
    assert response.status_code == 500 and SECRET not in response.text
