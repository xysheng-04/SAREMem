import dataclasses
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from aml_api.config import Settings
from aml_api.schemas import AddRequest
from aml_api.service import RouteAService
from aml_api.store import SQLiteStore
from aml_api.source_time import source_created_at
from aml_api.production.policy import Policy
from aml_api.production.process import StageRunner
from aml_api.production.service import ProductionService
from server import build_app
from faults import FakeEncoder, FakeVLM, FaultBackend

KEY = 'synthetic-test-key-not-a-real-credential'
HEADERS = {'X-Api-Key':KEY}
QUERY = {'user_id':'test-a','query':[{'type':'text','text':'synthetic bicycle'}],'top_k':20}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-placeholder')
    s = dataclasses.replace(Settings.from_env(), storage_path=tmp_path/'test.sqlite3',
                            memory_system_key=KEY, rerank_top_n=0, llm_base_url='https://invalid.test/v1',
                            embedding_base_url='https://invalid.test/v1')
    store = SQLiteStore(s.storage_path)
    add = RouteAService(s, store=store, embedder=FakeEncoder(), vlm=FakeVLM())
    # Initialize fixtures via the SAME HTTP Add entry, exercising idempotency and scope.
    svc = ProductionService(s, add_service=add)
    with TestClient(build_app(settings=s, service=svc), raise_server_exceptions=False) as c:
        for uid, count in [('test-a',4),('test-b',1)]:
            for i in range(count):
                body = {'request_id':uid+str(i),'user_id':uid,'session_id':'fixture',
                        'messages':[{'role':'user','content':[{'type':'text','text':f'synthetic bicycle {i}'}]}]}
                assert c.post('/add',json=body,headers=HEADERS).status_code == 200
                assert c.post('/add',json=body,headers=HEADERS).status_code == 200
    return s, add


def request(setup, backend=None, body=None, headers=None, policy=None, diagnostics=True):
    settings, add = setup
    events = []
    svc = ProductionService(settings, runner=StageRunner(backend or FaultBackend()),
                            policy=policy, diagnostics=events.append if diagnostics else None,
                            add_service=add)
    with TestClient(build_app(settings=settings,service=svc), raise_server_exceptions=False) as c:
        r = c.post('/search', json=QUERY if body is None else body,
                   headers=HEADERS if headers is None else headers)
    assert set(r.headers) <= {'content-type','content-length','retry-after'}
    assert 'MUST_NOT_LEAK' not in r.text
    assert not multiprocessing.active_children()
    return r, events


def check_success(r, settings):
    assert r.status_code == 200, r.text
    assert set(r.json()) == {'data'}
    rows = r.json()['data']
    assert len(rows) <= 20
    assert len({r['id'] for r in rows}) == len(rows)
    assert all(set(row) == {'id','content','score'} for row in rows)
    assert [x['score'] for x in rows] == sorted([x['score'] for x in rows],reverse=True)
    with sqlite3.connect(settings.storage_path) as db:
        for row in rows:
            original = db.execute('SELECT user_id,content_json FROM memories WHERE memory_id=?',(row['id'],)).fetchone()
            assert original[0] == 'test-a'
            assert row['content'] == json.loads(original[1])


def test_simple_success_and_short_result_no_extra_search(setup):
    r,e = request(setup)
    check_success(r,setup[0])
    assert len(r.json()['data']) == 4
    assert [x['path'] for x in e[0]['stages']] == ['simple']


@pytest.mark.parametrize('mode,code', [('provider','provider'),('schema','schema'),('id','candidate_id'),
    ('duplicate_ref','candidate_id'),('sort','ranking'),('membership','membership'),
    ('response_validation','response_validation'),('response_error','response_validation')])
def test_simple_failure_legacy_success(setup, mode, code):
    r,e = request(setup, FaultBackend(simple_mode=mode))
    check_success(r,setup[0])
    assert e[0]['final'] == 'legacy'
    assert e[0]['stages'][0]['code'] == ('provider' if mode in {'provider','schema','id','duplicate_ref'} else code)
    assert len(e[0]['stages']) == 2


def test_both_fail_verified_seed_reuses_original_order(setup):
    good,_ = request(setup)
    r,e = request(setup, FaultBackend(simple_mode='provider',legacy_mode='error'))
    check_success(r,setup[0])
    assert e[0]['final'] == 'seed'
    assert e[0]['stages'][-1]['seed_existing']
    assert [x['id'] for x in r.json()['data']] == [x['id'] for x in good.json()['data']]


def test_no_seed_deterministic_no_network(setup):
    r,e = request(setup, FaultBackend(simple_mode='before_seed',legacy_mode='error'))
    assert r.status_code==503 and len(e[0]['stages'])==1
    assert set(r.json())=={'detail'}


@pytest.mark.parametrize('simple_mode',['timeout','after_seed_timeout'])
def test_deadline_reserves_seed_and_reaps_processes(setup, simple_mode):
    # Spawn includes cold SDK/numpy imports: several seconds on the 2-vCPU target.
    # Keep a short total hard deadline while leaving real seed execution time.
    p = Policy(total_seconds=30, simple_seconds=8, legacy_seconds=7,
               seed_reserve_seconds=10, cleanup_seconds=.5)
    start = time.monotonic()
    r,e = request(setup, FaultBackend(simple_mode=simple_mode, legacy_mode='timeout'), policy=p)
    assert time.monotonic()-start < p.total_seconds + 1
    if simple_mode=='timeout':
        assert r.status_code==504 and len(e[0]['stages'])==1
    else:
        check_success(r,setup[0])
        assert e[0]['final']=='seed'
        assert [x['code'] for x in e[0]['stages'][:2]]==['deadline','deadline']


def raw_rows(settings, user='test-a'):
    with sqlite3.connect(settings.storage_path) as db:
        return [{'id':mid,'content':json.loads(content),'score':1/(i+1),'created_at':source_created_at(created)}
                for i,(mid,content,created) in enumerate(db.execute(
                    'SELECT memory_id,content_json,source_timestamp_ms FROM memories WHERE user_id=? ORDER BY seq',(user,)))]


def test_invalid_duplicate_legacy_candidates_drop_and_backfill(setup):
    rows = raw_rows(setup[0])
    rows.insert(1,rows[0].copy())
    rows.insert(2,{'id':'absent','content':[],'score':.9})
    r,e = request(setup,FaultBackend(simple_mode='after_seed_error',legacy_mode='raw',raw={'data':rows}))
    check_success(r,setup[0])
    assert len(r.json()['data']) == 4
    assert e[0]['stages'][-1]['dropped'] == 2


@pytest.mark.parametrize('mode',['scope_error','raw'])
def test_scope_violation_terminal_no_fallback(setup, mode):
    r,e = request(setup, FaultBackend(simple_mode=mode, raw={'data':raw_rows(setup[0],'test-b')}))
    assert r.status_code == 403
    assert e[0]['final'] == 'error'
    assert len(e[0]['stages']) == 1


@pytest.mark.parametrize('mode',['success','empty'])
def test_empty_is_success_not_retry(setup, mode):
    r,e = request(setup, FaultBackend(simple_mode=mode), body={**QUERY,'user_id':'new-user'})
    assert r.status_code == 200 and r.json() == {'data':[]}
    assert len(e[0]['stages']) == 1


def test_lexical_empty_for_image_only_or_no_match(setup):
    r,e = request(setup,FaultBackend(simple_mode='error',legacy_mode='error'),
                  body={**QUERY,'query':[{'type':'text','text':'zzzznomatch'}]})
    assert r.status_code==503 and set(r.json())=={'detail'}
    assert e[0]['final']=='error' and len(e[0]['stages'])==1


def test_database_unreadable_not_empty_success(setup):
    setup[0].storage_path.write_bytes(b'not a sqlite file')
    r,e = request(setup, FaultBackend(simple_mode='empty'))
    assert r.status_code == 500
    assert len(e[0]['stages']) == 1


@pytest.mark.parametrize('headers,body,status', [({},QUERY,401),
    (HEADERS,{**QUERY,'top_k':0},422), (HEADERS,{**QUERY,'unexpected':'SECRET'},422),
    (HEADERS,{**QUERY,'query':[{'type':'image_url','image_url':{'url':'https://invalid.test/a'}}]},400)])
def test_auth_and_invalid_request_never_fallback(setup,headers,body,status):
    r,e = request(setup, body=body, headers=headers)
    assert r.status_code == status
    assert 'SECRET' not in r.text
    assert e == []


def test_all_stages_fail_clear_503(setup):
    r,e = request(setup,FaultBackend(simple_mode='error',legacy_mode='error',seed_mode='error'))
    assert r.status_code == 503
    assert e[0]['final'] == 'error'
    assert [s['path'] for s in e[0]['stages']] == ['simple']


@pytest.mark.parametrize('k',[1,2,100])
def test_top_k(setup,k):
    r,e = request(setup, body={**QUERY,'top_k':k})
    check_success(r,setup[0])
    assert len(r.json()['data']) <= min(k,20)


@pytest.mark.parametrize('backend',[FaultBackend(),FaultBackend(simple_mode='provider'),
    FaultBackend(simple_mode='empty'),FaultBackend(simple_mode='source_error')])
def test_production_no_diagnostic_artifact(setup,backend,tmp_path):
    before = set(tmp_path.rglob('*'))
    r,e = request(setup,backend,diagnostics=False)
    assert e == []
    assert set(tmp_path.rglob('*')) - before <= {Path(str(setup[0].storage_path)+'-wal'),Path(str(setup[0].storage_path)+'-shm')}
    assert set(r.json()) == ({'data'} if r.status_code==200 else {'detail'})
