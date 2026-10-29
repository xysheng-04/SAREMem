import base64
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from aml_api.production.service import ProductionService
from aml_api.production.process import StageRunner
from aml_api.production.policy import Policy
from server import build_app
from faults import FaultBackend
from test_faults import setup, request, check_success, raw_rows, HEADERS, QUERY


@pytest.mark.parametrize('mode,path', [('sdk_no_usage','simple'),('sdk_timeout','legacy'),('sdk_bad_json','legacy')])
def test_real_sdk_boundary_without_usage_gate(setup,mode,path):
    r,e = request(setup,FaultBackend(simple_mode=mode))
    check_success(r,setup[0])
    assert e[0]['final'] == path


def test_legacy_response_validation_failure_recovers_seed(setup):
    r,e = request(setup,FaultBackend(simple_mode='provider',legacy_mode='response_error'))
    check_success(r,setup[0])
    assert e[0]['final']=='seed'


def test_duplicate_simple_id_and_score_order_failures(setup):
    for kind in ['duplicate','score']:
        rows = raw_rows(setup[0])
        if kind=='duplicate':
            rows.insert(1,rows[0])
        else:
            rows[1]['score']=2
        r,e = request(setup,FaultBackend(simple_mode='raw',raw={'data':rows}))
        check_success(r,setup[0])
        assert e[0]['final']=='legacy'


def test_invalid_media_skipped_without_changing_original(setup):
    s,_ = setup
    with sqlite3.connect(s.storage_path) as db:
        mid = db.execute('SELECT memory_id FROM memories WHERE user_id=? ORDER BY seq LIMIT 1',('test-a',)).fetchone()[0]
        db.execute('UPDATE memories SET content_json=? WHERE memory_id=?',
                   (json.dumps([{'type':'image_url','image_url':{'url':'data:image/png;base64,!!!'}}]),mid))
    r,e = request(setup,FaultBackend(simple_mode='error',legacy_mode='raw',raw={'data':raw_rows(s)}))
    check_success(r,s)
    assert len(r.json()['data'])==3
    assert e[0]['stages'][-1]['media_filtered']


def test_real_media_limit_and_backfill_http(setup):
    s,add = setup
    with TestClient(build_app(settings=s,service=ProductionService(s,add_service=add))) as c:
        assert c.post('/add',headers=HEADERS,json={'request_id':'fifth','user_id':'test-a','session_id':'fixture',
            'messages':[{'role':'user','content':[{'type':'text','text':'synthetic bicycle text-only tail'}]}]}).status_code==200
    uri = 'data:image/png;base64,'+base64.b64encode(b'x'*(8*1024*1024)).decode()
    with sqlite3.connect(s.storage_path) as db:
        ids = [r[0] for r in db.execute('SELECT memory_id FROM memories WHERE user_id=? ORDER BY seq',('test-a',))]
        for mid in ids[:4]:
            db.execute('UPDATE memories SET content_json=? WHERE memory_id=?',
                       (json.dumps([{'type':'image_url','image_url':{'url':uri}}]),mid))
    r,e = request(setup,FaultBackend(simple_mode='error',legacy_mode='raw',raw={'data':raw_rows(s)}))
    check_success(r,s)
    assert [x['id'] for x in r.json()['data']] == ids[:3]+ids[4:]
    assert e[0]['stages'][-1]['media_filtered']
    assert e[0]['stages'][-1]['dropped']==1
    assert 'base64' not in json.dumps(e)


def test_openapi_unchanged_search_contract(setup):
    from aml_api.schemas import SearchResponse, SearchItem
    s,add=setup
    app=build_app(settings=s,service=ProductionService(s,add_service=add))
    spec=app.openapi()
    assert set(spec['paths'])=={'/add','/search','/health'}
    assert spec['components']['schemas']['SearchResponse']['properties']=={'data':{'items':{'$ref':'#/components/schemas/SearchItem'},'type':'array','title':'Data'}}
    assert set(spec['components']['schemas']['SearchItem']['properties'])=={'id','content','score','created_at'}
    assert not app.debug


def test_standalone_production_tree_no_tests_or_eval_import(tmp_path):
    root=Path(__file__).resolve().parents[1]
    target=tmp_path/'fresh-production'
    shutil.copytree(root/'production',target,ignore=shutil.ignore_patterns('__pycache__'))
    # No test or evaluation package exists beside the runtime.
    script='''
import os,sys,tempfile
from pathlib import Path
from aml_api.store import SQLiteStore
p=Path('state/new.sqlite3').resolve(); SQLiteStore(p)
os.environ.update(AML_STORAGE_PATH=str(p),AML_SEARCH_PIPELINE='simple',
 AML_MEMORY_SYSTEM_KEY='synthetic-standalone-key-0000',AML_LLM_API_KEY='offline-placeholder',
 AML_EMBEDDING_API_KEY='offline-placeholder',AML_LLM_BASE_URL='https://invalid.test/v1',
 AML_EMBEDDING_BASE_URL='https://invalid.test/v1')
import socket
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('startup network'))
from server import build_app
app=build_app()
assert app and not app.debug
assert not any(n.startswith(('pytest','testing','eval','aml_api.simple.answer','aml_api.simple.runner')) for n in sys.modules)
assert not Path('testing').exists()
'''
    r=subprocess.run([sys.executable,'-B','-c',script],cwd=target,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert not r.stdout and not r.stderr


def test_real_tcp_formal_launcher_no_startup_calls(tmp_path):
    import socket,time,httpx
    root=Path(__file__).resolve().parents[1]
    target=tmp_path/'production'
    shutil.copytree(root/'production',target,ignore=shutil.ignore_patterns('__pycache__'))
    db=tmp_path/'persist/new.sqlite3'
    r=subprocess.run([sys.executable,'-B',str(target/'init_db.py'),'--path',str(db)],capture_output=True)
    assert r.returncode==0
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    config=tmp_path/'test.env'
    config.write_text('\n'.join(['AML_SEARCH_PIPELINE=simple','AML_STORAGE_PATH='+str(db),
        'AML_MEMORY_SYSTEM_KEY=synthetic-tcp-secret-000000','AML_LLM_API_KEY=offline-placeholder',
        'AML_EMBEDDING_API_KEY=offline-placeholder','AML_LLM_BASE_URL=http://127.0.0.1:1/v1',
        'AML_EMBEDDING_BASE_URL=http://127.0.0.1:1/v1','AML_BIND_HOST=127.0.0.1','AML_PORT='+str(port)]))
    proc=subprocess.Popen([sys.executable,'-B',str(target/'server.py'),'--config',str(config)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',trust_env=False) as c:
            for _ in range(60):
                try:
                    health=c.get('/health');break
                except httpx.ConnectError:
                    time.sleep(.05)
            assert health.json()=={'status':'ok'}
            assert c.post('/search',json=QUERY).status_code==401
            r=c.post('/search',json=QUERY,headers={'X-Api-Key':'synthetic-tcp-secret-000000'})
            assert r.status_code==200 and r.json()=={'data':[]}
            assert set(r.headers)<={'date','content-length','content-type'}
    finally:
        proc.terminate()
        out,err=proc.communicate(timeout=8)
    assert out==b'' and err==b''
    assert not list(target.rglob('*.jsonl'))


def test_no_auth_startup_rejected(setup):
    with pytest.raises(ValueError):
        build_app(settings=dataclasses.replace(setup[0],memory_system_key=None))


def test_network_cli_never_implicit():
    script=Path(__file__).with_name('run.py')
    for mode in ['provider','smoke','health']:
        r=subprocess.run([sys.executable,'-B',str(script),mode],capture_output=True)
        assert r.returncode==2


def test_policy_capacity_prevents_invalid_deadline():
    with pytest.raises(ValueError):
        Policy(total_seconds=20)


def test_add_failure_cannot_fake_success(setup):
    s,add=setup
    class Broken:
        def add(self,request):
            raise RuntimeError('MUST_NOT_LEAK')
    svc=ProductionService(s,add_service=Broken())
    with TestClient(build_app(settings=s,service=svc),raise_server_exceptions=False) as c:
        r=c.post('/add',headers=HEADERS,json={'request_id':'fails','user_id':'test-a','session_id':'fixture',
             'messages':[{'role':'user','content':[{'type':'text','text':'synthetic'}]}]})
    assert r.status_code==500 and 'MUST_NOT_LEAK' not in r.text


@pytest.mark.parametrize('backend',[FaultBackend(),FaultBackend(simple_mode='error'),
    FaultBackend(simple_mode='error',legacy_mode='error')])
def test_actual_twenty_item_ceiling_each_path(setup,backend):
    s,add=setup
    with TestClient(build_app(settings=s,service=ProductionService(s,add_service=add))) as c:
        for i in range(24):
            assert c.post('/add',headers=HEADERS,json={'request_id':'more'+str(i),'user_id':'test-a','session_id':'fixture',
                'messages':[{'role':'user','content':[{'type':'text','text':'synthetic bicycle extra '+str(i)}]}]}).status_code==200
    r,e=request(setup,backend,body={**QUERY,'top_k':100})
    check_success(r,s)
    assert len(r.json()['data'])==20


def test_archive_manifest_and_clean_allowlist(tmp_path):
    import tarfile
    root=Path(__file__).resolve().parents[1]
    archive=tmp_path/'candidate.tar.gz'
    result=subprocess.run([sys.executable,'-B',str(root/'build_candidate.py'),'--output',str(archive)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    info=json.loads(result.stdout)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['archive_sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        tar.extractall(tmp_path/'unpacked',filter='data')
    unpacked=tmp_path/'unpacked'/json.loads((root/'VERSION.json').read_text())['version']
    manifest=json.loads((unpacked/'MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((unpacked/name).read_bytes()).hexdigest()==expected
        assert not any(word in Path(name).parts for word in ['outputs','state','__pycache__','.venv','eval'])
        assert not name.endswith(('.sqlite3','.sqlite','.db','.log','.jsonl'))
    assert manifest['identity']==info['identity']
    import re
    assert (unpacked/'LICENSE').read_text().startswith('MIT License')
    for name in manifest['files']:
        assert not re.search(rb'v[-_ ]?mem', (unpacked/name).read_bytes(), re.I), name
    # The extracted distribution contains the independent production tree.
    assert (unpacked/'production/server.py').is_file()
    assert not (unpacked/'production/aml_api/simple/runner.py').exists()
    assert not (unpacked/'production/aml_api/simple/answer.py').exists()
    (unpacked/'testing').rename(tmp_path/'detached-testing')
    script='''
import os,sys,socket
from pathlib import Path
from aml_api.store import SQLiteStore
p=Path('fresh.sqlite3').resolve();SQLiteStore(p)
os.environ.update(AML_STORAGE_PATH=str(p),AML_SEARCH_PIPELINE='simple',
 AML_MEMORY_SYSTEM_KEY='synthetic-unpacked-key-0000',AML_LLM_API_KEY='offline-placeholder',
 AML_EMBEDDING_API_KEY='offline-placeholder',AML_LLM_BASE_URL='https://invalid.test/v1',
 AML_EMBEDDING_BASE_URL='https://invalid.test/v1')
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('startup network'))
from server import build_app
from fastapi.testclient import TestClient
with TestClient(build_app()) as client:
 assert client.get('/health').json()=={'status':'ok'}
assert not any(n.startswith(('pytest','testing','eval','aml_api.simple.answer')) for n in sys.modules)
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=unpacked/'production',capture_output=True,text=True)
    assert result.returncode==0,result.stderr
