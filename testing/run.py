"""Offline by default. Network and paid calls each require an explicit CLI opt-in."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'production'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['offline','health','provider','smoke'])
    parser.add_argument('--config')
    parser.add_argument('--allow-network', action='store_true')
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--isolated-target', action='store_true', help='confirm target API uses a dedicated test DB')
    args = parser.parse_args()
    if args.mode == 'offline':
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        raise SystemExit(subprocess.call([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(ROOT)],env=env))
    if not args.allow_network or not args.config:
        parser.error('network modes require --allow-network and --config')
    if args.mode in {'provider','smoke'} and not args.allow_paid:
        parser.error('model-calling modes require --allow-paid')
    if args.mode == 'smoke' and not args.isolated_target:
        parser.error('smoke requires --isolated-target; never write test memories to production DB')
    from configuration import load_config
    from aml_api.production.process import quiet
    load_config(args.config)
    quiet()
    from aml_api.production.policy import Policy
    policy = Policy.from_env()
    try:
        if args.mode == 'provider':
            from openai import OpenAI
            from httpx import Client, Timeout
            for kind, model in [('LLM','gpt-4o-mini'),('EMBEDDING','text-embedding-v4')]:
                with OpenAI(api_key=os.environ['AML_'+kind+'_API_KEY'],base_url=os.environ['AML_'+kind+'_BASE_URL'],
                            max_retries=0,http_client=Client(trust_env=False),
                            timeout=Timeout(policy.provider_timeout_seconds, connect=policy.connect_timeout_seconds)) as client:
                    if kind == 'LLM':
                        r = client.chat.completions.create(model=model,messages=[{'role':'user','content':'Reply OK.'}],max_tokens=8)
                        assert r.choices
                    else:
                        r = client.embeddings.create(model=model,input=['synthetic connectivity probe'])
                        assert len(r.data)==1 and r.data[0].embedding
        else:
            import httpx
            base = os.environ['AML_TEST_BASE_URL'].rstrip('/')
            timeout = 30 if args.mode == 'health' else policy.total_seconds + 60
            with httpx.Client(base_url=base, timeout=httpx.Timeout(timeout, connect=min(30,timeout)), trust_env=False) as c:
                if args.mode == 'health':
                    r = c.get('/health')
                    assert r.status_code==200 and r.json()=={'status':'ok'}
                else:
                    from aml_api.schemas import SearchResponse
                    key = {'X-Api-Key':os.environ['AML_MEMORY_SYSTEM_KEY']}
                    uid = 'synthetic-smoke-'+uuid.uuid4().hex
                    body = {'user_id':uid,'session_id':uid,'request_id':uid,
                            'messages':[{'role':'user','content':[{'type':'text','text':'Synthetic test bicycle is blue.'}]}]}
                    a = c.post('/add',json=body,headers=key)
                    b = c.post('/add',json=body,headers=key)
                    assert a.status_code==b.status_code==200 and a.json()==b.json()
                    q = {'user_id':uid,'query':[{'type':'text','text':'synthetic bicycle color'}],'top_k':20}
                    r = c.post('/search',json=q,headers=key)
                    assert r.status_code==200 and set(r.json())=={'data'}
                    value = SearchResponse.model_validate(r.json())
                    assert 0 < len(value.data) <= 20
                    assert len({i.id for i in value.data})==len(value.data)
                    assert all(set(i)<={'id','content','score','created_at'} for i in r.json()['data'])
                    assert c.post('/search',json=q).status_code==401
                    assert c.post('/search',json={**q,'user_id':uid+'-other'},headers=key).json()=={'data':[]}
        print(json.dumps({'test':args.mode,'status':'passed'}))
    except Exception:
        print(json.dumps({'test':args.mode,'status':'failed','code':'probe_failed'}))
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
