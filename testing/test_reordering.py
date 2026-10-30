from pathlib import Path
import sys,types
from unittest.mock import patch
import pytest
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'production'))
from aml_api.simple.reordering import stable_groups
from aml_api.simple.core import run_search
from aml_api.simple.contracts import Config,Assessment
from aml_api.postseed.seed_port import SeedConfig,SeedHit,SeedSnapshot,Binding,digest,content_digest
from aml_api.postseed.contracts import Context,Limits,Failure
from aml_api.schemas import SearchRequest
from contextlib import contextmanager

def test_stable_categories():
 ids=list('abcdef');labels=dict(zip(ids,['unrelated','context','uncertain','direct','context','direct']))
 assert stable_groups(ids,labels)==list('dfbeac')
 assert stable_groups(ids,dict.fromkeys(ids,'direct'))==ids
 assert stable_groups(ids,dict(zip(ids,['unrelated','uncertain']*3)))==ids
 with pytest.raises(Failure):stable_groups(ids+['a'],labels)

def setup_case():
 q=SearchRequest(user_id='synthetic',query=[{'type':'text','text':'synthetic query'}],top_k=20);ids=[f'm{i}' for i in range(80)];raw={mid:[{'type':'text','text':mid}] for mid in ids}
 context=Context(request_nonce=digest('n'),request_digest=digest(q.model_dump(mode='json')),scope_digest=digest(q.user_id),read_view_digest=digest('view'),config_digest=digest('cfg'),code_digest=digest('code'))
 bind=Binding(request_digest=context.request_digest,scope_digest=context.scope_digest,query_digest=digest({'query':[x.model_dump(mode='json') for x in q.query],'options':q.options}),preparation_digest=digest('p'),read_view_digest=context.read_view_digest,producer_digest=digest('producer'))
 snap=SeedSnapshot(binding=bind,config=SeedConfig(retrieve_pool=80,upstream_max_candidates=80),hits=tuple(SeedHit(memory_id=x,rank=i,score=1./i,content_digest=content_digest(raw[x])) for i,x in enumerate(ids,1)))
 class View:
  scope_digest=context.scope_digest;read_view_digest=context.read_view_digest;read_bytes=0;content_reads=0;cache_hits=0
  def contents(self,ids):return {x:raw[x] for x in ids}
  def metadata(self,mid):return {'seq':ids.index(mid),'created_at':None,'source_timestamp_ms':None,'session_id':'s'}
 class Meter:
  limits=Limits();data={'calls':[]}
  @contextmanager
  def phase(self,name):yield
  def count(self,**kw):pass
  def summary(self):return {}
 return q,context,View(),snap,Meter()

@pytest.mark.parametrize('pool',[20,40])
def test_pool_then_select20(pool):
 args=setup_case();calls=[]
 def assess(provider,meter,query,options,parts,metadata,parent):
  i=int(parent.split('_')[-1]);calls.append(i);label='direct' if i==30 else 'context' if i==25 else 'unrelated';return Assessment(relevance=label,fragment_ids=('f0',) if label!='unrelated' else (),possible_conflict=False)
 with patch('aml_api.simple.core.assess',assess):out=run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='stable_groups',assess_pool=pool))
 assert len(calls)==pool and len(out.response.data)==20
 assert [x.id for x in out.response.data][:2]==(['m30','m25'] if pool==40 else ['m0','m1'])
 assert len(out.stages['ranked_ids'])==pool
 for item in out.response.data:assert item.content[0].text==item.id

def test_media_budget_applied_after_group_sort():
 args=setup_case();args[4].limits=Limits(response_image_bytes=20)
 def assess(provider,meter,query,options,parts,metadata,parent):
  i=int(parent.split('_')[-1]);label='direct' if i==30 else 'context' if i==25 else 'unrelated';return Assessment(relevance=label,fragment_ids=('f0',) if label!='unrelated' else (),possible_conflict=False)
 def size(parts):return 21 if parts[0].text=='m30' else len(parts)
 with patch('aml_api.simple.core.assess',assess),patch('aml_api.simple.core.response_image_bytes',size):out=run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='stable_groups',assess_pool=40))
 assert len(out.response.data)==20 and out.response.data[0].id=='m25'
 assert 'm30' not in [x.id for x in out.response.data]
 assert out.stages['media_exclusions']==[{'id':'m30','reason':'response_media_budget'}]

def test_component_can_be_removed_in_bounded_mode(tmp_path):
 import shutil,subprocess,os
 shutil.copytree(B/'production',tmp_path/'production',ignore=shutil.ignore_patterns('__pycache__'))
 (tmp_path/'production/aml_api/simple/reordering.py').unlink()
 env=dict(os.environ,PYTHONPATH=str(tmp_path/'production'),PYTHONDONTWRITEBYTECODE='1')
 p=subprocess.run([sys.executable,'-B','-c',"from aml_api.simple.core import promote; assert promote(['a','b'],{'a':'unrelated','b':'direct'})==['b','a']"],cwd=tmp_path,env=env,capture_output=True)
 assert p.returncode==0,p.stderr

def test_bounded_path_still_selects20():
 args=setup_case();calls=[]
 def assess(provider,meter,query,options,parts,metadata,parent):
  i=int(parent.split('_')[-1]);calls.append(i);return Assessment(relevance='direct' if i==19 else 'unrelated',fragment_ids=('f0',) if i==19 else (),possible_conflict=False)
 with patch('aml_api.simple.core.assess',assess):out=run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='bounded'))
 assert len(calls)==20 and [x.id for x in out.response.data].index('m19')==16
