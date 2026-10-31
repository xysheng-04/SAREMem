import sys,os,json,subprocess,shutil,time
from pathlib import Path
from unittest.mock import patch
import pytest
from test_reordering import setup_case
from aml_api.simple.core import run_search
from aml_api.simple.contracts import Config,Assessment
from aml_api.postseed.contracts import Limits
from aml_api.simple.neighbor_fill import apply
B=Path(__file__).resolve().parents[1]

def enriched():
 q,ctx,view,snap,meter=setup_case()
 catalog={f'm{i}':{'seq':100+i,'session_id':'s','created_at':None,'source_timestamp_ms':None} for i in range(80)}
 catalog['m0']['seq']=10
 for mid,seq,session in [('before',9,'s'),('after',11,'s'),('other_session',8,'other')]:catalog[mid]={'seq':seq,'session_id':session,'created_at':None,'source_timestamp_ms':None}
 view.memories=catalog;view.metadata=lambda mid:catalog[mid];view.contents=lambda ids:{mid:[{'type':'text','text':mid}] for mid in ids}
 return q,ctx,view,snap,meter

def run(args,mode='session',label='direct',**kw):
 calls=[]
 def assess(provider,meter,query,options,parts,metadata,parent):
  i=int(parent.split('_')[-1]);calls.append(i);rel=label if i==0 else 'context' if i==1 else 'unrelated'
  return Assessment(relevance=rel,fragment_ids=('f0',) if rel!='unrelated' else (),possible_conflict=False)
 with patch('aml_api.simple.core.assess',assess):out=run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='stable_groups',assess_pool=40,neighbor_mode=mode),**kw)
 return out,calls

def test_live_core_adds_neighbors_without_model_calls():
 args=enriched();out,calls=run(args);ids=[x.id for x in out.response.data]
 assert ids[:4]==['m0','m1','before','after'] and len(ids)==20 and len(calls)==40
 assert 'other_session' not in ids and len(ids)==len(set(ids))
 assert [x.score for x in out.response.data]==[1./i for i in range(1,21)]
 assert all(x.content[0].text==x.id for x in out.response.data)
 side={x['id']:x for x in out.sidecar['items']};assert side['before']['assessment'] is None and side['before']['assessment_status']=='not_assessed'
 assert set(out.response.model_dump())=={'data'} and all(set(x.model_dump())=={'id','content','score'} for x in out.response.data)

def test_off_exact_original_and_removed_component(tmp_path):
 a,_=run(enriched(),'off');assert [x.id for x in a.response.data]==[f'm{i}' for i in range(20)]
 shutil.copytree(B/'production',tmp_path/'production',ignore=shutil.ignore_patterns('__pycache__'))
 (tmp_path/'production/aml_api/simple/neighbor_fill.py').unlink()
 env=dict(os.environ,PYTHONPATH=str(tmp_path/'production')+os.pathsep+str(B/'testing'),PYTHONDONTWRITEBYTECODE='1')
 script="from test_neighbor_integration import enriched,run;out,calls=run(enriched(),'off');assert len(calls)==40 and [x.id for x in out.response.data]==[f'm{i}' for i in range(20)]"
 p=subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,env=env,capture_output=True);assert p.returncode==0,p.stderr

def test_no_direct_leaves_original_results():
 out,calls=run(enriched(),label='context');assert [x.id for x in out.response.data]==[f'm{i}' for i in range(20)] and len(calls)==40

def test_topk_and_media_budget():
 args=list(enriched());args[0]=args[0].model_copy(update={'top_k':3})
 # Request binding must describe this top_k, as in the real pipeline.
 from aml_api.postseed.seed_port import digest
 args[1]=args[1].model_copy(update={'request_digest':digest(args[0].model_dump(mode='json'))})
 args[3]=args[3].model_copy(update={'binding':args[3].binding.model_copy(update={'request_digest':args[1].request_digest})})
 args[4].limits=Limits(response_image_bytes=3)
 def size(parts):return 4 if parts[0].text=='before' else 1
 with patch('aml_api.simple.core.response_image_bytes',size):out,calls=run(args)
 assert [x.id for x in out.response.data]==['m0','m1','after'] and len(calls)==40
 assert out.stages['neighbor_fill']['quota']==1

from test_faults import setup

def test_real_read_view_preserves_user_and_session_isolation(setup):
 from aml_api.schemas import AddRequest
 from aml_api.simple.source import SQLiteReadView
 from aml_api.production.runtime import Controls
 from aml_api.production.policy import Policy
 settings,add=setup
 add.add(AddRequest.model_validate({'request_id':'extra-session','user_id':'test-a','session_id':'other-session','messages':[{'role':'user','content':[{'type':'text','text':'different session'}]}]}))
 with SQLiteReadView(settings.storage_path,'test-a',Controls(Policy(),settings,time.monotonic()+30)) as view:
  ids=[x for x,m in view.memories.items() if m['session_id']=='fixture'];ids.sort(key=lambda x:view.memories[x]['seq']);anchor=ids[-1]
  out=apply([anchor],{anchor:'direct'},[anchor],view,limit=3,media_limit=100,size_of=lambda _:0)
  assert out['ids']==[anchor,ids[-2]]
  assert all(view.metadata(x)['session_id']=='fixture' for x in out['ids'])
