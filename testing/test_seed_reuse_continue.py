import copy,time,types
from unittest.mock import patch
import pytest
from test_reordering import setup_case
from aml_api.simple.core import run_search
from aml_api.simple.contracts import Config,Assessment
from aml_api.production.policy import SearchFault,Policy
from aml_api.postseed.contracts import Failure

@pytest.mark.parametrize('code',['provider','provider_retry_exhausted','provider_json','unknown_fragment','image_decode'])
def test_candidate_error_continues40_without_fake_label_or_seed_mutation(code):
 args=setup_case();args[2].memories={f'm{i}':args[2].metadata(f'm{i}') for i in range(80)};snapshot=args[3].model_dump_json();calls=[]
 def assess(provider,meter,query,options,parts,metadata,parent):
  index=int(parent.split('_')[-1]);calls.append(index)
  if index==1:
   if code=='image_decode':
    from aml_api.search_images import ImageInputError
    raise ImageInputError(code)
   if code in {'provider_json','unknown_fragment'}:raise Failure(code,'assessment')
   raise SearchFault(code)
  return Assessment(relevance='direct' if index==0 else 'unrelated',fragment_ids=('f0',) if index==0 else (),possible_conflict=False)
 with patch('aml_api.simple.core.assess',assess):
  out=run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='stable_groups',assess_pool=40,neighbor_mode='session',continue_candidate_errors=True))
 assert calls==list(range(40));assert args[3].model_dump_json()==snapshot
 assert out.stages['candidate_errors']==[{'id':'m1','code':code}]
 assert 'm1' not in out.stages['returned_ids'] and out.execution_status=='partial'
 assert set(out.response.model_dump())=={'data'}

def test_all40_fail_only_then_fallback():
 args=setup_case();calls=[]
 def assess(*a):calls.append(a[-1]);raise SearchFault('provider_retry_exhausted')
 with patch('aml_api.simple.core.assess',assess),pytest.raises(Failure,match='assessment_unavailable'):
  run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='stable_groups',assess_pool=40,continue_candidate_errors=True))
 assert len(calls)==40

@pytest.mark.parametrize('code,terminal',[('deadline',False),('capacity',False),('scope_denied',True)])
def test_global_error_not_swallowed(code,terminal):
 args=setup_case();calls=[]
 def assess(*a):calls.append(a[-1]);raise SearchFault(code,terminal=terminal)
 with patch('aml_api.simple.core.assess',assess),pytest.raises(SearchFault):
  run_search(*args[:4],None,args[4],Config(seed=args[3].config,reorder_mode='stable_groups',assess_pool=40,continue_candidate_errors=True))
 assert len(calls)==1

from test_faults import setup
from aml_api.postseed.seed_port import digest,content_digest
from aml_api.schemas import SearchRequest
from aml_api.production.validation import seed_response

def checkpoint(settings,request):
 from aml_api.production.validation import reader,original
 with reader(settings.storage_path) as db:
  ids=[x['memory_id'] for x in db.execute('SELECT memory_id FROM memories WHERE user_id=? ORDER BY seq DESC',(request.user_id,))]
  return {'request':digest(request.model_dump(mode='json')),'hits':[{'id':mid,'hash':content_digest(original(db,request.user_id,mid,1000000)[0]),'score':.8-i*.1} for i,mid in enumerate(ids)]}

def test_legacy_and_seed_reuse_exact_checkpoint_no_retrieve(setup):
 from aml_api.production import runtime
 from aml_api.service import RouteAService
 from faults import FakeEncoder,FakeVLM
 settings,_=setup;request=SearchRequest(user_id='test-a',query=[{'type':'text','text':'synthetic'}],top_k=20)
 seed=checkpoint(settings,request);before=copy.deepcopy(seed)
 class ForbiddenEncoder(FakeEncoder):
  def encode(self,*a):pytest.fail('recovery re-embedded query')
 class ForbiddenVLM(FakeVLM):
  def plan_query(self,*a):pytest.fail('recovery re-planned query')
  def rerank(self,*a):pytest.fail('recovery reranked')
 def service(s,store):return RouteAService(s,store=store,embedder=ForbiddenEncoder(),vlm=ForbiddenVLM())
 with patch.object(runtime,'RouteAService',service),patch('aml_api.service.hybrid_rank',side_effect=AssertionError('re-retrieval')):
  out=runtime.legacy(settings,request,Policy(),time.monotonic()+20,None,{},checkpoint=seed)
  assert out.data
 rows=seed_response(settings,request,Policy(),seed,{})['data']
 assert [x['id'] for x in rows]==[x['id'] for x in seed['hits']]
 assert [x['score'] for x in rows]==[x['score'] for x in seed['hits']]
 assert seed==before

@pytest.mark.parametrize('stage',['legacy','seed'])
def test_no_seed_never_fabricates_retrieval(setup,stage):
 from aml_api.production.process import execute
 settings,_=setup;request=SearchRequest(user_id='test-a',query=[{'type':'text','text':'synthetic'}],top_k=20)
 with patch('aml_api.retrieval.lexical_rank',side_effect=AssertionError('new lexical retrieval')),pytest.raises(SearchFault,match='seed_unavailable'):
  execute(stage,settings,request,Policy(),time.monotonic()+10,None,None,{})

def test_seed_failure_no_fallback_or_empty_success(setup):
 from aml_api.production.service import ProductionService
 from aml_api.errors import UpstreamUnavailable
 settings,_=setup;calls=[]
 class Runner:
  def run(self,stage,*a):calls.append(stage);return {'error':'provider'}
 request=SearchRequest(user_id='test-a',query=[{'type':'text','text':'synthetic'}],top_k=20)
 with pytest.raises(UpstreamUnavailable):ProductionService(settings,runner=Runner()).search(request)
 assert calls==['simple']

class OneBad40:
 def __call__(self,stage,settings,request,policy,deadline,checkpoint,publish,stats):
  assert stage=='simple','single-candidate failure caused global degradation'
  from aml_api.production import runtime
  from aml_api.production.process import execute
  from faults import FakeFrozen
  class Provider:
   def __init__(self,*a):self.calls=0
   def call(self,*a):
    self.calls+=1;stats['test_assessment_calls']=self.calls
    if self.calls==8:raise SearchFault('provider_retry_exhausted')
    return {'relevance':'context','fragment_ids':['f0'],'possible_conflict':False}
   def close(self):pass
  runtime.FrozenProviders=FakeFrozen;runtime.AssessmentProvider=Provider
  return execute(stage,settings,request,policy,deadline,checkpoint,publish,stats)

def test_real_http_completes40_after_one_candidate_error(setup):
 from aml_api.schemas import AddRequest
 from test_faults import request,check_success
 settings,add=setup
 for i in range(36):
  add.add(AddRequest(request_id=f'extra-{i}',user_id='test-a',session_id='fixture',messages=[{'role':'user','content':[{'type':'text','text':f'synthetic bicycle extra {i}'}]}]))
 r,e=request(setup,OneBad40());check_success(r,settings)
 assert e[0]['final']=='simple' and len(e[0]['stages'])==1
 stage=e[0]['stages'][0];assert stage['test_assessment_calls']==40 and len(stage['candidate_errors'])==1
 assert stage['candidate_errors'][0]['id'] not in {row['id'] for row in r.json()['data']}
 assert set(r.json())=={'data'}

@pytest.mark.parametrize('mutation',['hash','duplicate','score'])
def test_bad_checkpoint_never_retrieves_or_succeeds(setup,mutation):
 settings,_=setup;request=SearchRequest(user_id='test-a',query=[{'type':'text','text':'synthetic'}],top_k=20)
 seed=checkpoint(settings,request)
 if mutation=='hash':seed['hits'][0]['hash']='0'*64
 if mutation=='duplicate':seed['hits'][1]=seed['hits'][0].copy()
 if mutation=='score':seed['hits'][1]['score']=999
 with pytest.raises(SearchFault) as err:seed_response(settings,request,Policy(),seed,{})
 assert err.value.terminal


def test_invalid_query_image_is400_without_retrieval_or_fallback(setup):
 from test_faults import request,QUERY
 from faults import FaultBackend
 body={**QUERY,'query':[{'type':'image_url','image_url':{'url':'data:image/png;base64,YmFk'}}]}
 r,e=request(setup,FaultBackend(),body=body)
 assert r.status_code==400 and len(e[0]['stages'])==1
 assert e[0]['stages'][0]['code']=='query_image_invalid'
 assert set(r.json())=={'detail'}
