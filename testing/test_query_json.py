import json,types
import pytest
from aml_api.query_json import parse_query_plan
from aml_api.providers import VisionLanguageProvider
from aml_api.errors import UpstreamUnavailable
PLAN={'intent':'synthetic need','visual_description':'','text_anchors':['anchor','a {b}'],'keywords':['name','date']}
RAW=json.dumps(PLAN)
@pytest.mark.parametrize('raw',[RAW,' \n'+RAW+'\n ', '```json\n'+RAW+'\n```','```\n'+RAW+'\n```','Background only.\n'+RAW,RAW+' trailing','Background\n```json\n'+RAW+'\n```\nEnd.'])
def test_complete_shapes_preserve_values(raw):assert parse_query_plan(raw)==PLAN
@pytest.mark.parametrize('raw',['',None,'prefix '+RAW+' '+RAW,json.dumps([PLAN]),json.dumps({'outer':PLAN}),'```json\n'+RAW[:-2]+'\n```','prefix '+RAW[:-2],'{broken '+RAW,'prefix '+json.dumps({'outer':PLAN}),'prefix [ '+RAW+' ]','prefix {\"outer\":'+RAW,json.dumps({**PLAN,'extra':'x'}),json.dumps({k:v for k,v in PLAN.items() if k!='intent'}),json.dumps({**PLAN,'text_anchors':'anchor'}),json.dumps({**PLAN,'keywords':[1]}),'prefix '+RAW[:-1]+',"intent":"duplicate"}',json.dumps({**PLAN,'keywords':None})])
def test_invalid_ambiguous_or_wrong_fields_rejected(raw):
 with pytest.raises(ValueError):parse_query_plan(raw)

def provider(contents):
 v=VisionLanguageProvider.__new__(VisionLanguageProvider);v.model='gpt-4o-mini';v.sent=[]
 def create(**kwargs):
  v.sent.append(kwargs);text=contents[min(len(v.sent)-1,len(contents)-1)]
  return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text),finish_reason='stop')],usage=None)
 v.client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
 return v

def test_planner_recovers_prefix_without_another_send_and_keeps_image_input():
 v=provider(['preamble\n'+RAW]);result=v.plan_query('synthetic question',[],['data:image/jpeg;base64,c3ludGhldGlj'])
 assert result==PLAN and len(v.sent)==1 and v.sent[0]['max_tokens']==900
 assert v.sent[0]['messages'][1]['content'][-1]['image_url']=={'url':'data:image/jpeg;base64,c3ludGhldGlj'}

def test_parser_failure_uses_existing_three_attempts():
 v=provider(['prefix '+RAW+' '+RAW])
 with pytest.raises(UpstreamUnavailable):v.plan_query('q',[],[])
 assert [x['max_tokens'] for x in v.sent]==[900,1800,3600]

def test_generic_nonplanner_json_unchanged():
 v=provider(['{"description":"synthetic"}']);assert v._json(system='s',content=[])=={'description':'synthetic'}
 v=provider(['prefix {"description":"synthetic"}'])
 with pytest.raises(UpstreamUnavailable):v._json(system='s',content=[])
 assert len(v.sent)==3

def test_keywords_string_preserved():
 p={**PLAN,'keywords':'names and dates'};assert parse_query_plan('prefix '+json.dumps(p))==p

def test_frozen_legacy_seed_transport_uses_compatibility():
 import httpx
 from aml_api.seed_upstream.transport import FrozenProviders
 from aml_api.legacy_seed.source import query_texts
 sends=[];states=[]
 def respond(request):
  sends.append(json.loads(request.content))
  return httpx.Response(200,json={'id':'synthetic','object':'chat.completion','created':0,'model':'gpt-4o-mini','choices':[{'index':0,'message':{'role':'assistant','content':'prefix '+RAW},'finish_reason':'stop'}]})
 class Observer:
  def reserve(self,kind,index):return index
  def finish(self,token,outcome,status):states.append((token,outcome,status))
 p=FrozenProviders(base_url='https://invalid.test/v1',api_key='synthetic',transport=httpx.MockTransport(respond),observer=Observer())
 try:
  texts=query_texts('q',[],[],p);assert 'synthetic need' in texts and len(sends)==1 and states==[(1,'response',200)]
 finally:p.close()
