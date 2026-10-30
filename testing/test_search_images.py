import base64,io,time,json,types,copy
import pytest
from PIL import Image
from aml_api.search_images import Images,ImageInputError,scope,prepare,ACTIVE

def uri(size=(30,20),fmt='PNG',mode='RGB',exif=None):
 im=Image.new(mode,size,(25,70,120,80) if mode=='RGBA' else 90 if mode in {'L','P','CMYK'} else (25,70,120))
 b=io.BytesIO();im.save(b,format=fmt,**({'exif':exif} if exif else {}))
 return 'data:image/'+{'JPEG':'jpeg','PNG':'png','WEBP':'webp'}[fmt]+';base64,'+base64.b64encode(b.getvalue()).decode()

def decode(u):return Image.open(io.BytesIO(base64.b64decode(u.split(',')[1])))
def images(**kw):return Images(time.monotonic()+20,**kw)

@pytest.mark.parametrize('fmt',['JPEG','PNG','WEBP'])
def test_small_unchanged(fmt):
 u=uri(fmt=fmt);assert images().image(u)==u

@pytest.mark.parametrize('size',[(5000,3000),(100,6000),(2200,2200)])
def test_resize_bounds_ratio_full_frame(size):
 x=images();u=x.image(uri(size));im=decode(u)
 assert max(im.size)<=2048 and im.width*im.height<=2097152
 assert abs(im.width/im.height-size[0]/size[1])<.02
 assert im.getpixel((0,0))==im.getpixel((im.width-1,im.height-1))
 assert x.records[0]['resized'] and x.records[0]['input_sha256']!=x.records[0]['output_sha256']

def test_transparency():
 im=decode(images().image(uri((3000,2000),mode='RGBA')))
 assert im.mode=='RGBA' and im.getpixel((0,0))[3]==80

def test_exif():
 exif=Image.Exif();exif[274]=6
 im=decode(images().image(uri((40,20),'JPEG',exif=exif)))
 assert im.size==(20,40) and im.getexif().get(274,1)==1

def test_cmyk():assert decode(images().image(uri(fmt='JPEG',mode='CMYK'))).mode=='RGB'

def test_bad_and_pixel_limit():
 for u in ['data:image/png;base64,????',uri((8000,8000))]:
  with pytest.raises(ImageInputError):images().image(u)

def test_off_identity_and_order():
 blocks=[{'type':'text','text':'first'},{'type':'image_url','image_url':{'url':'invalid','detail':'high'}},{'type':'text','text':'last'}]
 assert images(mode='off').content(blocks)==blocks

def test_binding_request_cache_and_deadline():
 u=uri((3000,2000));x=images();assert x.image(u)==x.image(u) and len(x.records)==1
 assert images(max_edge=1000).binding!=x.binding
 with pytest.raises(TimeoutError):Images(time.monotonic()-1).image(u)
 with scope(time.monotonic()+20) as x:
  x.image(u)
 assert ACTIVE.get() is None and not x.cache

def test_planner_and_legacy_rerank_retries(monkeypatch):
 from aml_api.providers import VisionLanguageProvider
 from aml_api.production import process
 from aml_api.production import runtime
 sent=[]
 v=VisionLanguageProvider.__new__(VisionLanguageProvider);v.model=v.reranker_model='gpt-4o-mini'
 plan={'intent':'a','visual_description':'b','text_anchors':[],'keywords':[]}
 def create(**kw):
  sent.append(kw);content=json.dumps(plan if len(sent)<3 else {'scores':[]})
  if len(sent)==1:content='invalid'
  return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content),finish_reason='stop')],usage=None)
 v.client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
 u=uri((3200,2000));original=[{'type':'image_url','image_url':{'url':u}}];before=copy.deepcopy(original)
 def fake(*args,**kwargs):
  v.plan_query('q',[],[u]);v.rerank('q',[u],[{'id':'a','text':'a','images':[u]}]);return 'ok'
 monkeypatch.setattr(runtime,'legacy',fake)
 stats={};assert process.execute('legacy',None,None,None,time.monotonic()+20,None,None,stats)=='ok'
 assert len(stats['image_derivatives'])==1 and original==before
 assert sent[0]['messages']==sent[1]['messages']
 urls=[b['image_url']['url'] for s in sent for b in s['messages'][1]['content'] if b['type']=='image_url']
 assert urls and all(x!=u and max(decode(x).size)<=2048 for x in urls)

def test_assessment_query_memory_retry(monkeypatch):
 from aml_api.production.runtime import AssessmentProvider,Controls
 from aml_api.production.policy import Policy
 from aml_api.production.retry import AttemptBudget
 import httpx
 sent=[];u=uri((3200,2000));v=uri((3000,1000))
 controls=Controls(Policy(),types.SimpleNamespace(return_limit=20),time.monotonic()+20);controls.operation=AttemptBudget()
 p=AssessmentProvider.__new__(AssessmentProvider);p.controls=controls
 def create(**kw):
  sent.append(kw)
  if len(sent)==1:raise httpx.ReadTimeout('synthetic')
  return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content='{}'))],usage=None)
 p.client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
 payload={'query':[{'type':'image_url','image_url':{'url':u,'detail':'high'}}], 'memory':[{'type':'image_url','image_url':{'url':v}}]};original=copy.deepcopy(payload)
 with scope(time.monotonic()+20) as x:
  assert p.call('assessment',payload,{})=={}
  assert len(x.records)==2
 assert payload==original and sent[0]['messages']==sent[1]['messages']
 assert [b['type'] for b in sent[0]['messages'][1]['content']]==['text','text','image_url','text','image_url']

def test_decode_error_terminal():
 from aml_api.production.policy import classify
 fault=classify(ImageInputError('image_decode'));assert fault.terminal and fault.code=='image_decode'

from test_faults import setup

def test_add_database_http_keep_original_image(setup):
 from aml_api.schemas import AddRequest
 from test_faults import request,check_success
 from faults import FaultBackend
 import hashlib
 settings,add=setup
 u=uri((3200,2000))
 add.add(AddRequest.model_validate({'request_id':'image-original','user_id':'test-a','session_id':'fixture','messages':[{'role':'user','content':[{'type':'text','text':'synthetic bicycle image'},{'type':'image_url','image_url':{'url':u}}]}]}))
 before=hashlib.sha256(settings.storage_path.read_bytes()).hexdigest()
 r,e=request(setup,backend=FaultBackend(simple_mode='sdk_success'))
 check_success(r,settings)
 assert any(b.get('image_url',{}).get('url')==u for row in r.json()['data'] for b in row['content'])
 assert hashlib.sha256(settings.storage_path.read_bytes()).hexdigest()==before
 assert e[0]['stages'][0]['image_derivatives'] and set(r.json())=={'data'}

def test_off_component_physically_removed(tmp_path):
 import shutil,subprocess,os,sys
 from pathlib import Path
 root=Path(__file__).resolve().parents[1]
 shutil.copytree(root/'production',tmp_path/'production',ignore=shutil.ignore_patterns('__pycache__'))
 (tmp_path/'production/aml_api/search_images.py').unlink()
 script="from aml_api.production import runtime; from aml_api.production.process import execute; runtime.legacy=lambda *a,**kw:'ok'; assert execute('legacy',None,None,None,999999999,None,None,{})=='ok'"
 p=subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path/'production'),AML_SEARCH_IMAGE_MODE='off'),capture_output=True)
 assert p.returncode==0,p.stderr

def slow_decoder(stage,settings,request,policy,deadline,checkpoint,publish,stats):
 from PIL import Image
 from aml_api.search_images import scope,prepare
 original=Image.Image.load
 def blocked(im,*args,**kw):
  time.sleep(10)
  return original(im,*args,**kw)
 Image.Image.load=blocked
 with scope(deadline):prepare([{'type':'image_url','image_url':{'url':request}}])

def test_parent_deadline_terminates_blocked_decoder():
 from aml_api.production.process import StageRunner
 from aml_api.production.policy import Policy,SearchFault
 import multiprocessing
 start=time.monotonic()
 with pytest.raises(SearchFault):StageRunner(slow_decoder).run('simple',None,uri(),Policy(),start+1,None,lambda _:None)
 assert time.monotonic()-start<4 and not multiprocessing.active_children()

def test_large_jpeg_under_worker_memory_limit():
 import subprocess,sys
 code="""import resource,time
from test_search_images import uri,images,decode
resource.setrlimit(resource.RLIMIT_AS,(1024*1024*1024,1024*1024*1024))
u=uri((8000,6245),'JPEG')
x=images();out=x.image(u)
assert x.records[0]['original_pixels']==49960000
assert decode(out).width*decode(out).height<=2097152
"""
 if not sys.platform.startswith('linux'):pytest.skip('production Linux address-space ceiling')
 import os
 from pathlib import Path
 root=Path(__file__).resolve().parents[1]
 p=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,timeout=20,env=dict(os.environ,PYTHONPATH=str(root/'production')+os.pathsep+str(root/'testing')))
 assert p.returncode==0,p.stderr

def test_producer_identity_binds_policy(monkeypatch):
 from aml_api.seed_upstream.producer import production_identity
 original=production_identity()
 with scope(time.monotonic()+10):a=production_identity()
 monkeypatch.setenv('AML_SEARCH_IMAGE_MAX_EDGE','1024')
 with scope(time.monotonic()+10):b=production_identity()
 assert len({original,a,b})==3
