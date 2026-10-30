import base64,io,json,time,types,copy,multiprocessing
from unittest.mock import patch
import pytest
from PIL import Image
from aml_api import add_images
from aml_api.errors import BadRequest,UpstreamUnavailable
from aml_api.providers import VisionLanguageProvider
from aml_api.captioning import caption_policy

def uri(size=(40,20),mode='RGB',fmt='PNG',exif=None):
 im=Image.new(mode,size,(30,60,90,80) if mode=='RGBA' else (30,60,90));b=io.BytesIO();im.save(b,format=fmt,**({'exif':exif} if exif else {}));return 'data:image/'+fmt.lower().replace('jpg','jpeg')+';base64,'+base64.b64encode(b.getvalue()).decode()
def decoded(u):return Image.open(io.BytesIO(base64.b64decode(u.split(',')[1])))

@pytest.mark.parametrize('fmt',['PNG','JPEG','WEBP'])
def test_small_unchanged(fmt):
 u=uri(fmt=fmt);assert add_images.prepare([u])==[u]
def test_large_safe_and_audit():
 u=uri((7360,4912),fmt='JPEG');records=[];token=add_images.AUDIT.set(records)
 try:v=add_images.prepare([u])[0]
 finally:add_images.AUDIT.reset(token)
 im=decoded(v);assert im.width*im.height<=2097152 and max(im.size)<=2048
 assert abs(im.width/im.height-7360/4912)<.01 and records[0]['policy']['version']=='add-png-v1'
 assert 'images' not in records[0]
def test_alpha_exif_and_order():
 ex=Image.Exif();ex[274]=6
 a,b=add_images.prepare([uri((40,20),fmt='JPEG',exif=ex),uri((2400,1600),mode='RGBA')]);assert decoded(a).size==(20,40);assert decoded(b).getpixel((0,0))[3]==80
def test_off_corrupt_deadline(monkeypatch):
 monkeypatch.setenv('AML_ADD_IMAGE_MODE','off');assert add_images.prepare(['bad'])==['bad']
 monkeypatch.setenv('AML_ADD_IMAGE_MODE','auto')
 with pytest.raises(BadRequest):add_images.prepare(['data:image/png;base64,????'])
 token=add_images.DEADLINE.set(time.monotonic()-1)
 try:
  with pytest.raises(UpstreamUnavailable):add_images.prepare([uri()])
 finally:add_images.DEADLINE.reset(token)
 assert not multiprocessing.active_children()
def provider(sent):
 v=VisionLanguageProvider.__new__(VisionLanguageProvider);v.model='gpt-4o-mini';v.caption_policy=caption_policy('full')
 def create(**kw):
  sent.append(kw)
  content='bad' if len(sent)==1 else '{"images":[]}'
  return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content),finish_reason='stop')],usage=None)
 v.client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
 return v
def test_caption_json_retry_prepares_once():
 sent=[];v=provider(sent);u=uri((3000,2000));images=[u];original=copy.deepcopy(images)
 with patch.object(add_images,'prepare',wraps=add_images.prepare) as prep:v.describe_images('synthetic',images);assert prep.call_count==1
 assert images==original and sent[0]['messages']==sent[1]['messages']
 assert next(x['image_url']['url'] for x in sent[0]['messages'][1]['content'] if x['type']=='image_url')!=u

def test_add_original_database_and_idempotency(tmp_path,monkeypatch):
 from dataclasses import replace
 from aml_api.config import Settings
 from aml_api.store import SQLiteStore
 from aml_api.schemas import AddRequest
 from aml_api.service import RouteAService
 from faults import FakeEncoder
 monkeypatch.setenv('OPENAI_API_KEY','offline-only');settings=replace(Settings.from_env(),storage_path=tmp_path/'new.sqlite3');store=SQLiteStore(settings.storage_path)
 sent=[];v=provider(sent);s=RouteAService(settings,store=store,embedder=FakeEncoder(),vlm=v);u=uri((3000,2000));request=AddRequest(request_id='request',user_id='user',session_id='session',messages=[{'role':'user','content':[{'type':'text','text':'before'},{'type':'image_url','image_url':{'url':u}},{'type':'text','text':'after'}]}]);before=request.model_dump(mode='json');s.add(request);n=len(sent);s.add(request);assert len(sent)==n and request.model_dump(mode='json')==before
 import sqlite3
 with sqlite3.connect(settings.storage_path) as db:content=json.loads(db.execute('select content_json from memories').fetchone()[0])
 assert [x['type'] for x in content]==['text','image_url','text'] and content[1]['image_url']['url']==u
