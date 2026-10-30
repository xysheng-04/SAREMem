def test_legacy_projection_and_coarse_rank_equivalence():
    from pathlib import Path
    import types
    from unittest.mock import patch
    import numpy as np
    from aml_api.legacy_seed.source import LegacySeedSource
    from aml_api.providers import VisionLanguageProvider
    from aml_api.service import RouteAService
    from aml_api.schemas import SearchRequest
    from aml_api.postseed.seed_port import SeedConfig,digest
    from aml_api.seed_upstream.config import EvidenceSettings
    class Stop(Exception):pass
    class VLM:
     def _json(self,**kw):return {'intent':'needed fact','visual_description':'','text_anchors':['anchor one','anchor two'],'keywords':['bicycle','blue']}
     _multimodal_content=staticmethod(VisionLanguageProvider._multimodal_content)
     def plan_query(self,*args):return VisionLanguageProvider.plan_query(self,*args)
    class Emb:
     def encode(self,texts):self.texts=texts;return np.tile(np.array([[1.,0.]],dtype=np.float32),(len(texts),1))
    rows=[{'memory_id':f'm{i:03d}','text':f'blue bicycle {i}','kind':'message.user','embedding':np.array([1-i/200.,i/200.],dtype=np.float32)} for i in range(100)]
    req=SearchRequest(query=[{'type':'text','text':'Which bicycle?'}],options=['blue','red'],user_id='synthetic',top_k=20)
    class View:
     scope_digest=digest(req.user_id);read_view_digest=digest('synthetic')
     def views(self):return rows
     def contents(self,ids):return {i:[{'type':'text','text':i}] for i in ids}
    legacy=RouteAService.__new__(RouteAService);legacy.settings=types.SimpleNamespace(retrieve_pool=80,use_bm25=False,bm25_k1=1.5,bm25_b=.75)
    legacy.vlm=VLM();legacy.embedder=Emb();legacy.store=types.SimpleNamespace(views_for_user=lambda u:rows,memories_by_ids=lambda u,ids:(_ for _ in ()).throw(Stop()))
    import aml_api.service as service
    original=service.hybrid_rank;seen=[]
    def capture(*args,**kwargs):
     value=original(*args,**kwargs);seen.extend(value);return value
    with patch.object(service,'hybrid_rank',capture):
     try:legacy.search(req)
     except Stop:pass
    emb=Emb();out=LegacySeedSource().retrieve(req,EvidenceSettings(),SeedConfig(retrieve_pool=80,upstream_max_candidates=80),View(),VLM(),emb)
    assert legacy.embedder.texts==emb.texts
    assert [(h.memory_id,h.score) for h in out.hits]==seen and len(out.hits)==80
    class Empty(View):
     def views(self):return []
    assert not LegacySeedSource().retrieve(req,EvidenceSettings(),SeedConfig(retrieve_pool=80,upstream_max_candidates=80),Empty(),None,None).hits
    
    

from pathlib import Path
import time
from test_faults import setup
from aml_api.production.runtime import simple
from aml_api.production.policy import Policy
from aml_api.schemas import SearchRequest

def test_default_physically_removable_then_legacy_empty_scope(setup,monkeypatch):
 settings,_=setup;request=SearchRequest(user_id='empty-test-scope',query=[{'type':'text','text':'synthetic'}],top_k=20)
 module=Path(__file__).resolve().parents[1]/'production/aml_api/legacy_seed';hidden=module.with_name('_legacy_seed_test_hidden');module.rename(hidden)
 try:
  monkeypatch.delenv('AML_SEED_RETRIEVAL_MODE',raising=False)
  assert simple(settings,request,Policy(),time.monotonic()+60,lambda _:None,{'dropped':0}).data==[]
 finally:hidden.rename(module)
 monkeypatch.setenv('AML_SEED_RETRIEVAL_MODE','legacy')
 assert simple(settings,request,Policy(),time.monotonic()+60,lambda _:None,{'dropped':0}).data==[]
