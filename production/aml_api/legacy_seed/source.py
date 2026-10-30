"""Detachable legacy query preparation and coarse retrieval, without rerank/neighbors."""
from ..service import _as_text, _chunk_text
from ..providers import VisionLanguageProvider
from ..retrieval import hybrid_rank
from ..postseed.seed_port import Binding,SeedSnapshot,SeedHit,SeedError,digest,content_digest,_check_view
from ..seed_upstream.producer import production_identity
from ..media import plain_text,image_urls,validate_parts,MAX_SEARCH_IMAGE_BYTES
from pathlib import Path
import hashlib

def query_texts(question,options,images,vlm):
    plan=VisionLanguageProvider.plan_query(vlm,question,options,images)
    texts=[question]
    visual=_as_text(plan.get('visual_description'));intent=_as_text(plan.get('intent'))
    anchors=plan.get('text_anchors',[]);keywords=_as_text(plan.get('keywords'))
    if intent:texts.append(intent)
    if visual:texts.append(visual)
    if isinstance(anchors,list):texts.extend(str(x).strip() for x in anchors if str(x).strip())
    if keywords:texts.append(keywords)
    if options:texts.append('Possible answer concepts: '+' | '.join(options))
    return tuple(dict.fromkeys(chunk for text in texts if text for chunk in _chunk_text(text))) or ('visual query',)

class LegacySeedSource:
    def retrieve(self,request,settings,config,view,vlm,embedder):
        config.check_capacity();validate_parts(request.query,total_limit=MAX_SEARCH_IMAGE_BYTES)
        producer=digest({'base':production_identity(),'module':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'strategy':'legacy_coarse_only'})
        binding=Binding(request_digest=digest(request.model_dump(mode='json')),scope_digest=digest(request.user_id),query_digest=digest({'query':[x.model_dump(mode='json') for x in request.query],'options':request.options}),preparation_digest=digest(()),read_view_digest=view.read_view_digest,producer_digest=producer)
        _check_view(binding,view);rows=view.views();ranked=[]
        if rows:
            texts=query_texts(plain_text(request.query),request.options or [],image_urls(request.query),vlm)
            binding=binding.model_copy(update={'preparation_digest':digest(texts)})
            ranked=hybrid_rank(embedder.encode(list(texts)),'\n'.join(texts),rows,config.retrieve_pool,use_bm25=config.use_bm25,bm25_k1=config.bm25_k1,bm25_b=config.bm25_b)
        _check_view(binding,view);raw=view.contents([mid for mid,_ in ranked]);_check_view(binding,view)
        return SeedSnapshot(binding=binding,config=config,hits=tuple(SeedHit(memory_id=mid,rank=i,score=float(score),content_digest=content_digest(raw[mid])) for i,(mid,score) in enumerate(ranked,1)))
