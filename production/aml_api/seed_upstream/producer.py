"""Complete frozen upstream request -> preparation -> raw seed narrow port.

No old package imports. No post-seed execution, tools or reranker construction.
The only permitted legacy reuse here is frozen upstream input semantics and the
shared media/retrieval/provider algorithms; see frozen_provenance.json.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json

from pydantic import ValidationError
from ..media import MAX_SEARCH_IMAGE_BYTES, validate_parts, plain_text, image_urls
from ..schemas import SearchRequest
from ..postseed.seed_port import (
    Binding, SeedConfig, SeedSnapshot, SeedError, PreparedSeedSource, ReplaySeedSource,
    digest, prepare_query_texts, _check_view,
)
from .config import EvidenceSettings
from .models import PlannerEnvelope
from .plan import validate_plan,decode_plan
from .planner_provider import plan_evidence_query
from .schema_port import SchemaPort


def production_identity() -> str:
    """Hash actual frozen producer, its shared code and installed algorithm versions."""
    api=Path(__file__).resolve().parents[1]
    files=list((api/'seed_upstream').glob('*.py'))+list((api/'seed_upstream').glob('*.json'))
    files += [api/name for name in ['providers.py','retrieval.py','media.py','schemas.py','config.py',
                                   'errors.py','captioning.py','audit.py','postseed/seed_port.py','postseed/__init__.py']]
    return digest({'files':[(str(p.relative_to(api)),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],
                   'versions':{n:importlib.metadata.version(n) for n in ['numpy','pydantic','openai','httpx']}})


def _producer_digest(settings,forced_payload,frozen_envelope):
    return digest({'code':production_identity(),'settings':asdict(settings),
                   'forced':forced_payload,'frozen_envelope':frozen_envelope})


def _binding(request,view,preparation,producer):
    return Binding(request_digest=digest(request.model_dump(mode='json')),
                   scope_digest=digest(request.user_id),
                   query_digest=digest({'query':[p.model_dump(mode='json') for p in request.query],
                                        'options':request.options}),
                   preparation_digest=preparation,read_view_digest=view.read_view_digest,
                   producer_digest=producer)


class OnlineSeedSource:
    def retrieve(self,request: SearchRequest,settings: EvidenceSettings,config: SeedConfig,
                 view,vlm,embedder,*,forced_payload=None,frozen_envelope=None) -> SeedSnapshot:
        config.check_capacity()
        if config.upstream_max_candidates!=settings.max_candidates:
            raise SeedError('seed_config_binding')
        # Seed source does not open files from forced_plan_path. Resolve payload explicitly
        # outside the port, preserving exact JSON values; no silent path/env lookup.
        producer=_producer_digest(settings,forced_payload,frozen_envelope)
        binding=_binding(request,view,digest(()),producer)
        _check_view(binding,view)
        validate_parts(request.query,total_limit=MAX_SEARCH_IMAGE_BYTES)
        question=plain_text(request.query);images=image_urls(request.query);options=request.options or []
        rows=view.views()
        _check_view(binding,view)
        if not rows:
            return SeedSnapshot(binding=binding,config=config,hits=())
        has_visual=bool(images or any(row['kind'].startswith('visual.') for row in rows))
        registry=SchemaPort()
        try:
            if frozen_envelope is not None:
                envelope=PlannerEnvelope.model_validate_json(json.dumps(frozen_envelope),strict=True)
                validate_plan(envelope.plan,settings,registry,has_visual_signal=has_visual)
                output=envelope.model_dump(mode='json')
            elif settings.mode=='forced':
                if forced_payload is None: raise SeedError('forced_payload_required')
                if 'plan' in forced_payload:
                    envelope=PlannerEnvelope.model_validate_json(json.dumps(forced_payload),strict=True)
                    validate_plan(envelope.plan,settings,registry,has_visual_signal=has_visual)
                    output=envelope.model_dump(mode='json')
                else:
                    validate_plan(decode_plan(forced_payload),settings,registry,has_visual_signal=has_visual)
                    output={'query_anchors':[question or 'visual query'],'keywords':[],'visual_description':''}
            else:
                _,output=plan_evidence_query(vlm,registry,settings,question,options,images,has_visual_signal=has_visual)
            texts=prepare_query_texts(question,tuple(output.get('query_anchors',[])),
                                      tuple(output.get('keywords',[])),output.get('visual_description',''),tuple(options))
            _check_view(binding,view)
            binding=_binding(request,view,digest(texts),producer)
            return PreparedSeedSource().retrieve(binding,config,texts,view,embedder)
        except SeedError: raise
        except Exception: raise SeedError('upstream_producer') from None

    def replay(self,request,settings,config,view,snapshot,*,forced_payload=None,frozen_envelope=None):
        if config.upstream_max_candidates!=settings.max_candidates:
            raise SeedError('seed_config_binding')
        binding=_binding(request,view,snapshot.binding.preparation_digest,
                         _producer_digest(settings,forced_payload,frozen_envelope))
        return ReplaySeedSource(snapshot).retrieve(binding,config,view)
