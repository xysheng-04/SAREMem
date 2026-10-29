"""S is fixed before judgments; promotion may only cross unrelated neighbors."""
from dataclasses import dataclass
from ..schemas import SearchResponse,SearchItem
from ..media import validate_parts,MAX_SEARCH_IMAGE_BYTES,response_image_bytes,plain_text
from ..postseed.contracts import Failure,check_request
from ..postseed.seed_port import digest,content_digest,load_snapshot
from .assessment import fragments,assess

@dataclass(frozen=True)
class Outcome:
    response: SearchResponse
    sidecar: dict
    stages: dict
    context: dict
    execution_status: str
    assessment_status: str
    budget: dict

def promote(ids,labels,max_move=3):
    order=list(ids)
    if len(set(order))!=len(order) or set(order)!=set(labels):raise Failure('ranking_contract','sort')
    for mid in ids:
        if labels[mid]!='direct':continue
        i=order.index(mid);moved=0
        while i>0 and moved<max_move and labels[order[i-1]]=='unrelated':
            order[i-1],order[i]=order[i],order[i-1];i-=1;moved+=1
    return order

from .metadata import trusted_metadata


def run_search(request,context,view,snapshot,provider,meter,config):
    check_request(request,context);validate_parts(request.query,total_limit=MAX_SEARCH_IMAGE_BYTES)
    snapshot=load_snapshot(snapshot.model_dump_json())
    if (view.scope_digest,view.read_view_digest)!=(context.scope_digest,context.read_view_digest):raise Failure('source_binding','core')
    b=snapshot.binding
    if (b.request_digest,b.scope_digest,b.read_view_digest,b.query_digest)!=(context.request_digest,context.scope_digest,context.read_view_digest,digest({'query':[p.model_dump(mode='json') for p in request.query],'options':request.options})):
        raise Failure('seed_binding','core')
    chosen=[];raw={};metadata={};excluded=[];used=0;k=min(request.top_k,meter.limits.return_limit)
    with meter.phase('seed_set_media'):
        # All raw seed digests are validated, even candidates outside final S.
        for hit in snapshot.hits:
            value=view.contents([hit.memory_id])[hit.memory_id]
            if content_digest(value)!=hit.content_digest:raise Failure('source_content','core')
            if len(chosen)>=k:continue
            item=SearchItem(id=hit.memory_id,content=value)
            size=response_image_bytes(item.content)
            if used+size>meter.limits.response_image_bytes:
                excluded.append({'id':hit.memory_id,'reason':'response_media_budget'});continue
            used+=size;chosen.append(hit.memory_id);raw[hit.memory_id]=value
            metadata[hit.memory_id]=trusted_metadata(view.metadata(hit.memory_id))
    judgments={};side={};query=[p.model_dump(mode='json') for p in request.query]
    with meter.phase('assessment'):
        for index,mid in enumerate(chosen):
            parts,mapping,limited=fragments(context,f'm{index}',raw[mid],plain_text(request.query),meter.limits,view.metadata(mid),config.max_fragments)
            judgment=assess(provider,meter,query,request.options,parts,metadata[mid],f'candidate_{index}')
            judgments[mid]=judgment.relevance
            side[mid]={'id':mid,'content_digest':content_digest(raw[mid]),'time':metadata[mid],
                'assessment':judgment.model_dump(mode='json'),
                'fragments':{ref:mapping[ref] for ref in judgment.fragment_ids},
                'observation_limited':limited,'hint_source':'model','hint_uncertainty':'unverified_query_relative'}
    with meter.phase('sort'):
        order=promote(chosen,judgments,config.max_move)
        if set(order)!=set(chosen) or len(order)!=len(chosen):raise Failure('membership_changed','sort')
    with meter.phase('serialize_media_validate'):
        items=[]
        for rank,mid in enumerate(order,1):
            item=SearchItem(id=mid,content=raw[mid],score=1.0/rank,created_at=metadata[mid]['created_at'])
            if content_digest([p.model_dump(mode='json') for p in item.content])!=side[mid]['content_digest']:raise Failure('serialization_content','delivery')
            items.append(item)
        if sum(response_image_bytes(i.content) for i in items)>meter.limits.response_image_bytes:raise Failure('delivery_capacity','delivery')
        response=SearchResponse(data=items)
    sidecar={'schema_version':'simple.sidecar.v1','binding':context.binding,'returned_ids':order,
        'coverage':'limited_to_returned_set_and_visible_fragments','conflicts_resolved':False,
        'absence_of_hint_proves_no_conflict':False,'items':[side[mid] for mid in order]}
    meter.count(seed_count=len(snapshot.hits),selected_count=len(chosen),returned_count=len(order),
        read_bytes=view.read_bytes,content_reads=view.content_reads,content_cache_hits=view.cache_hits,returned_bytes=len(response.model_dump_json().encode()),
        returned_images=sum(p.type=='image_url' for item in items for p in item.content),
        returned_texts=sum(p.type=='text' for item in items for p in item.content))
    return Outcome(response,sidecar,{'seed':snapshot.model_dump(mode='json'),'selected_ids':chosen,
        'returned_ids':order,'media_exclusions':excluded,'max_move':config.max_move},context.model_dump(mode='json'),
        'succeeded','uncertain_present' if 'uncertain' in judgments.values() else 'assessed' if chosen else 'empty',meter.summary())
