"""Select evidence using bounded promotion or optional stable category ordering."""
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
    grouped=config.reorder_mode=='stable_groups'
    pool=config.assess_pool if grouped else k
    with meter.phase('seed_set_media'):
        # All raw seed digests are validated, even candidates outside final S.
        for hit in snapshot.hits:
            value=view.contents([hit.memory_id])[hit.memory_id]
            if content_digest(value)!=hit.content_digest:raise Failure('source_content','core')
            if len(chosen)>=pool:continue
            item=SearchItem(id=hit.memory_id,content=value)
            size=0 if grouped else response_image_bytes(item.content)
            if not grouped and used+size>meter.limits.response_image_bytes:
                excluded.append({'id':hit.memory_id,'reason':'response_media_budget'});continue
            used+=size;chosen.append(hit.memory_id);raw[hit.memory_id]=value
            metadata[hit.memory_id]=trusted_metadata(view.metadata(hit.memory_id))
    failed={};judgments={};side={};query=[p.model_dump(mode='json') for p in request.query]
    with meter.phase('assessment'):
        for index,mid in enumerate(chosen):
            try:
                parts,mapping,limited=fragments(context,f'm{index}',raw[mid],plain_text(request.query),meter.limits,view.metadata(mid),config.max_fragments)
                judgment=assess(provider,meter,query,request.options,parts,metadata[mid],f'candidate_{index}')
            except BaseException as exc:
                if not config.continue_candidate_errors:
                    raise
                from ..production.candidate_errors import candidate_error
                code=candidate_error(exc)
                if code is None:
                    raise
                failed[mid]=code
                continue
            judgments[mid]=judgment.relevance
            side[mid]={'id':mid,'content_digest':content_digest(raw[mid]),'time':metadata[mid],
                'assessment':judgment.model_dump(mode='json'),
                'fragments':{ref:mapping[ref] for ref in judgment.fragment_ids},
                'observation_limited':limited,'hint_source':'model','hint_uncertainty':'unverified_query_relative'}
    if chosen and not judgments:
        raise Failure('assessment_unavailable','assessment')
    successful=[mid for mid in chosen if mid in judgments]
    with meter.phase('sort'):
        if grouped:
            from .reordering import stable_groups
            order=stable_groups(successful,judgments)
        else:
            order=promote(successful,judgments,config.max_move)
        if set(order)!=set(successful) or len(order)!=len(successful):raise Failure('membership_changed','sort')
    with meter.phase('serialize_media_validate'):
        items=[];returned=[];output_bytes=0
        for mid in order:
            if grouped and len(items)>=k:break
            size=response_image_bytes(SearchItem(id=mid,content=raw[mid]).content)
            if grouped and output_bytes+size>meter.limits.response_image_bytes:
                excluded.append({'id':mid,'reason':'response_media_budget'});continue
            output_bytes+=size;returned.append(mid);rank=len(items)+1
            item=SearchItem(id=mid,content=raw[mid],score=1.0/rank,created_at=metadata[mid]['created_at'])
            if content_digest([p.model_dump(mode='json') for p in item.content])!=side[mid]['content_digest']:raise Failure('serialization_content','delivery')
            items.append(item)
        if sum(response_image_bytes(i.content) for i in items)>meter.limits.response_image_bytes:raise Failure('delivery_capacity','delivery')
        response=SearchResponse(data=items)
    neighbor_info=None
    if config.neighbor_mode=='session':
        with meter.phase('neighbor_fill'):
            from .neighbor_fill import apply
            base_returned=list(returned)
            def size_of(mid):
                if mid not in raw:
                    raw[mid]=view.contents([mid])[mid]
                    metadata[mid]=trusted_metadata(view.metadata(mid))
                return response_image_bytes(SearchItem(id=mid,content=raw[mid]).content)
            neighbor_info=apply(base_returned,judgments,chosen,view,limit=k,
                media_limit=meter.limits.response_image_bytes,size_of=size_of,blocked_ids=failed)
            returned=neighbor_info['ids']
            if not set(neighbor_info['protected'])<=set(returned):raise Failure('membership_changed','neighbor_fill')
            inserted={x['id']:x for x in neighbor_info['inserted']}
            items=[]
            for rank,mid in enumerate(returned,1):
                size_of(mid)
                expected=content_digest(raw[mid])
                if mid in side and expected!=side[mid]['content_digest']:raise Failure('source_content','neighbor_fill')
                if mid not in side:
                    side[mid]={'id':mid,'content_digest':expected,'time':metadata[mid],
                        'assessment':None,'assessment_status':'not_assessed','fragments':{},
                        'observation_limited':True,'hint_source':'session_neighbor',
                        'hint_uncertainty':'not_assessed_query_relative'}
                if mid in inserted:
                    side[mid]={**side[mid],'neighbor_origin':inserted[mid]}
                item=SearchItem(id=mid,content=raw[mid],score=1.0/rank,created_at=metadata[mid]['created_at'])
                if content_digest([p.model_dump(mode='json') for p in item.content])!=expected:raise Failure('serialization_content','delivery')
                items.append(item)
            if len(items)>k or len(set(returned))!=len(returned) or sum(response_image_bytes(i.content) for i in items)>meter.limits.response_image_bytes:raise Failure('delivery_capacity','delivery')
            response=SearchResponse(data=items)
    sidecar={'schema_version':'simple.sidecar.v1','binding':context.binding,'returned_ids':returned,
        'coverage':'limited_to_returned_set_and_visible_fragments','conflicts_resolved':False,
        'absence_of_hint_proves_no_conflict':False,'items':[side[mid] for mid in returned]}
    meter.count(seed_count=len(snapshot.hits),selected_count=len(chosen),returned_count=len(returned),
        read_bytes=view.read_bytes,content_reads=view.content_reads,content_cache_hits=view.cache_hits,returned_bytes=len(response.model_dump_json().encode()),
        returned_images=sum(p.type=='image_url' for item in items for p in item.content),
        returned_texts=sum(p.type=='text' for item in items for p in item.content))
    return Outcome(response,sidecar,{'seed':snapshot.model_dump(mode='json'),'selected_ids':chosen,
        'candidate_errors':[{'id':mid,'code':code} for mid,code in failed.items()],
        'returned_ids':returned,'media_exclusions':excluded,'max_move':config.max_move,**({'neighbor_fill':neighbor_info,'neighbor_mode':config.neighbor_mode} if neighbor_info is not None else {}),**({'ranked_ids':order,'reorder_mode':config.reorder_mode,'assess_pool':config.assess_pool} if grouped else {})},context.model_dump(mode='json'),
        'partial' if failed else 'succeeded','uncertain_present' if 'uncertain' in judgments.values() else 'assessed' if chosen else 'empty',meter.summary())
