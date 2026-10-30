"""Optional deterministic evidence assembly; no providers, labels or database writes."""
def supplement(base_ids, labels, seed_ids, neighbors, size_of, *, limit=20, media_limit=30*1024**2, enabled=True):
    if len(set(base_ids))!=len(base_ids) or len(base_ids)>limit:raise ValueError('base contract')
    if any(mid not in labels for mid in base_ids):raise ValueError('missing base label')
    if not enabled:return {'ids':list(base_ids),'protected':[],'anchors':[],'quota':0,'inserted':[],'excluded':[]}
    protected=[x for x in base_ids if labels[x] in {'direct','context'}]
    anchors=[x for x in seed_ids if x in protected and labels[x]=='direct']
    if len(set(seed_ids))!=len(seed_ids) or set(anchors)!={x for x in protected if labels[x]=='direct'}:raise ValueError('anchor binding')
    quota=limit-len(protected);out=list(protected);seen=set(out);used=sum(size_of(x) for x in out)
    if used>media_limit:raise ValueError('protected media exceeds limit')
    inserted=[];excluded=[]
    for anchor in anchors:
        if len(inserted)>=quota:break
        for direction,mid in neighbors(anchor):
            if len(inserted)>=quota:break
            if direction not in {'before','after'}:raise ValueError('direction contract')
            if mid in seen:excluded.append({'id':mid,'anchor':anchor,'reason':'duplicate'});continue
            size=size_of(mid)
            if used+size>media_limit:excluded.append({'id':mid,'anchor':anchor,'reason':'response_media_budget'});continue
            seen.add(mid);out.append(mid);used+=size;inserted.append({'id':mid,'anchor':anchor,'direction':direction,'new_to_baseline':mid not in base_ids})
    for mid in base_ids:
        if len(out)>=limit:break
        if mid in seen:continue
        size=size_of(mid)
        if used+size>media_limit:excluded.append({'id':mid,'reason':'backfill_media_budget'});continue
        seen.add(mid);out.append(mid);used+=size
    assert len(set(out))==len(out)<=limit and set(protected)<=set(out)
    return {'ids':out,'protected':protected,'anchors':anchors,'quota':quota,'inserted':inserted,'excluded':excluded,'media_bytes':used}


def apply(base_ids, labels, seed_ids, view, *, limit, media_limit, size_of, blocked_ids=()):
    """Use the existing scope-bound read transaction, never a second store."""
    sessions={};positions={}
    for mid,row in view.memories.items():
        if type(row['seq']) is not int or not isinstance(row['session_id'],str):
            from ..postseed.contracts import Failure
            raise Failure('source_sequence','neighbor_fill')
        sessions.setdefault(row['session_id'],[]).append(mid)
    for session,ids in sessions.items():
        ids.sort(key=lambda mid:(view.memories[mid]['seq'],mid))
        for i,mid in enumerate(ids):positions[mid]=(session,i)
    def neighbors(anchor):
        session,i=positions[anchor];ids=sessions[session];out=[]
        if i>0:out.append(('before',ids[i-1]))
        if i+1<len(ids):out.append(('after',ids[i+1]))
        return [(direction,mid) for direction,mid in out if mid not in blocked_ids]
    return supplement(base_ids,labels,seed_ids,neighbors,size_of,limit=limit,media_limit=media_limit)
