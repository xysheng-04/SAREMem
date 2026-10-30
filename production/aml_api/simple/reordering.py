from ..postseed.contracts import Failure

def stable_groups(ids,labels):
    if len(set(ids))!=len(ids) or set(ids)!=set(labels) or any(v not in {'direct','context','unrelated','uncertain'} for v in labels.values()):raise Failure('ranking_contract','sort')
    return [i for i in ids if labels[i]=='direct']+[i for i in ids if labels[i]=='context']+[i for i in ids if labels[i] in {'unrelated','uncertain'}]
