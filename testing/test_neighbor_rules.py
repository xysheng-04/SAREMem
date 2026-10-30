from aml_api.simple.neighbor_fill import supplement

def fill(base,labels,seed,ns,**kw):return supplement(base,labels,seed,lambda x:ns.get(x,[]),lambda x:1,**kw)
def test_quota_and_protected_context_and_anchor_order():
 b=list('abcde');labels=dict(zip(b,['direct','direct','context','unrelated','uncertain']))
 o=fill(b,labels,list('bacde'),{'b':[('before','x'),('after','y')],'a':[('before','z'),('after','w')]},limit=5)
 assert o['ids']==list('abcxy') and o['anchors']==list('ba') and o['quota']==2

def test_full_protected_no_neighbor_read():
 o=supplement(list('ab'),dict.fromkeys('ab','direct'),list('ab'),lambda _:(_ for _ in ()).throw(AssertionError()),lambda _:1,limit=2)
 assert o['ids']==list('ab') and o['quota']==0

def test_both_sides_dedup_and_backfill():
 o=fill(list('abcde'),dict(zip('abcde',['direct','context','unrelated','uncertain','unrelated'])),list('abcde'),{'a':[('before','b'),('after','x')]},limit=5)
 assert o['ids']==list('abxcd') and [x['id'] for x in o['inserted']]==['x']

def test_no_direct_unchanged():
 b=list('abc');o=fill(b,dict.fromkeys(b,'uncertain'),b,{},limit=3);assert o['ids']==b

def test_media_skip_and_neighbor_already_in_base():
 b=list('abc');labels=dict(zip(b,['direct','unrelated','uncertain']))
 o=supplement(b,labels,b,lambda _:[('before','big'),('after','c')],lambda x:100 if x=='big' else 1,limit=3,media_limit=3)
 assert o['ids']==list('acb') and not o['inserted'][0]['new_to_baseline']

def test_disabled_is_identity():
 b=list('abc');assert fill(b,dict.fromkeys(b,'context'),b,{},enabled=False)['ids']==b
