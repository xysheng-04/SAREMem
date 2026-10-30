"""Query-planner-only JSON compatibility; never invent or repair field values."""
import json,re
FIELDS={'intent','visual_description','text_anchors','keywords'}

def _pairs(items):
    value={}
    for key,item in items:
        if key in value:raise ValueError('duplicate query field')
        value[key]=item
    return value

def _constant(_):raise ValueError('nonfinite query value')

def _validate(value):
    if not isinstance(value,dict) or set(value)!=FIELDS:raise ValueError('invalid query fields')
    if not isinstance(value['intent'],str) or not isinstance(value['visual_description'],str):raise ValueError('invalid query text')
    if not isinstance(value['text_anchors'],list) or any(not isinstance(x,str) for x in value['text_anchors']):raise ValueError('invalid query anchors')
    k=value['keywords']
    if not (isinstance(k,str) or isinstance(k,list) and all(isinstance(x,str) for x in k)):raise ValueError('invalid query keywords')
    return value

def parse_query_plan(raw):
    if not isinstance(raw,str) or not raw.strip():raise ValueError('empty query response')
    text=raw.strip();decoder=json.JSONDecoder(object_pairs_hook=_pairs,parse_constant=_constant)
    try:value=decoder.decode(text)
    except json.JSONDecodeError:
        pass
    else:return _validate(value)
    match=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',text,re.I)
    if match:
        return _validate(decoder.decode(match.group(1).strip()))
    # Walk complete objects without interpreting prose or nested objects as plans.
    objects=[];i=0
    while i<len(text):
        starts=[j for j in (text.find('{',i),text.find('[',i)) if j>=0]
        if not starts:break
        start=min(starts)
        try:value,end=decoder.raw_decode(text,start)
        except json.JSONDecodeError:raise ValueError('incomplete query object') from None
        objects.append(value);i=end
        if len(objects)>1:raise ValueError('ambiguous query objects')
    if len(objects)!=1:raise ValueError('missing complete query object')
    return _validate(objects[0])
