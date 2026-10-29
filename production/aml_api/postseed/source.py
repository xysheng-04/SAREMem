"""One scoped SQLite transaction; bounded snapshot and original-coordinate views."""
import copy,json,sqlite3,hashlib,re
from pathlib import Path
from urllib.parse import quote
from dataclasses import dataclass
import numpy as np
from ..media import validate_image_data_uri
from .seed_port import digest,content_digest
from .contracts import Block,Descriptor,Failure

class SQLiteReadView:
 def __init__(self,path,user_id,budget):
  self.path=Path(path).resolve(strict=True);self.user_id=user_id;self.scope_digest=digest(user_id);self.budget=budget
 def __enter__(self):
  self.db=sqlite3.connect('file:'+quote(str(self.path),safe='/:')+'?mode=ro',uri=True)
  self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA query_only=ON');self.db.execute('BEGIN')
  self.progress_failure=None;self.read_bytes=0
  def progress():
   try:self.budget.store(vm_steps=100);return 0
   except Failure as exc:self.progress_failure=exc;return 1
  self.db.set_progress_handler(progress,100)
  try:
   self.memories={r['memory_id']:dict(r) for r in self._fetch('SELECT seq,memory_id,session_id,source_timestamp_ms,content_json,created_at FROM memories WHERE user_id=? ORDER BY seq,memory_id',(self.user_id,))}
   self._views=self._fetch('SELECT v.view_id,v.memory_id,v.kind,v.text,v.embedding,v.dimension FROM views v JOIN memories m ON m.memory_id=v.memory_id WHERE m.user_id=? ORDER BY m.seq,v.view_id',(self.user_id,))
   size=sum(len(r['content_json'].encode()) for r in self.memories.values())+sum(len(r['text'].encode())+len(r['embedding']) for r in self._views)
   if size>self.budget.limits.source_bytes:raise Failure('budget_limited','source_bytes')
   # Digest represents exactly this scoped read transaction, including indexed views.
   self.read_view_digest=digest({'memories':list(self.memories.values()),'views':[{**dict(r),'embedding':hashlib.sha256(r['embedding']).hexdigest()} for r in self._views]})
   return self
  except Exception:self.db.close();raise
 def _fetch(self,sql,params):
  self.budget.store(reads=1);out=[]
  try:
   cursor=self.db.execute(sql,params)
   while True:
    rows=cursor.fetchmany(1)
    if not rows:break
    for row in rows:
     self.read_bytes+=sum(len(v.encode()) if isinstance(v,str) else len(v) if isinstance(v,bytes) else 8 for v in row)
     if self.read_bytes>self.budget.limits.source_bytes:raise Failure('budget_limited','source_bytes')
    self.budget.store(rows=len(rows));out.extend(dict(r) for r in rows)
   return out
  except sqlite3.Error:
   if self.progress_failure:raise self.progress_failure from None
   raise Failure('source_read','store') from None
 def __exit__(self,*args):self.db.rollback();self.db.close()
 def ids(self):return tuple(self.memories)
 def views(self):
  self.budget.store(reads=1,rows=len(self._views))
  return [{**{k:v for k,v in r.items() if k not in {'embedding','dimension'}},'embedding':np.frombuffer(r['embedding'],dtype=np.float32,count=r['dimension']).copy()} for r in self._views]
 def contents(self,ids):
  self.budget.store(reads=1,rows=len(ids))
  if len(set(ids))!=len(ids) or any(mid not in self.memories for mid in ids):raise Failure('source_ids','store')
  return {mid:json.loads(self.memories[mid]['content_json']) for mid in ids}
 def metadata(self,mid):
  if mid not in self.memories:raise Failure('source_ids','store')
  return {k:v for k,v in self.memories[mid].items() if k!='content_json'}

@dataclass(frozen=True)
class Presented:
 descriptor:Descriptor
 content:tuple[dict,...]


def present(context,key,content,query,limits,metadata=None):
 metadata=metadata or {}
 blocks=[];visible=[];remaining=limits.source_chars;image_bytes=0;limited=False
 terms=re.findall(r'\w+',query.lower())
 for i,p in enumerate(content):
  if p['type']=='text':
   text=p['text'];n=min(len(text),remaining);remaining-=n
   positions=[text.lower().find(term) for term in terms if term in text.lower()]
   start=max(0,min(min(positions) if positions else 0,len(text)-n)-n//4) if n<len(text) else 0
   end=start+n;shown=n>0
   blocks.append(Block(index=i,kind='text',original_length=len(text),visible=shown,start=start if shown else None,end=end if shown else None))
   if shown:visible.append({'block_index':i,'start':start,'end':end,'type':'text','text':text[start:end]})
   limited|=n<len(text)
  elif p['type']=='image_url':
   size=validate_image_data_uri(p['image_url']['url']).decoded_bytes;shown=image_bytes+size<=limits.source_image_bytes
   if shown:image_bytes+=size;visible.append({'block_index':i,**copy.deepcopy(p)})
   blocks.append(Block(index=i,kind='image_url',original_length=size,visible=shown,start=None,end=None));limited|=not shown
  else:raise Failure('source_type')
 return Presented(Descriptor(binding=context.binding,memory_key=key,content_digest=content_digest(content),observation='limited' if limited else 'complete',source_timestamp_ms=metadata.get('source_timestamp_ms'),sequence=metadata.get('seq'),session_digest=digest(metadata['session_id']) if metadata.get('session_id') is not None else None,has_order=metadata.get('seq') is not None and metadata.get('session_id') is not None,blocks=tuple(blocks)),tuple(visible))
