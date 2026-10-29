"""Strict request-bound contracts. No evaluation metadata enters this module."""
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field,ValidationError,model_validator
from .seed_port import digest
Hash=Annotated[str,Field(pattern=r'^[0-9a-f]{64}$')]
Key=Annotated[str,Field(pattern=r'^[a-z][a-z0-9_]{0,31}$')]
Positive=Annotated[int,Field(gt=0)]

class Failure(RuntimeError):
 def __init__(self,code,stage='contract',path=''):
  self.code=code;self.stage=stage;self.path=path
  super().__init__(code)
 def safe(self):return {'schema_version':1,'code':self.code,'stage':self.stage,'path':self.path}

class Strict(BaseModel):
 model_config=ConfigDict(extra='forbid',strict=True,frozen=True,allow_inf_nan=False)
 schema_version:Literal[1]=1

class Wire(Strict):
 schema_version:Literal[1]

class Context(Strict):
 request_nonce:Hash
 request_digest:Hash
 scope_digest:Hash
 read_view_digest:Hash
 config_digest:Hash
 code_digest:Hash
 @property
 def binding(self):return digest(self.model_dump(mode='json'))

class Limits(Strict):
 contract_repairs:int=Field(default=4,ge=0,le=8)
 provider_calls:Positive=160
 input_tokens:Positive=5000000
 output_tokens:Positive=150000
 cost_micros:int=Field(default=0,ge=0)
 store_rows:Positive=100000
 store_vm_steps:Positive=10000000
 operations:Positive=4
 wall_seconds:Positive=600
 request_seconds:Positive=30
 source_chars:Positive=12000
 source_image_bytes:Positive=10485760
 payload_bytes:Positive=32000000
 response_image_bytes:Positive=31457280
 return_limit:int=Field(default=20,ge=1,le=100)
 source_bytes:Positive=64000000

class NeedWire(Wire):
 key:Key
 text:str=Field(min_length=1,max_length=512)
 category:Literal['fact','list','compare','temporal']
 compound:bool
 preference:Literal['search','neighbors','time_range']
 phrase:str|None=Field(max_length=256)
class RelationWire(Wire):
 key:Key
 left:Key
 right:Key
 kind:Literal['same_entity','same_event','temporal','compare']
 requirement:str=Field(min_length=1,max_length=256)
class PlanWire(Wire):
 catalog_id:Hash
 needs:tuple[NeedWire,...]=Field(min_length=1,max_length=4)
 relations:tuple[RelationWire,...]=Field(max_length=4)
class NeedPlan(Strict):
 binding:Hash
 needs:tuple[NeedWire,...]
 relations:tuple[RelationWire,...]

class Block(Strict):
 index:int=Field(ge=0)
 kind:Literal['text','image_url']
 original_length:int=Field(ge=0)
 visible:bool
 start:int|None
 end:int|None
class Descriptor(Strict):
 binding:Hash
 memory_key:Key
 content_digest:Hash
 observation:Literal['complete','limited']
 source_timestamp_ms:int|None=None
 sequence:int|None=None
 session_digest:Hash|None=None
 has_order:bool=False
 blocks:tuple[Block,...]
class Ref(Wire):
 memory_key:Key
 block_index:int=Field(ge=0)
 start:int|None
 end:int|None
class AssessmentItem(Wire):
 need_key:Key
 label:Literal['supports','partial','contradicts','irrelevant','unknown']
 refs:tuple[Ref,...]
class AssessmentWire(Wire):
 catalog_id:Hash
 memory_key:Key
 observation:Literal['complete','limited']
 items:tuple[AssessmentItem,...]
class Assessment(Strict):
 binding:Hash
 memory_key:Key
 need_key:Key
 label:Literal['supports','partial','contradicts','irrelevant','unknown']
 observation:Literal['complete','limited']
 refs:tuple[Ref,...]

class MemberRole(Wire):
 memory_key:Key
 role:Literal['fact','bridge','context','counterevidence']
class Endpoint(Strict):
 need_key:Key
 members:tuple[Key,...]

class BundleWire(Wire):
 catalog_id:Hash
 verdict:Literal['supported','partial','contradicted','unknown']
 members:tuple[MemberRole,...]=Field(min_length=1,max_length=3)
 refs:tuple[Ref,...]
class Bundle(Strict):
 binding:Hash
 target:Key
 target_kind:Literal['need','relation']
 members:tuple[Key,...]
 verdict:Literal['supported','partial','contradicted','unknown']
 roles:tuple[MemberRole,...]
 refs:tuple[Ref,...]
 endpoints:tuple[Endpoint,...]=()
 resolves:tuple[Hash,...]=()


def decode(model,value,stage):
 try:return model.model_validate_json(__import__('json').dumps(value,allow_nan=False),strict=True)
 except (ValidationError,ValueError,TypeError):raise Failure('schema',stage) from None

def check_request(request,context):
 if digest(request.model_dump(mode='json'))!=context.request_digest or digest(request.user_id)!=context.scope_digest:
  raise Failure('request_binding')

def check_refs(refs,descriptors):
 seen=set()
 for ref in refs:
  key=(ref.memory_key,ref.block_index,ref.start,ref.end)
  if key in seen:raise Failure('duplicate_ref')
  seen.add(key)
  d=descriptors.get(ref.memory_key)
  if d is None or ref.block_index>=len(d.blocks):raise Failure('source_ref')
  b=d.blocks[ref.block_index]
  if b.index!=ref.block_index or not b.visible:raise Failure('invisible_ref')
  if b.kind=='image_url':
   if ref.start is not None or ref.end is not None:raise Failure('image_span')
  elif ref.start is None or ref.end is None or not (b.start<=ref.start<ref.end<=b.end):
   raise Failure('text_span')
