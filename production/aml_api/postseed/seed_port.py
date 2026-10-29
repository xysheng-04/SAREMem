"""Lossless seed boundary; no post-seed dependencies or provider construction.

PreparedSeedSource consumes frozen upstream query preparation, not a new planner.
A production scoped read transaction and budgeted embedder are supplied at M3.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError, model_validator
from ..retrieval import hybrid_rank

Hash = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Name = Annotated[str, Field(min_length=1)]


class SeedError(ValueError):
    """Fixed diagnostic codes only; never include raw input or source content."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def content_digest(content: list[dict]) -> str:
    # Ordered original blocks, including original image URI, hashed only in memory.
    return digest(content)


class Frozen(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True,frozen=True,allow_inf_nan=False)
    schema_version: Literal[1] = 1


class Binding(Frozen):
    request_digest: Hash
    scope_digest: Hash
    query_digest: Hash
    preparation_digest: Hash
    read_view_digest: Hash
    producer_digest: Hash


class SeedConfig(Frozen):
    retrieve_pool: Annotated[StrictInt,Field(ge=1)] = 80
    upstream_max_candidates: Annotated[StrictInt,Field(ge=1)] = 48
    capacity: Annotated[StrictInt,Field(ge=1,le=80)] = 80
    use_bm25: bool = False
    bm25_k1: Annotated[StrictFloat,Field(gt=0)] = 1.5
    bm25_b: Annotated[StrictFloat,Field(ge=0,le=1)] = .75

    def check_capacity(self) -> None:
        if min(self.retrieve_pool,self.upstream_max_candidates)>self.capacity:
            raise SeedError('seed_capacity')


class SeedHit(Frozen):
    memory_id: Name
    rank: Annotated[StrictInt,Field(ge=1)]
    score: StrictFloat
    content_digest: Hash


class SeedSnapshot(Frozen):
    binding: Binding
    config: SeedConfig
    hits: tuple[SeedHit,...]

    @model_validator(mode='after')
    def validate_hits(self):
        self.config.check_capacity()
        if len(self.hits)>min(self.config.retrieve_pool,self.config.upstream_max_candidates):
            raise ValueError('seed_count')
        if len({h.memory_id for h in self.hits})!=len(self.hits):
            raise ValueError('duplicate_id')
        if tuple(h.rank for h in self.hits)!=tuple(range(1,len(self.hits)+1)):
            raise ValueError('seed_rank')
        return self


def load_snapshot(raw: str | bytes) -> SeedSnapshot:
    try:
        return SeedSnapshot.model_validate_json(raw,strict=True)
    except (ValidationError,ValueError) as exc:
        raise SeedError('snapshot_schema') from None


def prepare_query_texts(question: str, anchors: tuple[str,...], keywords: tuple[str,...],
                       visual_description: str, options: tuple[str,...]) -> tuple[str,...]:
    """Frozen service's query-text projection; caller supplies validated upstream fields."""
    texts=[question,*anchors,' '.join(keywords),visual_description]
    if options:
        texts.append('Possible answer concepts: '+' | '.join(options))
    return tuple(dict.fromkeys(text.strip() for text in texts if text and text.strip())) or ('visual query',)


class SeedReadView(Protocol):
    scope_digest: str
    read_view_digest: str
    def views(self) -> list[dict]: ...
    def contents(self, ids: list[str]) -> dict[str,list[dict]]: ...


def _check_view(binding: Binding, view: SeedReadView) -> None:
    if (view.scope_digest,view.read_view_digest)!=(binding.scope_digest,binding.read_view_digest):
        raise SeedError('source_binding')


def _contents(binding: Binding, view: SeedReadView, ids: list[str]) -> dict[str,list[dict]]:
    _check_view(binding,view)
    values=view.contents(ids)
    _check_view(binding,view)
    if set(values)!=set(ids):
        raise SeedError('source_ids')
    return values


class ReplaySeedSource:
    def __init__(self,snapshot: SeedSnapshot):
        # Revalidate even a caller-provided model (model_copy/construct can bypass validation).
        self.snapshot=load_snapshot(snapshot.model_dump_json())

    def retrieve(self,binding: Binding,config: SeedConfig,view: SeedReadView) -> SeedSnapshot:
        if binding!=self.snapshot.binding or config!=self.snapshot.config:
            raise SeedError('snapshot_binding')
        contents=_contents(binding,view,[h.memory_id for h in self.snapshot.hits])
        if any(content_digest(contents[h.memory_id])!=h.content_digest for h in self.snapshot.hits):
            raise SeedError('source_content')
        return self.snapshot


class PreparedSeedSource:
    """Fixed-query online retrieval adapter, not a live query-preparation producer."""
    def retrieve(self,binding: Binding,config: SeedConfig,query_texts: tuple[str,...],
                 view: SeedReadView,embedder: Any) -> SeedSnapshot:
        config.check_capacity()
        if not query_texts or digest(query_texts)!=binding.preparation_digest:
            raise SeedError('query_preparation')
        _check_view(binding,view)
        rows=view.views()
        _check_view(binding,view)
        ranked=[]
        if rows:
            vectors=embedder.encode(list(query_texts))
            ranked=hybrid_rank(vectors,'\n'.join(query_texts),rows,
                               min(config.retrieve_pool,config.upstream_max_candidates),
                               use_bm25=config.use_bm25,bm25_k1=config.bm25_k1,bm25_b=config.bm25_b)
        contents=_contents(binding,view,[mid for mid,_ in ranked])
        try:
            hits=tuple(SeedHit(memory_id=mid,rank=i,score=float(score),
                               content_digest=content_digest(contents[mid]))
                       for i,(mid,score) in enumerate(ranked,1))
            return SeedSnapshot(binding=binding,config=config,hits=hits)
        except (ValidationError,ValueError):
            raise SeedError('seed_output') from None
