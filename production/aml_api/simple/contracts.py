from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from ..postseed.contracts import Limits, Failure
from ..postseed.seed_port import SeedConfig

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)

class Assessment(Strict):
    relevance: Literal['direct', 'context', 'unrelated', 'uncertain']
    fragment_ids: tuple[str, ...] = Field(max_length=32)
    possible_conflict: bool

class Price(Strict):
    endpoint_alias: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,40}$')
    currency: str = 'CNY'
    unit: Literal['micro_currency_per_token'] = 'micro_currency_per_token'
    effective_utc: str
    source: str = Field(max_length=120)
    input_micros: int | None = Field(default=None, ge=0)
    output_micros: int | None = Field(default=None, ge=0)
    cached_input_micros: int | None = Field(default=None, ge=0)

class Config(Strict):
    schema_version: Literal['simple.v1'] = 'simple.v1'
    limits: Limits = Limits(contract_repairs=0)
    seed: SeedConfig = SeedConfig()
    upstream: dict = Field(default_factory=dict)
    forced_payload: dict | None = None
    frozen_envelope: dict | None = None
    continue_candidate_errors: bool = False
    neighbor_mode: Literal['off','session'] = 'off'
    reorder_mode: Literal['bounded','stable_groups'] = 'bounded'
    assess_pool: int = Field(default=20,ge=1,le=80)
    max_move: int = Field(default=3, ge=0, le=3)
    max_fragments: int = Field(default=32, ge=1, le=32)
    input_bound: int = Field(default=128000, gt=0)
    output_bound: int = Field(default=8192, gt=0)
    llm_price: Price = Price(endpoint_alias='llm', effective_utc='unspecified', source='unavailable')
    embedding_price: Price = Price(endpoint_alias='embedding', effective_utc='unspecified', source='unavailable')
