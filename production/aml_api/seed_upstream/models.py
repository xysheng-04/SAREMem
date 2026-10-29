from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, conint, confloat

from .enums import (
    AssemblyMode, Capability, DropReason, ExpansionToolName, IncrementalRole,
    PlanReasonCode, PurposeCode, RerankPolicy, RetrievalMode, SlotName,
    StopReason, ToolCaller, ToolCategory, ToolName,
)

StrictId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

JsonScalar = str | int | float | bool | None

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class SeedTopNSelector(StrictModel):
    kind: Literal["seed_top_n"]
    n: conint(ge=1, le=100)

class ToolResultSelector(StrictModel):
    kind: Literal["tool_result"]
    call_id: Annotated[str, StringConstraints(pattern=r"^t[1-9][0-9]*$")]
    top_n: conint(ge=1, le=100)

AnchorSelector = Annotated[
    Union[SeedTopNSelector, ToolResultSelector], Field(discriminator="kind")
]

class ToolCallSpec(StrictModel):
    call_id: Annotated[str, StringConstraints(pattern=r"^t[1-9][0-9]*$")]
    tool: ExpansionToolName
    anchors: AnchorSelector
    arguments: dict[str, JsonScalar]
    purpose_code: PurposeCode
    fills_slots: list[SlotName]

class EvidencePlan(StrictModel):
    schema_version: Literal[1]
    mode: RetrievalMode
    capabilities: list[Capability]
    required_slots: list[SlotName]
    rerank_policy: RerankPolicy
    tool_calls: list[ToolCallSpec]
    assembly_mode: AssemblyMode
    seed_reserve: conint(ge=1)
    expansion_budget: conint(ge=0)
    confidence: confloat(ge=0.0, le=1.0)
    reason_code: PlanReasonCode

class PlannerEnvelope(StrictModel):
    query_anchors: list[str] = Field(min_length=1, max_length=4)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    visual_description: str
    plan: EvidencePlan

class ValidatedEvidencePlan(EvidencePlan):
    """Frozen plan emitted only after model and cross-field validation."""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

class AdjacentArgs(StrictModel):
    direction: Literal["before", "after", "both"] = "both"
    window: conint(ge=1, le=20) = 1

class SameSessionArgs(StrictModel):
    ranking: Literal["query", "distance", "mixed"] = "mixed"
    limit: conint(ge=1, le=100) = 12

class TimeWindowArgs(StrictModel):
    direction: Literal["before", "after", "around"] = "around"
    time_window_ms: conint(ge=0) = 86_400_000
    same_session_only: bool = False
    limit: conint(ge=1, le=100) = 12

class EntityArgs(StrictModel):
    entity: StrictId
    limit: conint(ge=1, le=100) = 12

class CrossSessionArgs(EntityArgs):
    """Entity-constrained cross-session lookup arguments."""

class RelatedMemoryArgs(StrictModel):
    entity: str = ""
    event_terms: str = ""
    limit: conint(ge=1, le=100) = 12

class StateArgs(StrictModel):
    entity: StrictId
    attribute: str = ""
    limit: conint(ge=1, le=100) = 12

class StateAtTimeArgs(StateArgs):
    target_timestamp_ms: int

class SemanticBridgeArgs(StrictModel):
    bridge_queries: list[str] = Field(default_factory=list, max_length=4)
    limit: conint(ge=1, le=100) = 12

class VisualCoreferenceArgs(StrictModel):
    coreference_type: Literal["person", "object", "place", "interface", "scene", "auto"] = "auto"
    limit: conint(ge=1, le=100) = 12
