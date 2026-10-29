from enum import Enum


class Capability(str, Enum):
    DIRECT_FACT = "direct_fact_lookup"
    VISUAL_GROUNDING = "visual_grounding"
    CROSS_TURN = "cross_turn_reference"
    CROSS_SESSION = "cross_session"
    ENTITY_RELATION = "entity_relation"
    TEMPORAL_ORDERING = "temporal_ordering"
    LATEST_STATE = "latest_state_resolution"
    STATE_AT_TIME = "state_at_time"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation_or_set"
    ABSTENTION = "negative_evidence_or_abstention"


class ToolName(str, Enum):
    SEED_RETRIEVE = "seed.retrieve"
    SEED_RERANK = "seed.rerank"
    CONTEXT_ADJACENT = "context.adjacent"
    CONTEXT_SAME_SESSION = "context.same_session"
    TIME_WINDOW = "time.window"
    ENTITY_MENTIONS = "entity.mentions"
    SESSION_CROSS = "session.cross"
    ENTITY_RELATED_MEMORY = "entity.related_memory"
    STATE_TIMELINE = "state.timeline"
    STATE_UPDATES = "state.updates"
    STATE_LATEST = "state.latest"
    STATE_AT_TIME = "state.at_time"
    SEMANTIC_BRIDGE = "semantic.bridge"
    VISUAL_COREFERENCE = "visual.coreference"
    VERIFY_DIRECT = "verify.direct"
    VERIFY_INCREMENTAL = "verify.incremental"
    VERIFY_COMPLETENESS = "verify.completeness"
    ASSEMBLE_RANKED = "assemble.ranked"
    ASSEMBLE_CHAIN = "assemble.chain"
    ASSEMBLE_TIMELINE = "assemble.timeline"
    ASSEMBLE_GROUPED = "assemble.grouped"


class ExpansionToolName(str, Enum):
    CONTEXT_ADJACENT = "context.adjacent"
    CONTEXT_SAME_SESSION = "context.same_session"
    TIME_WINDOW = "time.window"
    ENTITY_MENTIONS = "entity.mentions"
    SESSION_CROSS = "session.cross"
    ENTITY_RELATED_MEMORY = "entity.related_memory"
    STATE_TIMELINE = "state.timeline"
    STATE_UPDATES = "state.updates"
    STATE_LATEST = "state.latest"
    STATE_AT_TIME = "state.at_time"
    SEMANTIC_BRIDGE = "semantic.bridge"
    VISUAL_COREFERENCE = "visual.coreference"


class SlotName(str, Enum):
    FACT = "fact"
    ENTITY = "entity"
    RELATION = "relation"
    EVENT = "event"
    EVENT_TIME = "event_time"
    STATE_BEFORE = "state_before"
    STATE_AFTER = "state_after"
    CURRENT_STATE = "current_state"
    HISTORICAL_STATE = "historical_state"
    CROSS_SESSION_CONTEXT = "cross_session_context"
    VISUAL_IDENTITY = "visual_identity"
    COMPARISON_SET = "comparison_set"
    NEGATIVE_EVIDENCE = "negative_evidence"


class ToolCategory(str, Enum):
    SEED = "seed"
    EXPANSION = "expansion"
    VERIFICATION = "verification"
    ASSEMBLY = "assembly"


class ToolCaller(str, Enum):
    PLANNER = "planner"
    ORCHESTRATOR_ONLY = "orchestrator_only"


class RetrievalMode(str, Enum):
    DIRECT = "direct"
    ASSEMBLE = "assemble"
    HYBRID = "hybrid"


class AssemblyMode(str, Enum):
    RANKED = "ranked"
    CHAIN = "chain"
    TIMELINE = "timeline"
    GROUPED = "grouped"


class RerankPolicy(str, Enum):
    ON = "on"
    OFF = "off"


class PurposeCode(str, Enum):
    RESOLVE_REFERENCE = "resolve_reference"
    FIND_TEMPORAL_CONTEXT = "find_temporal_context"
    FIND_ENTITY_CONTEXT = "find_entity_context"
    FIND_STATE_CHANGE = "find_state_change"
    FIND_CROSS_SESSION_CONTEXT = "find_cross_session_context"
    FIND_SEMANTIC_COMPLEMENT = "find_semantic_complement"
    FIND_VISUAL_MATCH = "find_visual_match"


class PlanReasonCode(str, Enum):
    DIRECT_SUFFICIENT = "direct_sufficient"
    NEED_CONTEXT = "need_context"
    NEED_TEMPORAL = "need_temporal"
    NEED_ENTITY = "need_entity"
    NEED_STATE = "need_state"
    NEED_VISUAL = "need_visual"
    NEED_MULTI_EVIDENCE = "need_multi_evidence"


class IncrementalRole(str, Enum):
    NEW_FACT = "new_fact"
    CLARIFICATION = "clarification"
    CORROBORATION = "corroboration"
    PREVIOUS_STATE = "previous_state"
    UPDATE = "update"
    REDUNDANT = "redundant"
    CONFLICT = "conflict"
    IRRELEVANT = "irrelevant"


class DropReason(str, Enum):
    BUDGET = "budget"
    REDUNDANT = "redundant"
    LOW_INCREMENTAL = "low_incremental_support"
    MISSING_TIMESTAMP = "missing_timestamp"
    INVALID_RELATION = "invalid_relation"
    MEDIA_BUDGET = "media_budget"
    LOWER_RANK_BACKFILL = "lower_rank_backfill"


class StopReason(str, Enum):
    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DIRECT = "direct"
    NO_CANDIDATES = "no_candidates"
