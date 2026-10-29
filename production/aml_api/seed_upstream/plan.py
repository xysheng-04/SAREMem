from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .config import EvidenceSettings
from .enums import (
    AssemblyMode, Capability, ExpansionToolName, RetrievalMode, SlotName, ToolName,
)
from .errors import PlanContractError, PlanDecodeError, ToolInputError
from .models import EvidencePlan, ToolResultSelector, ValidatedEvidencePlan

TEMPORAL_CAPS = {Capability.TEMPORAL_ORDERING, Capability.LATEST_STATE, Capability.STATE_AT_TIME}
TEMPORAL_TOOLS = {
    ExpansionToolName.TIME_WINDOW, ExpansionToolName.STATE_TIMELINE,
    ExpansionToolName.STATE_UPDATES, ExpansionToolName.STATE_LATEST,
    ExpansionToolName.STATE_AT_TIME,
}
VISUAL_TOOLS = {ExpansionToolName.VISUAL_COREFERENCE}
DIRECT_SLOTS = {SlotName.FACT, SlotName.VISUAL_IDENTITY, SlotName.NEGATIVE_EVIDENCE}
_UNSAFE_ARGUMENT = re.compile(
    r"(?:^|\s)(?:select\s+.+\s+from|insert\s+into|update\s+.+\s+set|delete\s+from|"
    r"import\s+\w+|from\s+\w+\s+import|def\s+\w+)|(?:^|[\s'\"])(?:[a-zA-Z]:[\\/]|/[^\s]+|\.\.[\\/])",
    re.IGNORECASE,
)


def decode_plan(payload: str | dict[str, Any]) -> EvidencePlan:
    try:
        raw = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise PlanDecodeError(f"planner output is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise PlanDecodeError("planner output must be a JSON object")
    try:
        return EvidencePlan.model_validate_json(json.dumps(raw), strict=True)
    except ValidationError as exc:
        raise PlanContractError(str(exc)) from exc


def validate_plan(
    plan: EvidencePlan,
    settings: EvidenceSettings,
    registry: Any,
    *,
    has_visual_signal: bool,
) -> ValidatedEvidencePlan:
    capabilities = list(plan.capabilities)
    if not capabilities or len(set(capabilities)) != len(capabilities):
        raise PlanContractError("capabilities must be non-empty and unique")
    if set(item.value for item in Capability) & set(item.value for item in ToolName):
        raise PlanContractError("Capability and ToolName value sets must be disjoint")
    if len(plan.tool_calls) > settings.max_tool_calls:
        raise PlanContractError("tool call budget exceeded")
    if plan.seed_reserve + plan.expansion_budget > settings.max_candidates:
        raise PlanContractError("candidate budget exceeded")
    if plan.mode is RetrievalMode.DIRECT:
        if plan.tool_calls or plan.assembly_mode is not AssemblyMode.RANKED:
            raise PlanContractError("direct plan requires no tools and ranked assembly")
        if plan.expansion_budget != 0 or any(slot not in DIRECT_SLOTS for slot in plan.required_slots):
            raise PlanContractError("direct plan has invalid slots or expansion budget")
    elif plan.mode is RetrievalMode.ASSEMBLE:
        if not plan.tool_calls or plan.assembly_mode not in {
            AssemblyMode.CHAIN, AssemblyMode.TIMELINE, AssemblyMode.GROUPED
        }:
            raise PlanContractError("assemble plan requires tools and chain/timeline/grouped assembly")
        if plan.expansion_budget <= 0 or not plan.required_slots:
            raise PlanContractError("assemble plan requires expansion budget and slots")
    elif not plan.tool_calls or plan.expansion_budget <= 0:
        raise PlanContractError("hybrid plan requires tools and expansion budget")

    call_ids = [call.call_id for call in plan.tool_calls]
    expected = [f"t{i}" for i in range(1, len(call_ids) + 1)]
    if call_ids != expected:
        raise PlanContractError("call IDs must be unique, consecutive, and ordered")
    covered: set[SlotName] = set()
    depth: dict[str, int] = {}
    for index, call in enumerate(plan.tool_calls):
        if any(slot not in plan.required_slots for slot in call.fills_slots):
            raise PlanContractError("tool fills_slots must be a subset of required_slots")
        covered.update(call.fills_slots)
        if isinstance(call.anchors, ToolResultSelector):
            if call.anchors.call_id not in call_ids[:index]:
                raise PlanContractError("tool_result anchor must reference a prior call")
            depth[call.call_id] = depth[call.anchors.call_id] + 1
        else:
            if call.anchors.n > settings.max_candidates:
                raise PlanContractError("seed anchor budget exceeded")
            depth[call.call_id] = 1
        if depth[call.call_id] > settings.max_depth:
            raise PlanContractError("tool dependency depth exceeded")
        if call.tool in VISUAL_TOOLS and not has_visual_signal:
            raise PlanContractError("visual tool requires a visual query or available visual evidence")
        if any(isinstance(value, str) and _UNSAFE_ARGUMENT.search(value)
               for value in call.arguments.values()):
            raise PlanContractError("tool arguments must not contain SQL, code, or file paths")
        try:
            registry.validate_planner_arguments(call.tool, call.arguments)
        except ToolInputError as exc:
            raise PlanContractError(str(exc)) from exc
    if plan.mode is not RetrievalMode.DIRECT and plan.required_slots and not covered:
        raise PlanContractError("expansion tools do not cover any required slot")
    if plan.assembly_mode is AssemblyMode.TIMELINE:
        if not (set(plan.capabilities) & TEMPORAL_CAPS) or not (
            {call.tool for call in plan.tool_calls} & TEMPORAL_TOOLS
        ):
            raise PlanContractError("timeline assembly requires temporal/state capability and tool")
    return ValidatedEvidencePlan.model_validate_json(plan.model_dump_json(), strict=True)


def direct_plan(settings: EvidenceSettings) -> ValidatedEvidencePlan:
    return ValidatedEvidencePlan.model_validate_json(json.dumps({
        "schema_version": 1,
        "mode": "direct",
        "capabilities": ["direct_fact_lookup"],
        "required_slots": ["fact"],
        "rerank_policy": "on",
        "tool_calls": [],
        "assembly_mode": "ranked",
        "seed_reserve": min(settings.seed_reserve, settings.max_candidates),
        "expansion_budget": 0,
        "confidence": 1.0,
        "reason_code": "direct_sufficient",
    }), strict=True)
