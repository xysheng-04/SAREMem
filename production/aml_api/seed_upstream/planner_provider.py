from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import EvidenceSettings
from .errors import PlanContractError
from .models import PlannerEnvelope, ValidatedEvidencePlan
from .plan import decode_plan, validate_plan


def plan_evidence_query(
    vlm: Any,
    registry: Any,
    settings: EvidenceSettings,
    question: str,
    options: list[str],
    images: list[str],
    *,
    has_visual_signal: bool,
) -> tuple[ValidatedEvidencePlan, dict[str, Any]]:
    schema = registry.planner_schema()
    prompt = f"""Plan bounded memory evidence retrieval; never answer the question.
Question: {question or '(image-only query)'}
Options are retrieval hints only: {json.dumps(options, ensure_ascii=False)}

Capability enum (capabilities, NOT tools): {json.dumps(schema['capabilities'])}
Planner-callable tools and exact argument schemas: {json.dumps(schema['tools'], ensure_ascii=False)}
Return one strict object with query_anchors (1-4 short retrieval paraphrases), keywords (0-12
short terms), visual_description (faithful and short, empty without a query image), and plan.
Required plan JSON schema: {json.dumps(schema['plan_schema'], ensure_ascii=False)}

Rules:
- direct: tool_calls=[], assembly_mode=ranked, expansion_budget=0.
- assemble: at least one tool; assembly_mode chain/timeline/grouped; nonzero expansion budget.
- hybrid: at least one tool and nonzero expansion budget; any assembly mode.
- assembly_mode is always required and never null.
- call IDs are t1,t2,...; anchors use seed_top_n or an earlier tool_result only.
- capability values and tool names are different namespaces.
- output JSON only; do not reveal reasoning or answer the benchmark question.
Examples:
{{"query_anchors":["requested fact"],"keywords":[],"visual_description":"","plan":{{"schema_version":1,"mode":"direct","capabilities":["direct_fact_lookup"],"required_slots":["fact"],"rerank_policy":"on","tool_calls":[],"assembly_mode":"ranked","seed_reserve":10,"expansion_budget":0,"confidence":0.9,"reason_code":"direct_sufficient"}}}}
{{"query_anchors":["prior turn reference"],"keywords":["reference"],"visual_description":"","plan":{{"schema_version":1,"mode":"assemble","capabilities":["cross_turn_reference"],"required_slots":["fact"],"rerank_policy":"on","tool_calls":[{{"call_id":"t1","tool":"context.adjacent","anchors":{{"kind":"seed_top_n","n":3}},"arguments":{{"direction":"both","window":1}},"purpose_code":"resolve_reference","fills_slots":["fact"]}}],"assembly_mode":"chain","seed_reserve":10,"expansion_budget":10,"confidence":0.8,"reason_code":"need_context"}}}}
"""
    last_error: Exception | None = None
    repair = ""
    for attempt in range(3):
        raw = vlm._json(
            system="You are a strict evidence retrieval planner. Return only the requested JSON.",
            content=vlm._multimodal_content(prompt + repair, images),
            max_tokens=1800,
        )
        try:
            envelope = PlannerEnvelope.model_validate_json(json.dumps(raw), strict=True)
            plan = validate_plan(
                envelope.plan, settings, registry, has_visual_signal=has_visual_signal
            )
            return plan, envelope.model_dump(mode="json")
        except (ValidationError, PlanContractError) as exc:
            last_error = exc
            if attempt == 2:
                break
            # Replanning remains strict: invalid values are never coerced or executed.  Feed back
            # only the schema error and enum lists, not hidden reasoning or benchmark answers.
            repair = (
                "\n\nYour previous object failed strict validation. Produce a completely new object."
                f"\nValidation error: {str(exc)[:1200]}"
                f"\nCapabilities MUST be chosen only from: {json.dumps(schema['capabilities'])}."
                "\nTool names belong only in plan.tool_calls[].tool, never in capabilities."
            )
    raise PlanContractError(f"planner contract failed after 3 attempts: {last_error}") from last_error
