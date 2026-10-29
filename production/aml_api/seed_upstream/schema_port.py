"""Frozen planner metadata only. No registry, tool objects, or executable tools."""
import json
from pydantic import ValidationError
from . import models
from .enums import Capability, ExpansionToolName
from .errors import ToolInputError

# Original build_registry insertion order; purpose equals name (no class docstrings).
_ARGUMENTS = (
    ('context.adjacent', models.AdjacentArgs),
    ('context.same_session', models.SameSessionArgs),
    ('time.window', models.TimeWindowArgs),
    ('entity.mentions', models.EntityArgs),
    ('session.cross', models.CrossSessionArgs),
    ('entity.related_memory', models.RelatedMemoryArgs),
    ('state.timeline', models.StateArgs),
    ('state.updates', models.StateArgs),
    ('state.latest', models.StateArgs),
    ('state.at_time', models.StateAtTimeArgs),
    ('semantic.bridge', models.SemanticBridgeArgs),
    ('visual.coreference', models.VisualCoreferenceArgs),
)


class SchemaPort:
    def planner_schema(self):
        return {'capabilities':[item.value for item in Capability],
                'tools':{name:{'purpose':name,'arguments':model.model_json_schema(),'max_fanout':12}
                         for name,model in _ARGUMENTS},
                'plan_schema':models.EvidencePlan.model_json_schema()}

    def validate_planner_arguments(self,name: ExpansionToolName,arguments: dict):
        model=dict(_ARGUMENTS).get(name.value)
        if model is None:
            raise ToolInputError(f'unregistered memory tool: {name.value}')
        try:
            return model.model_validate_json(json.dumps(arguments),strict=True)
        except ValidationError as exc:
            raise ToolInputError(f'invalid arguments for {name.value}: {exc}') from exc
