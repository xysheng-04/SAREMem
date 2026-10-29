class EvidencePipelineError(RuntimeError):
    """Base class for all strict Evidence execution failures."""

class PlanDecodeError(EvidencePipelineError):
    """Planner output was not a JSON object."""

class PlanContractError(EvidencePipelineError):
    """Planner output violated the schema or a cross-field invariant."""

class ToolInputError(EvidencePipelineError):
    """A tool call failed strict argument validation."""
