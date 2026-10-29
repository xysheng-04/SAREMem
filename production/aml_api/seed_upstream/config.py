from __future__ import annotations

from dataclasses import dataclass
import os


def _int(name: str, default: int, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class EvidenceSettings:
    mode: str = "auto"
    error_policy: str = "fail_fast"
    trace_level: str = "compact"
    trace_max_bytes: int = 65536
    max_tool_calls: int = 8
    max_depth: int = 2
    max_candidates: int = 48
    max_fanout_per_call: int = 12
    seed_reserve: int = 10
    expansion_budget: int = 10
    forced_plan_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "forced", "auto"}:
            raise ValueError("AML_EVIDENCE_MODE must be shadow, forced, or auto")
        if self.error_policy not in {"fail_fast", "api_fallback"}:
            raise ValueError("AML_EVIDENCE_ERROR_POLICY must be fail_fast or api_fallback")
        if self.trace_level not in {"compact", "none"}:
            raise ValueError("AML_EVIDENCE_TRACE_LEVEL must be compact or none")
        if self.seed_reserve + self.expansion_budget > self.max_candidates:
            raise ValueError("seed_reserve + expansion_budget exceeds max_candidates")

    @classmethod
    def from_env(cls) -> "EvidenceSettings":
        return cls(
            mode=os.getenv("AML_EVIDENCE_MODE", "auto").strip().lower(),
            error_policy=os.getenv("AML_EVIDENCE_ERROR_POLICY", "fail_fast").strip().lower(),
            trace_level=os.getenv("AML_EVIDENCE_TRACE_LEVEL", "compact").strip().lower(),
            trace_max_bytes=_int("AML_EVIDENCE_TRACE_MAX_BYTES", 65536, 1024),
            max_tool_calls=_int("AML_EVIDENCE_MAX_TOOL_CALLS", 8, 1),
            max_depth=_int("AML_EVIDENCE_MAX_DEPTH", 2, 1),
            max_candidates=_int("AML_EVIDENCE_MAX_CANDIDATES", 48, 1),
            max_fanout_per_call=_int("AML_EVIDENCE_MAX_FANOUT_PER_CALL", 12, 1),
            seed_reserve=_int("AML_EVIDENCE_SEED_RESERVE", 10, 1),
            expansion_budget=_int("AML_EVIDENCE_EXPANSION_BUDGET", 10, 0),
            forced_plan_path=os.getenv("AML_EVIDENCE_FORCED_PLAN") or None,
        )
