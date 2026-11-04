"""Approved request-local three-503 recovery; no cross-request counters."""
import time
from ..errors import UpstreamUnavailable
from .http_errors import SearchDeadlineExceeded
from .diagnostics import active
from .request_clock import request_cancelled

class Sufficiency503(UpstreamUnavailable):
    """Only an eligible sufficiency computation failure, not evidence validation."""

BLOCKED = frozenset({'scope_denied','source_integrity','source_changed','source_unavailable',
    'response_validation','candidate_id','ranking','membership','seed_integrity',
    'seed_binding','query_image_invalid','provider_auth','cleanup_failed'})

def unavailable(code, terminal=False):
    cls = UpstreamUnavailable if terminal or code in BLOCKED else Sufficiency503
    return cls('context sufficiency unavailable')

def recover(attempt, validate_original, deadline, evidence_count):
    record = active.get()
    if record is not None:
        record.update(sufficiency_attempts=0, sufficiency_failures_503=0,
                      sufficiency_fallback=False, sufficiency_outcome='error',
                      evidence_count=evidence_count)
    def check():
        cancelled = request_cancelled.get()
        if time.monotonic() >= deadline or (cancelled is not None and cancelled.is_set()):
            raise SearchDeadlineExceeded('memory search deadline exceeded')
    for index in range(1, 4):
        check()
        if record is not None: record['sufficiency_attempts'] = index
        try:
            value = attempt()
            check()
        except Sufficiency503:
            if record is not None: record['sufficiency_failures_503'] = index
            check()
            if index < 3: continue
            value = validate_original()
            check()
            if record is not None:
                record.update(sufficiency_fallback=True, sufficiency_outcome='fallback', code='http_error')
            return value
        if record is not None:
            record.update(sufficiency_outcome='pass' if value.data else 'refuse', code='http_error')
        return value
