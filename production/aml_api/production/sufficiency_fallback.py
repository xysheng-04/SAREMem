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

def recover(attempt, validate_original, deadline, evidence_count, attempt_seconds=90.0):
    record = active.get()
    if record is not None:
        record.update(sufficiency_attempts=0, sufficiency_failures_503=0,
                      sufficiency_fallback=False, sufficiency_outcome='error',
                      evidence_count=evidence_count, sufficiency_stop_reason='none')
    def check(attempt_deadline=None):
        cancelled = request_cancelled.get()
        now = time.monotonic()
        reason = ('request_cancelled' if cancelled is not None and cancelled.is_set()
                  else 'recovery_deadline' if now >= deadline
                  else 'attempt_deadline' if attempt_deadline is not None and now >= attempt_deadline
                  else None)
        if reason is not None:
            if record is not None: record['sufficiency_stop_reason'] = reason
            raise SearchDeadlineExceeded('memory search deadline exceeded')
    for index in range(1, 4):
        check()
        if record is not None: record['sufficiency_attempts'] = index
        attempt_deadline = min(time.monotonic()+attempt_seconds, deadline)
        try:
            value = attempt(attempt_deadline)
            check(attempt_deadline)
        except Sufficiency503:
            if record is not None: record['sufficiency_failures_503'] = index
            check(attempt_deadline)
            if index < 3: continue
            value = validate_original()
            check()
            if record is not None:
                record.update(sufficiency_fallback=True, sufficiency_outcome='fallback', code='http_error')
            return value
        except SearchDeadlineExceeded:
            check(attempt_deadline)
            if record is not None: record['sufficiency_stop_reason'] = 'calculation_deadline_or_retry_budget'
            raise
        if record is not None:
            record.update(sufficiency_outcome='pass' if value.data else 'refuse', code='http_error')
        return value
