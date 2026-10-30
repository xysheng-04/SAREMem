"""Finite production policy and fixed, non-payload failure vocabulary."""
from dataclasses import dataclass, fields
import math
import os


class SearchFault(Exception):
    def __init__(self, code, *, terminal=False):
        self.code = code
        self.terminal = terminal
        super().__init__(code)


@dataclass(frozen=True)
class Policy:
    total_seconds: float = 1680
    simple_seconds: float = 1140
    legacy_seconds: float = 420
    seed_reserve_seconds: float = 90
    cleanup_seconds: float = 5
    provider_timeout_seconds: float = 300
    connect_timeout_seconds: float = 30
    source_bytes: int = 256_000_000
    max_wire_bytes: int = 64_000_000
    max_concurrent_searches: int = 2
    worker_memory_mb: int = 2048

    def __post_init__(self):
        times = (self.total_seconds, self.simple_seconds, self.legacy_seconds,
                 self.seed_reserve_seconds, self.cleanup_seconds)
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError('invalid production time budget')
        # AML permits 1800 seconds. Keep 120 seconds outside the application
        # budget for response transport, TLS and the participant's gateway.
        if self.total_seconds > 1680:
            raise ValueError('production total budget must not exceed 1680 seconds')
        if (not math.isfinite(self.provider_timeout_seconds) or
                not 0 < self.provider_timeout_seconds <= 600 or
                not math.isfinite(self.connect_timeout_seconds) or
                not 0 < self.connect_timeout_seconds <= min(60, self.provider_timeout_seconds)):
            raise ValueError('invalid provider timeouts')
        if self.total_seconds <= self.legacy_seconds + self.seed_reserve_seconds + 3*self.cleanup_seconds:
            raise ValueError('total deadline must reserve legacy, seed and process cleanup')
        if self.source_bytes < 1 or self.max_wire_bytes < 1024 or self.max_concurrent_searches < 1 or self.worker_memory_mb < 512:
            raise ValueError('invalid production capacity')

    @classmethod
    def from_env(cls):
        return cls(**{field.name: field.type(os.environ['AML_PROD_'+field.name.upper()])
                      for field in fields(cls)
                      if 'AML_PROD_'+field.name.upper() in os.environ})


def classify(exc):
    """Never serialize exception messages, attributes with payloads, or tracebacks."""
    import sqlite3
    from pydantic import ValidationError
    from ..errors import Forbidden, Unauthorized, UpstreamUnavailable
    from ..postseed.contracts import Failure
    from ..postseed.seed_port import SeedError
    import sys
    module = sys.modules.get('aml_api.search_images')
    if module is not None and isinstance(exc, module.ImageInputError):
        return SearchFault(exc.code, terminal=True)
    if isinstance(exc, SearchFault):
        return exc
    if isinstance(exc, (Forbidden, Unauthorized)):
        return SearchFault('scope_denied', terminal=True)
    if isinstance(exc, TimeoutError):
        return SearchFault('deadline')
    if isinstance(exc, ConnectionError):
        return SearchFault('provider')
    if isinstance(exc, (sqlite3.Error, OSError)):
        return SearchFault('source_unavailable', terminal=True)
    if isinstance(exc, UpstreamUnavailable):
        return SearchFault('provider')
    if isinstance(exc, ValidationError):
        return SearchFault('schema')
    if isinstance(exc, (Failure, SeedError)):
        code = getattr(exc, 'code', None)
        # SeedError.args are intentionally not emitted (third-party errors may be wrapped).
        mapping = {'assessment_unavailable':'provider', 'assessment_schema': 'schema', 'duplicate_fragment': 'candidate_id',
                   'unknown_fragment': 'candidate_id', 'missing_fragment': 'candidate_id',
                   'ranking_contract': 'ranking', 'membership_changed': 'membership',
                   'serialization_content': 'response_validation', 'source_read': 'source_unavailable',
                   'source_binding': 'scope_denied', 'request_binding': 'scope_denied',
                   'source_content': 'source_integrity', 'seed_binding': 'scope_denied',
                   'budget_limited': 'capacity', 'source_drift': 'source_changed',
                   'upstream_unavailable': 'provider', 'provider_json': 'schema',
                   'provider_length': 'schema', 'provider_failure': 'schema'}
        result = mapping.get(code, 'seed_contract' if isinstance(exc, SeedError) else 'internal')
        return SearchFault(result, terminal=result in {'scope_denied','source_unavailable','source_integrity'})
    return SearchFault('internal')
