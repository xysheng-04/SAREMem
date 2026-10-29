"""Only this API adapter enables simple -> legacy -> seed recovery."""
import threading
import time

from ..errors import Forbidden, ServiceError, UpstreamUnavailable
from ..media import validate_parts, MAX_SEARCH_IMAGE_BYTES
from ..schemas import SearchResponse
from ..service import RouteAService
from ..store import ReadOnlySQLiteStoreMixin, SQLiteStore
from .policy import Policy, SearchFault, classify
from .process import StageRunner
from .request_clock import request_started
from .http_errors import CapacityExceeded, SearchDeadlineExceeded


class ExistingStore(ReadOnlySQLiteStoreMixin, SQLiteStore):
    """Initialize only through the explicit init command; use original Add transactions."""
    _connection = SQLiteStore._connection


class ProductionService:
    def __init__(self, settings, *, policy=None, runner=None, diagnostics=None, add_service=None):
        self.settings = settings
        self.policy = policy or Policy.from_env()
        self.runner = runner or StageRunner()
        self.diagnostics = diagnostics  # Callable exists only in the separate testing entrypoint.
        self.add_service = add_service
        self.slots = threading.BoundedSemaphore(self.policy.max_concurrent_searches)

    def add(self, request):
        if self.add_service is not None:
            return self.add_service.add(request)
        service = RouteAService(self.settings, store=ExistingStore(self.settings.storage_path))
        try:
            return service.add(request)
        finally:
            service.vlm.client.close()
            service.embedder.client.close()

    def search(self, request):
        entered = request_started.get()
        started = entered if entered is not None else time.monotonic()
        # Validation and authorization failures are outside the fallback chain.
        validate_parts(request.query, total_limit=MAX_SEARCH_IMAGE_BYTES)
        if not self.slots.acquire(blocking=False):
            raise CapacityExceeded('memory service busy', retry_after=1)
        events = [] if self.diagnostics is not None else None
        final = 'error'
        seed = None
        p = self.policy
        end = started + p.total_seconds
        def publish(value):
            nonlocal seed
            if seed is None:
                seed = value
        try:
            for stage in ('simple', 'legacy', 'seed'):
                begin = time.monotonic()
                if stage == 'simple':
                    deadline = min(begin+p.simple_seconds, end-p.legacy_seconds-p.seed_reserve_seconds-3*p.cleanup_seconds)
                elif stage == 'legacy':
                    deadline = min(begin+p.legacy_seconds, end-p.seed_reserve_seconds-2*p.cleanup_seconds)
                else:
                    deadline = end-p.cleanup_seconds
                stats = {'phase':stage, 'candidate_count':None, 'dropped':0,
                         'media_filtered':False, 'seed_existing':seed is not None}
                code = 'ok'
                terminal = False
                try:
                    if deadline <= begin:
                        raise SearchFault('deadline')
                    result = self.runner.run(stage, self.settings, request, p, deadline, seed, publish)
                    stats.update(result.get('stats', {}))
                    if 'error' in result:
                        raise SearchFault(result['error'], terminal=result.get('terminal', False))
                    # Response construction is still within this stage's recoverable boundary.
                    response = SearchResponse.model_validate(result['response'])
                    response.model_dump_json()
                    if time.monotonic() > end:
                        raise SearchFault('deadline')
                    final = stage if response.data else 'empty'
                    return response
                except Exception as exc:
                    fault = classify(exc)
                    code, terminal = fault.code, fault.terminal
                    if terminal:
                        if code == 'scope_denied':
                            raise Forbidden('memory scope validation failed') from None
                        raise ServiceError('internal memory service error') from None
                finally:
                    if events is not None:
                        events.append({'path':stage,'code':code, **stats,
                                       'elapsed_ms':round((time.monotonic()-begin)*1000,3),
                                       'deadline_exhausted':code=='deadline'})
            if code == 'deadline' or time.monotonic() >= end:
                raise SearchDeadlineExceeded('memory search deadline exceeded')
            raise UpstreamUnavailable('memory search unavailable')
        finally:
            self.slots.release()
            if events is not None:
                self.diagnostics({'final':final, 'stages':events,
                                  'elapsed_ms':round((time.monotonic()-started)*1000,3)})
