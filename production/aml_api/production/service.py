"""Only this API adapter enables simple -> legacy -> seed recovery."""
import os
import threading
import time

from ..errors import BadRequest, Forbidden, ServiceError, UpstreamUnavailable
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
    def __init__(self, settings, *, policy=None, runner=None, diagnostics=None, add_service=None, add_runner=None):
        self.settings = settings
        self.policy = policy or Policy.from_env()
        self.runner = runner or StageRunner()
        self.diagnostics = diagnostics  # Callable exists only in the separate testing entrypoint.
        self.add_service = add_service
        self.add_runner = add_runner  # Python-only injection for offline fault verification.
        mode=os.environ.get('AML_SUFFICIENCY_MODE','off')
        if mode not in {'off','on'}:raise ValueError('invalid sufficiency mode')
        self.sufficiency_config=None
        if mode=='on':
            from ..context_sufficiency import configuration
            self.sufficiency_config=configuration()
        self.slots = threading.BoundedSemaphore(self.policy.max_concurrent_searches)
        add_capacity = int(os.environ.get('AML_PROD_MAX_CONCURRENT_ADDS', '1'))
        if not 1 <= add_capacity <= 32: raise ValueError('invalid Add capacity')
        self.add_slots = threading.BoundedSemaphore(add_capacity)

    def add(self, request):
        if self.add_service is not None:
            return self.add_service.add(request)
        from .add_control import add
        import sqlite3
        if not self.add_slots.acquire(blocking=False):
            raise CapacityExceeded('memory service busy',retry_after=1)
        try:
            return add(self.settings, request, self.policy, runner=self.add_runner)
        except sqlite3.Error:
            raise UpstreamUnavailable('Add persistence temporarily unavailable') from None
        finally:
            self.add_slots.release()

    def search(self, request):
        entered = request_started.get()
        started = entered if entered is not None else time.monotonic()
        # Validation and authorization failures are outside the fallback chain.
        from .diagnostics import stage as diagnostic_stage
        diagnostic_stage("image_processing")
        validate_parts(request.query, total_limit=MAX_SEARCH_IMAGE_BYTES)
        if not self.slots.acquire(blocking=False):
            raise CapacityExceeded('memory service busy', retry_after=1)
        events = [] if self.diagnostics is not None else None
        final = 'error'
        seed = None
        p = self.policy
        end = started + p.total_seconds
        retrieval_end=end-(self.sufficiency_config[1] if self.sufficiency_config else 0)
        def publish(value):
            nonlocal seed
            if seed is None:
                seed = value
        try:
            for stage in ('simple', 'legacy', 'seed'):
                from .diagnostics import stage as diagnostic_stage, error as diagnostic_error
                diagnostic_stage(stage)
                begin = time.monotonic()
                if stage == 'simple':
                    deadline = min(begin+p.simple_seconds, retrieval_end-p.legacy_seconds-p.seed_reserve_seconds-3*p.cleanup_seconds)
                elif stage == 'legacy':
                    deadline = min(begin+p.legacy_seconds, retrieval_end-p.seed_reserve_seconds-2*p.cleanup_seconds)
                else:
                    deadline = retrieval_end-p.cleanup_seconds
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
                        detail = result.get('diagnostic',{})
                        diagnostic_stage(detail.get('stage')); diagnostic_error(detail.get('code'))
                        raise SearchFault(result['error'], terminal=result.get('terminal', False))
                    # Response construction is still within this stage's recoverable boundary.
                    diagnostic_stage('response_validation')
                    response = SearchResponse.model_validate(result['response'])
                    response.model_dump_json()
                    if time.monotonic() > end:
                        raise SearchFault('deadline')
                    from .diagnostics import active
                    if active.get() is not None: active.get()['code'] = 'http_error'
                    final = stage if response.data else 'empty'
                    break
                except Exception as exc:
                    fault = classify(exc)
                    code, terminal = fault.code, fault.terminal
                    from .diagnostics import active
                    if active.get() is not None and active.get().get("code") == "http_error": diagnostic_error(code)
                    if terminal:
                        if code == 'query_image_invalid':
                            raise BadRequest('invalid query image') from None
                        if code == 'scope_denied':
                            raise Forbidden('memory scope validation failed') from None
                        raise ServiceError('internal memory service error') from None
                    if seed is None:
                        if code == 'deadline':
                            raise SearchDeadlineExceeded('memory search deadline exceeded') from None
                        raise UpstreamUnavailable('seed retrieval unavailable') from None
                finally:
                    if events is not None:
                        events.append({'path':stage,'code':code, **stats,
                                       'elapsed_ms':round((time.monotonic()-begin)*1000,3),
                                       'deadline_exhausted':code=='deadline'})
            if final != 'error':
                return self._sufficiency(request,response,p,end,events)
            if code == 'deadline' or time.monotonic() >= end:
                raise SearchDeadlineExceeded('memory search deadline exceeded')
            raise UpstreamUnavailable('memory search unavailable')
        finally:
            self.slots.release()
            if events is not None:
                try:
                    self.diagnostics({'final':final, 'stages':events,
                                      'elapsed_ms':round((time.monotonic()-started)*1000,3)})
                except Exception: pass


    def _sufficiency(self,request,response,p,end,events):
        if self.sufficiency_config is None or not response.data:return response
        import copy
        from .diagnostics import stage as diagnostic_stage
        from .sufficiency_fallback import recover, unavailable
        from .validation import finalize
        diagnostic_stage('sufficiency')
        begin=time.monotonic()
        deadline=min(begin+270.0,end-p.cleanup_seconds)
        snapshot=response.model_dump(mode='json')
        def attempt(attempt_deadline):
            diagnostic_stage('sufficiency')
            try:
                result=self.runner.run('sufficiency',self.settings,request,p,attempt_deadline,
                                       copy.deepcopy(snapshot),lambda _:None)
            except ServiceError:
                raise
            except Exception as exc:
                fault=classify(exc)
                if fault.code in {'deadline','retry_budget'}:
                    raise SearchDeadlineExceeded('memory search deadline exceeded') from None
                raise unavailable(fault.code,fault.terminal) from None
            if 'error' in result:
                fault=result['error']
                if fault in {'deadline','retry_budget'}:
                    raise SearchDeadlineExceeded('memory search deadline exceeded')
                raise unavailable(fault,result.get('terminal',False))
            # Malformed or mutated evidence is never a retry/fallback permission.
            try:
                value=SearchResponse.model_validate(result['response'])
                if value.data and value.model_dump(mode='json')!=snapshot:
                    raise ValueError('sufficiency evidence changed')
                return value
            except Exception:
                raise UpstreamUnavailable('context sufficiency unavailable') from None
        def validate_original():
            try:
                value=finalize(self.settings,request,p,copy.deepcopy(snapshot),'simple',{})
                if value!=snapshot:raise ValueError('evidence changed before fallback')
                return SearchResponse.model_validate(value)
            except Exception:
                raise UpstreamUnavailable('context sufficiency unavailable') from None
        code='error'
        try:
            value=recover(attempt,validate_original,deadline,len(response.data),
                          attempt_seconds=min(90.0,self.sufficiency_config[1]))
            code='ok'
            return value
        finally:
            if events is not None:
                events.append({'path':'sufficiency','code':code,
                               'elapsed_ms':round((time.monotonic()-begin)*1000,3)})
