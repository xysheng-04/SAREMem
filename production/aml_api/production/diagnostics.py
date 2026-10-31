"""Failure-only fixed-schema diagnostics; bounded nonblocking producer."""
from contextvars import ContextVar
import json, os, queue, threading, time
from pathlib import Path

active = ContextVar('failure_diagnostic', default=None)
STAGES = {'sufficiency','parse_validation','authentication','admission','queue','image_processing','provider','compute_process','database_claim','database_commit','response_validation','response_send','simple','legacy','seed'}
CODES = {'http_error','deadline','cancelled','send_failed','worker_exit','internal','provider','schema','capacity','source_unavailable','scope_denied','query_image_invalid','cleanup_failed','source_integrity','source_changed','response_validation'}
def stage(value):
    record = active.get()
    if record is not None and value in STAGES:
        record['stage'] = value

def error(code):
    record = active.get()
    if record is not None:
        record['code'] = code if code in CODES else 'internal'

def validation_fields(errors):
    """Bounded fixed vocabulary only; never retain input, ctx, msg or unknown keys."""
    from itertools import islice
    from .http_errors import FIELDS, VALIDATION_MESSAGES
    result = []
    truncated = False
    for index, item in enumerate(islice(iter(errors), 5)):
        if index == 4:
            truncated = True
            break
        kind = item.get('type')
        kind = kind if type(kind) is str and kind in VALIDATION_MESSAGES else 'value_error'
        loc = []
        for depth, part in enumerate(islice(iter(item.get('loc', ('body',))), 9)):
            if depth == 8:
                truncated = True
                break
            if type(part) is int:
                loc.append(part if 0 <= part <= 1000000 else '<index>')
            else:
                loc.append(part if type(part) is str and part in FIELDS else '<unknown>')
        result.append({'type':kind, 'loc':loc})
    return result, truncated


def validation_failure(errors):
    try:
        record = active.get()
        if record is None: return
        issues, truncated = validation_fields(errors)
        record.update(stage='parse_validation', code='schema',
                      validation_errors=issues, validation_errors_truncated=truncated)
    except Exception:
        pass  # Diagnostics must never alter the original validation response.


def safe_failure(exc):
    from .policy import classify
    record = active.get() or {}
    code = classify(exc).code
    phase = record.get('stage','compute_process')
    if code == 'provider' and phase == 'compute_process': phase = 'provider'
    return {'stage':phase, 'code':code if code in CODES else 'internal'}

class FailureSink:
    def __init__(self, directory, *, maximum=1048576, backups=3):
        self.directory = Path(directory)
        self.maximum, self.backups = maximum, backups
        self.items = queue.Queue(maxsize=128)
        self.thread = threading.Thread(target=self._run, daemon=True, name='failure-diagnostics')
        self.thread.start()
    def emit(self, record):
        try:
            # No arbitrary exception strings, IDs, headers or application fields.
            keys = {'correlation','operation','status_attempted','response_started','response_completed','delivery','stage','code','elapsed_ms','queue_ms'}
            value = {k:v for k,v in record.items() if k in keys}
            if 'validation_errors' in record:
                issues, truncated = validation_fields(record['validation_errors'])
                value['validation_errors'] = issues
                value['validation_errors_truncated'] = truncated or record.get('validation_errors_truncated') is True
            raw = (json.dumps(value, separators=(',',':'), allow_nan=False)+'\n').encode()
            if len(raw) <= 2048: self.items.put_nowait(raw)
        except Exception: pass
    def _run(self):
        while True:
            raw = self.items.get()
            try: self._write(raw)
            except Exception: pass
            finally: self.items.task_done()
    def _write(self, raw):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink(): return
        os.chmod(self.directory,0o700)
        path = self.directory/'failures.jsonl'
        if path.exists() and path.stat().st_size + len(raw) > self.maximum:
            for index in range(self.backups,0,-1):
                source = path if index==1 else Path(str(path)+'.'+str(index-1))
                target = Path(str(path)+'.'+str(index))
                if source.exists(): os.replace(source,target)
        fd = os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            os.fchmod(fd,0o600)
            os.write(fd,raw)
        finally: os.close(fd)
