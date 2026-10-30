"""Bounded Add computation without database access; fenced atomic parent commit."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import select
import signal
import socket
import sqlite3
import sys
import threading
import tempfile
import time
from urllib.parse import quote

from ..errors import BadRequest, Conflict, ServiceError, UpstreamUnavailable
from ..media import MAX_ADD_IMAGE_BYTES, validate_parts
from ..schemas import AddResponse
from ..service import RouteAService, _flatten_messages, _stable_id
from ..store import SQLiteStore, payload_digest, canonical_json
from .http_errors import CapacityExceeded, AddDeadlineExceeded
from .request_clock import request_started, request_deadline, request_cancelled


def check(deadline, cancelled=None):
    if time.monotonic() >= deadline or (cancelled is not None and cancelled.is_set()):
        raise AddDeadlineExceeded('Add deadline exceeded')


class CleanupUnavailable(UpstreamUnavailable):
    pass


class PreparedStore:
    """The existing Add algorithm can compute, but cannot claim or write a database."""
    def claim_add(self, *args): return None
    def abandon_add(self, *args): pass
    def complete_add(self, memory, views, response):
        self.value = {'memory': {k:v for k,v in memory.items() if k != 'content'},
                      'views': [{**r, 'embedding': r['embedding'].tolist()} for r in views],
                      'response': response}


def compute(settings, request, policy, deadline):
    from ..add_images import DEADLINE
    from .transport import bounded_search_client
    token = DEADLINE.set(deadline)
    store = PreparedStore()
    service = RouteAService(settings, store=store)
    try:
        service.vlm.client = bounded_search_client(service.vlm.client, policy, deadline)
        service.embedder.client = bounded_search_client(service.embedder.client, policy, deadline)
        service.add(request)
        return {**store.value, 'input_digest': payload_digest(request.model_dump(mode='json'))}
    finally:
        DEADLINE.reset(token)
        service.vlm.client.close(); service.embedder.client.close()


def worker(sock, settings, request, policy, deadline, backend, owner_pid=None):
    from .process import quiet
    quiet()
    done = threading.Event()
    owner_pid = owner_pid or os.getppid()
    def supervise_parent():
        # This watchdog lives only in the computation process. Parent death or
        # loss of its socket stops this entire group, including image children.
        try:
            while not done.wait(.05):
                readable,_,_ = select.select([sock],[],[],0)
                if os.getppid() != owner_pid or time.monotonic() >= deadline or (readable and not sock.recv(1,socket.MSG_PEEK)):
                    if os.name == 'posix': os.killpg(os.getpgrp(),signal.SIGKILL)
                    os._exit(1)
        except (OSError,ValueError): pass
    try:
        if os.name == 'posix': os.setsid()
        if __import__('sys').platform.startswith('linux'):
            import resource
            limit = policy.worker_memory_mb * 1024**2
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        sock.sendall(b'{"ready":true}\n')
        threading.Thread(target=supervise_parent,daemon=True,name='add-parent-watch').start()
        result = backend(settings, request, policy, deadline)
        data = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()+b'\n'
        if len(data) > policy.max_wire_bytes: raise UpstreamUnavailable('Add computation capacity exceeded')
        sock.sendall(data)
    except BaseException as exc:
        status = exc.status_code if isinstance(exc, ServiceError) else 503 if isinstance(exc, (MemoryError, TimeoutError)) else 500
        try: sock.sendall(json.dumps({'error':status}).encode()+b'\n')
        except BaseException: pass
    finally:
        done.set()
        sock.close()


class ComputeRunner:
    def __init__(self, backend=compute): self.backend = backend

    def run(self, settings, request, policy, deadline, cancelled=None):
        parent, child = socket.socketpair()
        proc = multiprocessing.get_context('spawn').Process(target=worker,
            args=(child, settings, request, policy, deadline, self.backend, os.getpid()), name='aml-add-compute')
        started = False; group = False; buffer = bytearray()
        try:
            proc.start(); started = True; child.close(); parent.setblocking(False)
            while True:
                check(deadline, cancelled)
                readable,_,_ = select.select([parent], [], [], min(.05, max(0, deadline-time.monotonic())))
                if not readable: continue
                check(deadline,cancelled)
                block = parent.recv(65536)
                if not block: raise UpstreamUnavailable('Add computation unavailable')
                buffer.extend(block)
                if len(buffer) > policy.max_wire_bytes: raise UpstreamUnavailable('Add computation capacity exceeded')
                while b'\n' in buffer:
                    raw,_,tail = buffer.partition(b'\n'); buffer = bytearray(tail)
                    message = json.loads(raw)
                    if message.get('ready') is True: group = True; continue
                    check(deadline, cancelled)
                    if 'error' in message:
                        status = message['error']
                        if status == 400: raise BadRequest('invalid Add image or request')
                        if status == 504: raise AddDeadlineExceeded('Add deadline exceeded')
                        if status == 503: raise UpstreamUnavailable('Add provider unavailable')
                        raise ServiceError('internal Add computation error')
                    return message
        finally:
            parent.close(); child.close()
            if started:
                def stop(sig):
                    try:
                        if sys.platform.startswith('linux') and (group or os.getpgid(proc.pid) == proc.pid): os.killpg(proc.pid, sig)
                        elif proc.is_alive(): os.kill(proc.pid, sig)
                    except ProcessLookupError: pass
                stop(signal.SIGTERM); proc.join(policy.cleanup_seconds/2)
                stop(signal.SIGKILL); proc.join(policy.cleanup_seconds/2)
                if proc.is_alive(): raise CleanupUnavailable('Add computation cleanup unavailable')
                proc.close()


class OwnedStore(SQLiteStore):
    """No schema migration. updated_at is compared as the acquired claim token."""
    def __init__(self, settings, deadline, cancelled):
        self.path = settings.storage_path
        self.processing_timeout_seconds = settings.add_processing_timeout_seconds
        self.deadline, self.cancelled, self.owner = deadline, cancelled, None

    def marker(self, request_id):
        key = hashlib.sha256((str(self.path.resolve())+'\x1f'+request_id).encode()).hexdigest()
        return self.path.parent/('.add-lease-'+key+'.json')

    def read_marker(self, request_id, digest):
        path = self.marker(request_id)
        if not path.exists(): return []
        if path.stat().st_size > 65536: raise UpstreamUnavailable('Add execution state unavailable')
        try:
            value = json.loads(path.read_text())
            if value['digest'] != digest: raise ValueError()
            owners = value['owners']
            if not isinstance(owners,list) or len(owners)>1000 or any(type(v) not in {int,float} or not math.isfinite(v) for v in owners): raise ValueError()
            return owners
        except (ValueError,KeyError,TypeError):
            raise UpstreamUnavailable('Add execution state unavailable') from None

    def write_marker(self, request_id, digest, owners):
        if len(owners)>1000: raise UpstreamUnavailable('Add execution state unavailable')
        path = self.marker(request_id)
        fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=self.path.parent)
        try:
            with os.fdopen(fd,'w') as f:
                json.dump({'digest':digest,'owners':owners},f,allow_nan=False)
                f.flush(); os.fsync(f.fileno())
            os.replace(temporary,path)
            directory = os.open(self.path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    def clear_marker(self, request_id):
        try: self.marker(request_id).unlink()
        except OSError: pass  # A stale completed marker cannot authorize another computation.

    @contextmanager
    def _connection(self):
        check(self.deadline, self.cancelled)
        remaining = self.deadline-time.monotonic()
        db = sqlite3.connect('file:'+quote(self.path.as_posix(),safe='/:')+'?mode=rw',
                             uri=True,timeout=min(5., remaining))
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.set_progress_handler(lambda: int(time.monotonic() >= self.deadline or
            (self.cancelled is not None and self.cancelled.is_set())), 1000)
        try: yield db
        finally: db.close()

    def cached(self, request_id, digest):
        with self._connection() as db:
            row = db.execute('SELECT payload_hash,state,response_json FROM add_requests WHERE request_id=?', (request_id,)).fetchone()
            if row is not None:
                if row['payload_hash'] != digest: raise Conflict('request_id was already used with a different payload')
                if row['state'] == 'complete': return json.loads(row['response_json'])

    def claim(self, request_id, digest):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM add_requests WHERE request_id=?', (request_id,)).fetchone()
            now = time.time()
            owners = self.read_marker(request_id,digest)
            if row is not None:
                if row['payload_hash'] != digest: raise Conflict('request_id was already used with a different payload')
                if row['state'] == 'complete': return json.loads(row['response_json'])
                # The caller holds the kernel lock. A matching durable marker
                # proves that a previous supervised parent is no longer active.
                recoverable = float(row['updated_at']) in owners
                if not recoverable and now-float(row['updated_at']) <= self.processing_timeout_seconds:
                    raise Conflict('this Add request is still processing')
            self.owner = math.nextafter(max(now, float(row['updated_at']) if row else now), math.inf)
            if row is None:
                db.execute("INSERT INTO add_requests VALUES (?,?,'processing',NULL,?)", (request_id,digest,self.owner))
            else:
                db.execute('UPDATE add_requests SET updated_at=? WHERE request_id=?', (self.owner,request_id))
            # Preserve previous owners until COMMIT: a crash on either side of
            # the transaction remains recoverable without a schema migration.
            self.write_marker(request_id,digest,owners+[self.owner])
            check(self.deadline, self.cancelled); db.commit()

    def complete_owned(self, memory, views, response, digest):
        import numpy as np
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT 1 FROM add_requests WHERE request_id=? AND payload_hash=? AND state='processing' AND updated_at=?",
                (memory['request_id'],digest,self.owner)).fetchone()
            if row is None: raise Conflict('Add request lost its processing claim')
            db.execute('INSERT INTO memories (memory_id,request_id,user_id,session_id,source_timestamp_ms,content_json,search_text,created_at) VALUES (?,?,?,?,?,?,?,?)',
                (memory['memory_id'],memory['request_id'],memory['user_id'],memory['session_id'],memory.get('source_timestamp_ms'),canonical_json(memory['content']),memory['search_text'],memory['created_at']))
            for row in views:
                vector = np.asarray(row['embedding'],dtype=np.float32)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all(): raise ValueError('invalid prepared vector')
                db.execute('INSERT INTO views VALUES (?,?,?,?,?,?)', (row['view_id'],memory['memory_id'],row['kind'],row['text'],sqlite3.Binary(vector.tobytes()),int(vector.shape[0])))
            check(self.deadline,self.cancelled)
            changed = db.execute("UPDATE add_requests SET state='complete',response_json=?,updated_at=? WHERE request_id=? AND payload_hash=? AND state='processing' AND updated_at=?",
                (canonical_json(response),time.time(),memory['request_id'],digest,self.owner)).rowcount
            if changed != 1: raise Conflict('Add request lost its processing claim')
            check(self.deadline,self.cancelled); db.commit()
        self.clear_marker(memory['request_id'])

    def abandon_owned(self, request_id, digest):
        if self.owner is None: return
        # Only after all computation descendants are reaped; never delete a newer claim.
        self.deadline, self.cancelled = time.monotonic()+2, None
        with self._connection() as db:
            removed = db.execute("DELETE FROM add_requests WHERE request_id=? AND payload_hash=? AND state='processing' AND updated_at=?", (request_id,digest,self.owner)).rowcount
            db.commit()
            remaining = db.execute('SELECT 1 FROM add_requests WHERE request_id=?',(request_id,)).fetchone()
        if removed or remaining is None: self.clear_marker(request_id)


def validate_prepared(value, request, digest):
    if value['input_digest'] != digest: raise ValueError('Add input binding')
    response = AddResponse.model_validate(value['response'])
    if (response.request_id,response.user_id,response.session_id) != (request.request_id,request.user_id,request.session_id): raise ValueError('Add scope binding')
    memory = value['memory']
    for k in ('request_id','user_id','session_id'):
        if memory[k] != getattr(request,k): raise ValueError('Add scope binding')
    if memory['memory_id'] != _stable_id('mem',request.user_id,request.request_id): raise ValueError('Add identity binding')
    memory['content'] = _flatten_messages(request)[0]
    views = value['views']
    for i,row in enumerate(views):
        if row['view_id'] != _stable_id('view',memory['memory_id'],str(i),row['kind'],row['text']): raise ValueError('Add view binding')
    return memory, views, response


def add(settings, request, policy, runner=None):
    entered = request_started.get() or time.monotonic()
    deadline = request_deadline.get() or entered+policy.total_seconds
    cancelled = request_cancelled.get()
    finish = deadline-policy.cleanup_seconds
    store = OwnedStore(settings, finish, cancelled)
    digest = payload_digest(request.model_dump(mode='json'))
    check(finish,cancelled)
    total = sum(validate_parts(m.content,total_limit=MAX_ADD_IMAGE_BYTES) for m in request.messages)
    if total > MAX_ADD_IMAGE_BYTES: raise BadRequest('decoded images exceed the 30 MiB Add request limit')
    cached = store.cached(request.request_id,digest)
    if cached is not None: return AddResponse.model_validate(cached)
    # Kernel ownership survives threads and excludes concurrent writes across service processes.
    name = '.add-'+hashlib.sha256(str(settings.storage_path.resolve()).encode()).hexdigest()[:16]+'.lock'
    fd = os.open(settings.storage_path.parent/name, os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0), 0o600)
    locked = False
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); locked = True
        except BlockingIOError:
            with store._connection() as db:
                row = db.execute('SELECT payload_hash,state FROM add_requests WHERE request_id=?',(request.request_id,)).fetchone()
            if row is not None and row['payload_hash'] == digest and row['state'] == 'processing': raise Conflict('this Add request is still processing')
            raise CapacityExceeded('memory service busy',retry_after=1)
        cached = store.claim(request.request_id,digest)
        if cached is not None: return AddResponse.model_validate(cached)
        compute_end = finish-min(30.,policy.total_seconds*.15)
        check(compute_end,cancelled)
        prepared = (runner or ComputeRunner()).run(settings,request,policy,compute_end,cancelled)
        memory,views,response = validate_prepared(prepared,request,digest)
        check(finish,cancelled)
        store.complete_owned(memory,views,response.model_dump(mode='json'),digest)
        return response
    except BaseException as exc:
        # A lost response after COMMIT must not turn into another paid computation.
        try:
            store.deadline,store.cancelled = time.monotonic()+2,None
            cached = store.cached(request.request_id,digest)
            if cached is not None: return AddResponse.model_validate(cached)
            if not isinstance(exc,CleanupUnavailable):
                store.abandon_owned(request.request_id,digest)
        except (sqlite3.Error,ServiceError): pass  # Keep ambiguous state for a later lookup.
        if isinstance(exc,(sqlite3.Error,OSError)): raise UpstreamUnavailable('Add persistence temporarily unavailable') from None
        raise
    finally:
        if locked: fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)
