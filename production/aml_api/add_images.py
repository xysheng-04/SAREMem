"""Caption-only visual derivatives; original Add payload and persistence stay intact."""
from contextvars import ContextVar
import hashlib
import json
import multiprocessing
import os
import threading
import time

from .errors import BadRequest, ServiceError, UpstreamUnavailable

AUDIT = ContextVar('add_image_audit', default=None)
DEADLINE = ContextVar('add_image_deadline', default=None)
SLOT = threading.BoundedSemaphore(1)
VERSION = 'add-png-v1'


def worker(conn, images, deadline, mode, edge, pixels):
    from .production.process import quiet
    quiet()
    try:
        if os.name == 'posix' and __import__('sys').platform.startswith('linux'):
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        from .search_images import Images, ImageInputError
        policy = Images(deadline, mode, edge, pixels)
        policy.policy['version'] = VERSION
        policy.binding = hashlib.sha256(json.dumps(policy.policy, sort_keys=True).encode()).hexdigest()
        derived = [policy.image(uri) for uri in images]
        result = {'images': derived, 'policy': policy.policy, 'binding': policy.binding,
                  'derivatives': policy.records}
    except BaseException as exc:
        from .search_images import ImageInputError
        result = {'error': exc.code if isinstance(exc, ImageInputError) else
                  'deadline' if isinstance(exc, TimeoutError) else 'image_processing'}
    try:
        conn.send_bytes(json.dumps(result, separators=(',', ':')).encode())
    except BaseException:
        pass
    finally:
        conn.close()


def prepare(images):
    from .production.diagnostics import stage
    stage("image_processing")
    if not images:
        return images
    mode = os.environ.get('AML_ADD_IMAGE_MODE', 'auto')
    if mode == 'off':
        return images
    try:
        edge = int(os.environ.get('AML_ADD_IMAGE_MAX_EDGE', '2048'))
        pixels = int(os.environ.get('AML_ADD_IMAGE_MAX_PIXELS', '2097152'))
        if mode != 'auto' or not 64 <= edge <= 2048 or not 4096 <= pixels <= 4194304:
            raise ValueError()
    except (TypeError, ValueError):
        raise ServiceError('invalid Add image policy') from None
    deadline = min(time.monotonic()+30, DEADLINE.get() or float('inf'))
    if not SLOT.acquire(timeout=max(0, deadline-time.monotonic())):
        raise UpstreamUnavailable('Add image processing busy')
    parent, child = multiprocessing.get_context('spawn').Pipe(duplex=False)
    proc = multiprocessing.get_context('spawn').Process(
        target=worker, args=(child, images, deadline, mode, edge, pixels), name='aml-add-images')
    started = False
    try:
        proc.start(); started = True; child.close()
        if not parent.poll(max(0, deadline-time.monotonic())):
            raise UpstreamUnavailable('Add image processing deadline exceeded')
        try:
            result = json.loads(parent.recv_bytes(64000000))
        except (EOFError, OSError, ValueError):
            raise UpstreamUnavailable('Add image processing unavailable') from None
        if time.monotonic() >= deadline or result.get('error') == 'deadline':
            raise UpstreamUnavailable('Add image processing deadline exceeded')
        if 'error' in result:
            if result['error'] == 'image_policy':
                raise ServiceError('invalid Add image policy')
            raise BadRequest('invalid or unsupported Add image')
        audit = AUDIT.get()
        if audit is not None:
            audit.append({k: v for k, v in result.items() if k != 'images'})
        return result['images']
    finally:
        parent.close(); child.close()
        if started:
            if proc.is_alive(): proc.terminate()
            proc.join(.5)
            if proc.is_alive(): proc.kill(); proc.join(.5)
            proc.close()
        SLOT.release()
