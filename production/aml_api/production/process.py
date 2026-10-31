"""Spawned stage supervision. No detached threads; socket reads obey the deadline."""
import json
import logging
import multiprocessing
import os
import select
import socket
import time
import sys

from .policy import SearchFault, classify


def quiet():
    # Startup applies this before provider imports; spawned workers apply it again.
    logging.disable(logging.CRITICAL)
    os.environ.pop('OPENAI_LOG', None)


def execute(stage, settings, request, policy, deadline, checkpoint, publish, stats):
    if stage in {'simple', 'legacy'} and os.environ.get('AML_SEARCH_IMAGE_MODE', 'auto') != 'off':
        from ..search_images import scope
        with scope(deadline) as images:
            try:
                return _execute(stage, settings, request, policy, deadline, checkpoint, publish, stats)
            finally:
                stats['image_policy'] = images.policy
                stats['image_policy_binding'] = images.binding
                stats['image_derivatives'] = images.records
                stats['provider_sends'] = images.sends
                stats['provider_usage'] = images.usage
    return _execute(stage, settings, request, policy, deadline, checkpoint, publish, stats)

def _execute(stage, settings, request, policy, deadline, checkpoint, publish, stats):
    if stage == 'sufficiency':
        from ..context_sufficiency import execute as sufficiency
        return sufficiency(settings,request,policy,deadline,checkpoint,stats)
    from .runtime import simple, legacy
    from .validation import seed_response
    if stage == 'simple':
        return simple(settings, request, policy, deadline, publish, stats)
    if stage == 'legacy':
        return legacy(settings, request, policy, deadline, publish, stats, checkpoint=checkpoint)
    if stage == 'seed':
        stats['phase'] = 'seed_reuse' if checkpoint is not None else 'seed_missing'
        return seed_response(settings, request, policy, checkpoint, stats)
    raise SearchFault('internal')


def worker(sock, stage, settings, request, policy, deadline, checkpoint, backend):
    quiet()
    from .diagnostics import active, safe_failure
    active.set({"stage":stage})
    stats = {'phase': stage, 'candidate_count': None, 'dropped': 0,
             'media_filtered': False, 'seed_existing': False}
    def send(value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',',':')).encode()+b'\n'
        if len(data) > policy.max_wire_bytes:
            raise SearchFault('capacity')
        sock.sendall(data)
    try:
        if sys.platform.startswith('linux'):
            import resource
            ceiling = policy.worker_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        from .validation import finalize
        result = backend(stage, settings, request, policy, deadline, checkpoint,
                         lambda seed: send({'seed': seed}), stats)
        stats['phase'] = 'final_validation'
        result = finalize(settings, request, policy, result, stage, stats)
        send({'response': result, 'stats': stats})
    except BaseException as exc:
        fault = classify(exc)
        try:
            send({'error': fault.code, 'terminal': fault.terminal, 'stats': stats, 'diagnostic':safe_failure(exc)})
        except BaseException:
            # The parent may have closed the socket at its deadline. It observes EOF/timeout;
            # never let multiprocessing print an unsanitized child traceback on this path.
            pass
    finally:
        sock.close()


class StageRunner:
    def __init__(self, backend=execute):
        # Injection is Python-only for offline tests, never configuration or HTTP input.
        self.backend = backend

    def run(self, stage, settings, request, policy, deadline, checkpoint, publish):
        from .request_clock import request_cancelled
        cancelled = request_cancelled.get()
        if cancelled is not None and cancelled.is_set():
            raise SearchFault('deadline')
        parent, child = socket.socketpair()
        proc = multiprocessing.get_context('spawn').Process(
            target=worker, args=(child, stage, settings, request, policy, deadline, checkpoint, self.backend),
            daemon=True, name='aml-search-'+stage)
        buffer = bytearray()
        started = False
        try:
            proc.start()
            started = True
            child.close()
            parent.setblocking(False)
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise SearchFault('deadline')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise SearchFault('deadline')
                readable,_,_ = select.select([parent], [], [], min(remaining,.05))
                if not readable:
                    continue
                block = parent.recv(65536)
                if not block:
                    raise SearchFault('worker_exit')
                buffer.extend(block)
                if len(buffer) > policy.max_wire_bytes:
                    raise SearchFault('capacity')
                while b'\n' in buffer:
                    raw,_,tail = buffer.partition(b'\n')
                    buffer = bytearray(tail)
                    message = json.loads(raw)
                    if 'seed' in message:
                        publish(message['seed'])
                    else:
                        return message
        finally:
            parent.close()
            child.close()
            if started:
                # Stages own no subprocesses. Reap even after timeout, response or serialization failure.
                if proc.is_alive():
                    proc.terminate()
                proc.join(policy.cleanup_seconds/2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(policy.cleanup_seconds/2)
                if proc.is_alive():
                    raise SearchFault('cleanup_failed', terminal=True)
                proc.close()
