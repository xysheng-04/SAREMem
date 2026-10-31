"""Search-only operational timeouts; frozen provider algorithms stay intact."""
import time

import httpx
from openai import OpenAI


def network_timeout(policy, deadline, maximum=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError()
    read = min(policy.provider_timeout_seconds, remaining,
               maximum if maximum is not None else policy.provider_timeout_seconds)
    short = min(policy.connect_timeout_seconds, read)
    return httpx.Timeout(read, connect=short, write=short, pool=short)


def clip_request_timeout(request, policy, deadline):
    bounds = network_timeout(policy, deadline).as_dict()
    requested = request.extensions.get('timeout', {})
    request.extensions['timeout'] = {
        name: min(bound, requested[name]) if requested.get(name) is not None else bound
        for name, bound in bounds.items()}


class DeadlineTransport(httpx.BaseTransport):
    def __init__(self, policy, deadline, delegate=None):
        self.policy = policy
        self.deadline = deadline
        self.delegate = delegate if delegate is not None else httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        from .diagnostics import stage
        stage("provider")
        # Recompute on every SDK send, including frozen SDK retries. Never let
        # a new send inherit a timeout larger than its stage's remaining time.
        clip_request_timeout(request, self.policy, self.deadline)
        import sys, math
        module = sys.modules.get('aml_api.search_images')
        images = module.ACTIVE.get() if module is not None else None
        started = time.monotonic()
        record = {'kind':'llm' if request.url.path.endswith('/chat/completions') else 'embedding'}
        try:
            response = self.delegate.handle_request(request)
            record['http_status'] = response.status_code
            try:
                delay = float(response.headers.get('retry-after', 'nan'))
                if math.isfinite(delay) and delay >= 0:
                    record['retry_after_seconds'] = delay
            except (ValueError, TypeError):
                pass
            return response
        except Exception as exc:
            name = type(exc).__name__
            record['error_type'] = name if name in {'ConnectError','ReadError','WriteError','ReadTimeout','WriteTimeout','ConnectTimeout','PoolTimeout','RemoteProtocolError'} else 'TransportError'
            raise
        finally:
            record['elapsed_ms'] = round((time.monotonic()-started)*1000,3)
            if images is not None:
                images.sends.append(record)

    def close(self):
        self.delegate.close()


def bounded_search_client(client, policy, deadline):
    # Alternate injected provider clients own their transport; the supervised
    # process deadline still applies. The production RouteA clients are OpenAI.
    if not isinstance(client, OpenAI):
        return client
    # Preserve the frozen client's actual pool, proxy/TLS/redirect settings,
    # existing hooks and retry count. The SDK is version-pinned; only use its
    # underlying HTTPX client's public request-hook interface, without replacing it.
    client.timeout = network_timeout(policy, deadline)
    def clip(request):
        from .diagnostics import stage
        stage("provider")
        clip_request_timeout(request, policy, deadline)
        import sys
        module = sys.modules.get('aml_api.search_images')
        images = module.ACTIVE.get() if module is not None else None
        if images is not None:
            images.sends.append({'kind':'legacy', 'sent_at_monotonic':time.monotonic()})
    client._client.event_hooks.setdefault('request', []).append(clip)
    return client
