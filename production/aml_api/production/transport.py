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
        # Recompute on every SDK send, including frozen SDK retries. Never let
        # a new send inherit a timeout larger than its stage's remaining time.
        clip_request_timeout(request, self.policy, self.deadline)
        return self.delegate.handle_request(request)

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
        clip_request_timeout(request, policy, deadline)
    client._client.event_hooks.setdefault('request', []).append(clip)
    return client
