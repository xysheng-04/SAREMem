"""Frozen provider algorithms with observation at *each* SDK transport send.

Transport and observer are mandatory dependencies. No default network transport,
credentials lookup, production ledger, or legacy provider mutation is provided.
The frozen upstream exception retains its historical retries; new post-seed
providers must use the distinct single-attempt M3 boundary.
"""
from types import SimpleNamespace
from typing import Protocol
import httpx
from openai import OpenAI, DefaultHttpxClient, DEFAULT_MAX_RETRIES
from ..providers import VisionLanguageProvider, EmbeddingProvider
from ..postseed.seed_port import SeedError


class SendObserver(Protocol):
    def reserve(self,kind: str,index: int) -> object: ...
    def finish(self,token: object,outcome: str,status: int | None) -> None: ...


class _BoundaryAbort(BaseException):
    """Cross SDK's Exception catch without retrying an engineering failure.

    Converted immediately at the public provider boundary to a safe SeedError.
    This is intentionally not a network error or ordinary retryable Exception.
    """


class _ObservedTransport(httpx.BaseTransport):
    def __init__(self,delegate: httpx.BaseTransport,observer: SendObserver):
        self.delegate=delegate
        self.observer=observer
        self.index=0

    def handle_request(self,request):
        if request.url.path.endswith('/chat/completions'): kind='llm'
        elif request.url.path.endswith('/embeddings'): kind='embedding'
        else: raise _BoundaryAbort() from None
        self.index+=1
        try: token=self.observer.reserve(kind,self.index)
        except Exception: raise _BoundaryAbort() from None
        try: response=self.delegate.handle_request(request)
        except Exception:
            try: self.observer.finish(token,'transport_error',None)
            except Exception: raise _BoundaryAbort() from None
            raise
        try: self.observer.finish(token,'response',response.status_code)
        except Exception:
            try: response.close()
            except Exception: pass  # secondary close failure must not trigger SDK resend
            raise _BoundaryAbort() from None
        return response

    def close(self): self.delegate.close()


class FrozenProviders:
    _JSON_ATTEMPTS=VisionLanguageProvider._JSON_ATTEMPTS
    _MAX_JSON_TOKENS=VisionLanguageProvider._MAX_JSON_TOKENS
    _multimodal_content=staticmethod(VisionLanguageProvider._multimodal_content)

    def __init__(self,*,base_url: str,api_key: str,transport: httpx.BaseTransport,observer: SendObserver):
        # Pin the approved frozen SDK retry policy rather than inherit future defaults.
        if DEFAULT_MAX_RETRIES!=2:
            raise SeedError('sdk_retry_identity')
        self.model='gpt-4o-mini'
        self.client=OpenAI(api_key=api_key,base_url=base_url,max_retries=2,
                           http_client=DefaultHttpxClient(transport=_ObservedTransport(transport,observer),
                                                          trust_env=False))

    def _json(self,**kwargs):
        if kwargs.get('model') not in (None,'gpt-4o-mini'):
            raise SeedError('model_identity')
        try: return VisionLanguageProvider._json(self,**kwargs)
        except _BoundaryAbort: raise SeedError('send_boundary') from None

    def encode(self,texts):
        try:
            return EmbeddingProvider.encode(SimpleNamespace(client=self.client,model='text-embedding-v4'),texts)
        except _BoundaryAbort: raise SeedError('send_boundary') from None

    def close(self): self.client.close()
