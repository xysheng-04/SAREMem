"""Production assessment only; no payload logging or experiment retry semantics."""
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import math
import time

import httpx
from openai import APIConnectionError, APIResponseValidationError, APIStatusError

from .policy import SearchFault
from ..postseed.contracts import Failure

TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
BACKOFF_SECONDS = 0.5
MIN_ATTEMPT_SECONDS = 1.0
MAX_ATTEMPT_SECONDS = 600.0


def retry_delay(exc):
    """Use a valid Retry-After, including HTTP-date; never retain other headers."""
    value = exc.response.headers.get('retry-after') if isinstance(exc, APIStatusError) else None
    if value is not None and len(value) <= 128:
        try:
            seconds = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                seconds = max(0.0, date.timestamp() - time.time()) if date.tzinfo else math.nan
            except (ValueError, TypeError, OverflowError):
                seconds = math.nan
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return BACKOFF_SECONDS


def transient(exc):
    if isinstance(exc, APIConnectionError) and isinstance(exc.__cause__,
            (httpx.UnsupportedProtocol, httpx.LocalProtocolError)):
        return False
    return (isinstance(exc, (APIConnectionError, httpx.TimeoutException,
                            httpx.NetworkError, httpx.RemoteProtocolError)) or
            isinstance(exc, APIStatusError) and exc.status_code in TRANSIENT_STATUS)


@dataclass
class AttemptBudget:
    # One instance per candidate, shared by original and protocol repair calls.
    sends: int = 0
    transport_retries: int = 0

    def send(self, controls, operation):
        while True:
            controls.check_time()
            remaining = controls.deadline - time.monotonic()
            if remaining < MIN_ATTEMPT_SECONDS:
                raise SearchFault('retry_budget')
            if self.sends >= 3:
                raise SearchFault('retry_limit')
            self.sends += 1
            try:
                response = operation(min(MAX_ATTEMPT_SECONDS, controls.provider_timeout_seconds, remaining))
            except (APIResponseValidationError, json.JSONDecodeError):
                # The HTTP response arrived but its envelope could not be decoded.
                # This is eligible for the same one protocol repair, not a network retry.
                controls.check_time()
                controls.data['calls'].append({'state':'complete', 'send_started':True})
                raise Failure('provider_json', 'assessment') from None
            except Exception as exc:
                controls.check_time()
                if not transient(exc):
                    raise SearchFault('provider_auth' if isinstance(exc,APIStatusError) and exc.status_code in {401,403} else 'provider') from None
                if self.transport_retries >= 1 or self.sends >= 3:
                    raise SearchFault('provider_retry_exhausted') from None
                delay = retry_delay(exc)
                if delay + MIN_ATTEMPT_SECONDS >= controls.deadline - time.monotonic():
                    raise SearchFault('retry_budget') from None
                self.transport_retries += 1
                time.sleep(delay)
                # The deadline is checked again after sleeping, never reset.
                continue
            controls.check_time()
            return response
