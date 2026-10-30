"""Public HTTP errors: actionable fixed messages, never input or exception text."""
from ..errors import ServiceError, UpstreamUnavailable


class CapacityExceeded(ServiceError):
    status_code = 429


class SearchDeadlineExceeded(UpstreamUnavailable):
    status_code = 504


class AddDeadlineExceeded(UpstreamUnavailable):
    status_code = 504


REASONS = {
    400: 'invalid memory request; check content format and media limits',
    401: 'invalid Memory System Key',
    403: 'memory scope validation failed',
    404: 'endpoint not found; check the Add/Search URL',
    405: 'HTTP method not allowed for this endpoint',
    408: 'request timed out; retry the same logical request with backoff',
    409: 'memory request conflict; retry Add with the same request_id and payload only',
    413: 'request exceeds the supported payload size',
    415: 'unsupported request content type',
    425: 'memory service not ready; retry with backoff',
    429: 'memory service capacity exceeded; retry with backoff',
    500: 'internal memory service error',
    502: 'upstream service returned an invalid response; retry with backoff',
    503: 'memory service unavailable; retry with backoff',
    504: 'memory request deadline exceeded; retry with backoff',
    524: 'gateway request timed out; retry with backoff',
}

# Only literals written by our code may be more specific than the status message.
SAFE_DETAILS = {
    'request_id was already used with a different payload',
    'this Add request is still processing',
    'Add request lost its processing claim',
    'image_url.url must be an inline Base64 Data URI',
    'image type must be JPEG, PNG, or WebP',
    'image_url.url contains invalid Base64 data',
    'image_url.url decodes to an empty image',
    'each decoded image must be at most 10 MiB',
    'decoded images exceed the 30 MiB request limit',
    'decoded images exceed the 30 MiB Add request limit',
}


def reason(status, detail=None):
    if status in {400, 409} and isinstance(detail, str) and detail in SAFE_DETAILS:
        return detail
    return REASONS.get(status, 'memory service request failed')


def retry_headers(exc):
    value = exc.retry_after
    # Never echo an arbitrary header or accidentally turn a permanent error into retry advice.
    if exc.status_code in {408, 425, 429, 502, 503, 504, 524}:
        if type(value) is int and 1 <= value <= 60:
            return {'Retry-After': str(value)}
    return None


FIELDS = frozenset({'body', 'query', 'path', 'header', 'request_id', 'user_id', 'session_id',
                    'messages', 'role', 'content', 'timestamp', 'options', 'top_k',
                    'type', 'text', 'image_url', 'url', 'authorization', 'X-Api-Key'})
VALIDATION_MESSAGES = {
    'missing': 'field required',
    'extra_forbidden': 'unknown field is not allowed',
    'json_invalid': 'invalid JSON body',
    'string_type': 'must be a string',
    'string_too_short': 'must not be empty',
    'list_type': 'must be an array',
    'model_attributes_type': 'must be an object',
    'model_type': 'must be an object',
    'int_type': 'must be an integer',
    'int_parsing': 'must be an integer',
    'int_from_float': 'must be an integer',
    'greater_than_equal': 'must be at least the allowed minimum',
    'less_than_equal': 'must not exceed the allowed maximum',
    'literal_error': 'unsupported role or content type',
    'union_tag_invalid': 'content type must be text or image_url',
    'union_tag_not_found': 'content type is required: text or image_url',
    'value_error': 'invalid value; check the field format and required nonempty arrays',
}


def validation_details(errors):
    result = []
    for error in errors:
        kind = error.get('type')
        kind = kind if kind in VALIDATION_MESSAGES else 'value_error'
        # Unknown property names and union tags can themselves contain secrets.
        loc = [part if type(part) is int or (isinstance(part, str) and part in FIELDS)
               else '<unknown>' for part in error.get('loc', ('body',))]
        message = VALIDATION_MESSAGES[kind]
        if loc == ['body', 'top_k'] and kind == 'greater_than_equal':
            message = 'top_k must be at least 1'
        elif loc == ['body', 'top_k'] and kind == 'less_than_equal':
            message = 'top_k must not exceed 100'
        elif kind == 'value_error' and loc in (['body', 'messages'], ['body', 'query']):
            message = 'must contain at least one item'
        result.append({'type': kind, 'loc': loc, 'msg': message})
    return result
