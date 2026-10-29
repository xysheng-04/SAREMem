"""Validation helpers for AML inline image Data URIs."""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass

from .errors import BadRequest
from .schemas import ContentPart, ImagePart, TextPart

MIB = 1024 * 1024
MAX_IMAGE_BYTES = 10 * MIB
MAX_ADD_IMAGE_BYTES = 30 * MIB
MAX_SEARCH_IMAGE_BYTES = 30 * MIB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ValidatedImage:
    data_uri: str
    mime_type: str
    decoded_bytes: int


def validate_image_data_uri(value: str) -> ValidatedImage:
    match = _DATA_URI_RE.match(value)
    if not match:
        raise BadRequest("image_url.url must be an inline Base64 Data URI")
    mime_type = match.group(1).lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise BadRequest("image type must be JPEG, PNG, or WebP")
    encoded = re.sub(r"\s+", "", match.group(2))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest("image_url.url contains invalid Base64 data") from exc
    if not raw:
        raise BadRequest("image_url.url decodes to an empty image")
    if len(raw) > MAX_IMAGE_BYTES:
        raise BadRequest("each decoded image must be at most 10 MiB")
    return ValidatedImage(value, mime_type, len(raw))


def validate_parts(parts: list[ContentPart], *, total_limit: int) -> int:
    total = 0
    for part in parts:
        if isinstance(part, ImagePart):
            total += validate_image_data_uri(part.image_url.url).decoded_bytes
            if total > total_limit:
                raise BadRequest(f"decoded images exceed the {total_limit // MIB} MiB request limit")
    return total


def plain_text(parts: list[ContentPart]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextPart)).strip()


def image_urls(parts: list[ContentPart]) -> list[str]:
    return [part.image_url.url for part in parts if isinstance(part, ImagePart)]


def response_image_bytes(parts: list[ContentPart]) -> int:
    return sum(
        validate_image_data_uri(part.image_url.url).decoded_bytes
        for part in parts
        if isinstance(part, ImagePart)
    )
