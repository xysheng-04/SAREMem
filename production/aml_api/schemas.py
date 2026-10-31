"""Strict AML Add/Search request and response schemas."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, field_validator, model_serializer

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: NonEmpty


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: NonEmpty


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[Union[TextPart, ImagePart], Field(discriminator="type")]


def normalize_content_input(value):
    """Only string input gains a canonical text block; existing arrays are untouched."""
    if isinstance(value, str):
        return [{'type':'text', 'text':value}]
    return value  # The existing list/part validators reject objects, null and scalars.


ContentInput = Annotated[
    list[ContentPart],
    BeforeValidator(normalize_content_input, json_schema_input_type=Union[NonEmpty, list[ContentPart]]),
]


class AddMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: ContentInput
    timestamp: int | None = None

    @field_validator("content")
    @classmethod
    def non_empty_content(cls, value: list[ContentPart]) -> list[ContentPart]:
        if not value:
            raise ValueError("content must contain at least one part")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: NonEmpty
    messages: list[AddMessage]
    user_id: NonEmpty
    session_id: NonEmpty

    @field_validator("messages")
    @classmethod
    def non_empty_messages(cls, value: list[AddMessage]) -> list[AddMessage]:
        if not value:
            raise ValueError("messages must contain at least one message")
        return value


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: ContentInput
    options: list[str] | None = None
    user_id: NonEmpty
    top_k: int = Field(ge=1, le=100)

    @field_validator("query")
    @classmethod
    def non_empty_query(cls, value: list[ContentPart]) -> list[ContentPart]:
        if not value:
            raise ValueError("query must contain at least one part")
        return value


class SearchItem(BaseModel):
    id: str
    content: list[ContentPart]
    score: float | None = None
    created_at: str | None = None

    @model_serializer(mode="wrap")
    def omit_unknown_source_time(self, handler):
        value = handler(self)
        if self.created_at is None:
            value.pop("created_at", None)
        return value


class SearchResponse(BaseModel):
    data: list[SearchItem]
