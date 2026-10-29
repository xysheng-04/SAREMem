"""Strict AML Add/Search request and response schemas."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

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


class AddMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: list[ContentPart]
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
    query: list[ContentPart]
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


class SearchResponse(BaseModel):
    data: list[SearchItem]
