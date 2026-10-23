
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1)
    source_language: str = Field(default="auto", min_length=2, max_length=10)
    target_language: Literal["en"] = "en"

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty or whitespace.")
        return value

    @field_validator("source_language")
    @classmethod
    def normalize_source_language(cls, value: str) -> str:
        return value.strip().lower()


class TranslationResponse(BaseModel):
    success: bool
    original_text: str
    translated_text: str
    source_language: str | None = None
    target_language: str = "en"
