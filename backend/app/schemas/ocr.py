
from typing import Literal

from pydantic import BaseModel, Field


class PageOCRResult(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    extraction_method: Literal[
        "embedded_text",
        "character_classifier",
        "easyocr",
        "unavailable",
    ]
    warning: str | None = None


class OCRResponse(BaseModel):
    success: bool
    filename: str
    input_type: Literal["image", "pdf"]
    page_count: int | None = None
    extracted_text: str
    pages: list[PageOCRResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
