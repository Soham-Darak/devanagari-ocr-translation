
from io import BytesIO
from pathlib import Path

import fitz
import httpx
import pytest

from app.core.config import Settings
from app.services.model_adapter import CharacterClassifier
from app.services.pdf_service import (
    PDFProcessingError,
    extract_pdf_text,
)
from app.services.translation_service import (
    TranslationError,
    TranslationService,
)


def create_pdf(pages: list[str]) -> bytes:
    """Create a small in-memory PDF for testing."""
    document = fitz.open()

    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)

    content = document.tobytes()
    document.close()
    return content


def test_pdf_extracts_embedded_text():
    content = create_pdf(["Hello from Lekha", "Second page"])

    result = extract_pdf_text(content, max_pages=5)

    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[0].extraction_method == "embedded_text"
    assert "Hello from Lekha" in result.pages[0].text
    assert "Second page" in result.pages[1].text
    assert "Hello from Lekha" in result.combined_text


def test_pdf_marks_scanned_page_as_unavailable():
    content = create_pdf([""])

    result = extract_pdf_text(content, max_pages=5)

    assert len(result.pages) == 1
    assert result.pages[0].extraction_method == "unavailable"
    assert result.pages[0].text == ""
    assert result.pages[0].warning is not None
    assert result.warnings


def test_pdf_rejects_invalid_content():
    with pytest.raises(PDFProcessingError, match="valid PDF"):
        extract_pdf_text(b"not a PDF")


def test_pdf_enforces_page_limit():
    content = create_pdf(["First", "Second"])

    with pytest.raises(PDFProcessingError, match="limit"):
        extract_pdf_text(content, max_pages=1)


def test_missing_checkpoint_does_not_load_model(tmp_path: Path):
    classifier = CharacterClassifier(
        checkpoint_path=tmp_path / "missing.pth",
        requested_device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        classifier.load()

    assert classifier.is_loaded is False


class MockGoogleResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict,
    ) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class MockAsyncClient:
    response_status = 200
    response_payload = {
        "data": {
            "translations": [
                {
                    "translatedText": "Hello",
                    "detectedSourceLanguage": "hi",
                }
            ]
        }
    }
    requests: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.__class__.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, *, params, json):
        self.__class__.requests.append(
            {"url": url, "params": params, "json": json}
        )
        return MockGoogleResponse(
            self.response_status,
            self.response_payload,
        )


@pytest.mark.asyncio
async def test_translation_success(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    settings = Settings(
        _env_file=None,
        google_translate_api_key="test-key",
    )
    service = TranslationService(settings)

    result = await service.translate(
        text="नमस्ते",
        source_language="auto",
    )

    assert result["translated_text"] == "Hello"
    assert result["source_language"] == "hi"
    assert result["target_language"] == "en"

    request = MockAsyncClient.requests[0]
    assert request["params"]["key"] == "test-key"
    assert request["json"]["q"] == "नमस्ते"
    assert request["json"]["target"] == "en"
    assert "source" not in request["json"]


@pytest.mark.asyncio
async def test_translation_handles_invalid_credentials(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    monkeypatch.setattr(MockAsyncClient, "response_status", 403)

    settings = Settings(
        _env_file=None,
        google_translate_api_key="invalid-key",
    )
    service = TranslationService(settings)

    with pytest.raises(TranslationError, match="authentication failed"):
        await service.translate(
            text="नमस्ते",
            source_language="hi",
        )


@pytest.mark.asyncio
async def test_translation_rejects_missing_api_key():
    settings = Settings(
        _env_file=None,
        google_translate_api_key="",
    )
    service = TranslationService(settings)

    with pytest.raises(TranslationError, match="not configured"):
        await service.translate(
            text="नमस्ते",
            source_language="hi",
        )


def test_translation_chunking_preserves_all_input_characters():
    text = ("नमस्ते " * 1500) + "\n\n" + ("मराठी " * 1000)

    chunks = TranslationService._chunks(text, max_chars=4000)

    assert len(chunks) > 1
    assert all(len(chunk) <= 4000 for chunk in chunks)
    assert "".join(chunks) == text
