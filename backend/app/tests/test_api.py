
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_reports_api_status():
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Lekha API"
    assert body["ocr_model"]["status"] in {"loaded", "unavailable"}


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"


def test_translation_rejects_blank_text():
    with TestClient(app) as client:
        response = client.post(
            "/api/translate",
            json={
                "text": "   ",
                "source_language": "hi",
                "target_language": "en",
            },
        )

    assert response.status_code == 422


def test_translation_rejects_unsupported_target_language():
    with TestClient(app) as client:
        response = client.post(
            "/api/translate",
            json={
                "text": "नमस्ते",
                "source_language": "hi",
                "target_language": "fr",
            },
        )

    assert response.status_code == 422


def test_translation_does_not_require_google_credentials():
    class FakeLocalTranslationService:
        def translate(
            self,
            text,
            source_language="auto",
            target_language="en",
        ):
            return {
                "translated_text": "Hello",
                "source_language": source_language,
                "target_language": target_language,
            }

    with TestClient(app) as client:
        original_service = app.state.translation_service
        original_key = app.state.settings.google_translate_api_key

        try:
            app.state.settings.google_translate_api_key = ""
            app.state.translation_service = FakeLocalTranslationService()

            response = client.post(
                "/api/translate",
                json={
                    "text": "नमस्ते",
                    "source_language": "hi",
                    "target_language": "en",
                },
            )
        finally:
            app.state.translation_service = original_service
            app.state.settings.google_translate_api_key = original_key

    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["original_text"] == "नमस्ते"
    assert body["translated_text"] == "Hello"
    assert body["source_language"] == "hi"
    assert body["target_language"] == "en"
