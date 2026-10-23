
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


def test_translation_reports_missing_credentials():
    # This test is deterministic because the Google client is not called
    # when the API key is absent.
    with TestClient(app) as client:
        settings = app.state.settings
        original_key = settings.google_translate_api_key

        try:
            settings.google_translate_api_key = ""
            response = client.post(
                "/api/translate",
                json={
                    "text": "नमस्ते",
                    "source_language": "hi",
                    "target_language": "en",
                },
            )
        finally:
            settings.google_translate_api_key = original_key

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
