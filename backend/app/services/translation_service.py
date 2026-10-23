
import logging

import httpx

from app.core.config import Settings


logger = logging.getLogger("lekha.translation")

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
SUPPORTED_SOURCE_LANGUAGES = {"auto", "hi", "mr", "ne"}


class TranslationError(RuntimeError):
    pass


class TranslationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _chunks(text: str, max_chars: int = 4000) -> list[str]:
        """Split text without losing characters or paragraph separators."""
        chunks: list[str] = []
        current = ""

        for paragraph in text.splitlines(keepends=True):
            while len(paragraph) > max_chars:
                available = max_chars - len(current)
                if available > 0:
                    current += paragraph[:available]
                    paragraph = paragraph[available:]
                chunks.append(current)
                current = ""

            if len(current) + len(paragraph) > max_chars:
                if current:
                    chunks.append(current)
                current = paragraph
            else:
                current += paragraph

        if current:
            chunks.append(current)

        return chunks or [text]

    async def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "en",
    ) -> dict[str, str]:
        if not text.strip():
            raise TranslationError("Enter some text to translate.")

        if len(text) > self.settings.max_translation_characters:
            raise TranslationError(
                "The text exceeds the configured translation limit of "
                f"{self.settings.max_translation_characters} characters."
            )

        if target_language != "en":
            raise TranslationError("Only English translation is enabled.")

        source_language = source_language.strip().lower()
        if source_language not in SUPPORTED_SOURCE_LANGUAGES:
            raise TranslationError(
                "Choose auto-detection, Hindi, Marathi, or Nepali."
            )

        api_key = self.settings.google_translate_api_key.strip()
        if not api_key:
            raise TranslationError(
                "Google Cloud Translation is not configured. "
                "Set GOOGLE_TRANSLATE_API_KEY in backend/.env."
            )

        chunks = self._chunks(text)
        translated_chunks: list[str] = []
        detected_language: str | None = None

        timeout = httpx.Timeout(
            self.settings.google_translate_timeout_seconds
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            for chunk in chunks:
                payload: dict[str, str] = {
                    "q": chunk,
                    "target": target_language,
                    "format": "text",
                }
                if source_language != "auto":
                    payload["source"] = source_language

                try:
                    response = await client.post(
                        TRANSLATE_URL,
                        params={"key": api_key},
                        json=payload,
                    )
                except httpx.TimeoutException as exc:
                    raise TranslationError(
                        "Google Translation timed out. Please try again."
                    ) from exc
                except httpx.RequestError as exc:
                    raise TranslationError(
                        "Could not connect to Google Cloud Translation."
                    ) from exc

                if response.status_code in (401, 403):
                    logger.warning(
                        "Google Translation rejected the credentials "
                        "or project permissions (HTTP %s).",
                        response.status_code,
                    )
                    raise TranslationError(
                        "Google Translation authentication failed. "
                        "Check your API key, API enablement, and permissions."
                    )

                if response.status_code == 429:
                    raise TranslationError(
                        "Google Translation quota or rate limit exceeded. "
                        "Check your Cloud quota and billing settings."
                    )

                if response.status_code >= 400:
                    logger.warning(
                        "Google Translation returned HTTP %s.",
                        response.status_code,
                    )
                    raise TranslationError(
                        "Google Translation could not process this request. "
                        "Check the configured project and request limits."
                    )

                try:
                    data = response.json()["data"]["translations"][0]
                    translated = data["translatedText"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise TranslationError(
                        "Google Translation returned an unexpected response."
                    ) from exc

                translated_chunks.append(translated)

                detected = data.get("detectedSourceLanguage")
                if detected:
                    detected_language = detected

        return {
            "translated_text": "".join(translated_chunks),
            "source_language": (
                detected_language
                or (None if source_language == "auto" else source_language)
            ),
            "target_language": target_language,
        }
