
import logging
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.services.translation_service import TranslationError


logger = logging.getLogger("lekha.translation")

MODEL_LANGUAGE_CODES = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
}

TARGET_LANGUAGE_CODE = "eng_Latn"

# Keep chunks smaller than the model's maximum input length.
MAX_INPUT_TOKENS = 400


class LocalTranslationService:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_characters: int = 30000,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_characters = max_characters

        self.tokenizer = None
        self.model = None

        # Prevent concurrent generation from increasing memory usage
        # unexpectedly on a CPU-only machine.
        self._inference_lock = threading.Lock()

    def load_model(self) -> None:
        """Load the tokenizer and model once during application startup."""
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info("Loading local translation model: %s", self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        model.to(self.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

        logger.info(
            "Local translation model loaded successfully on %s.",
            self.device,
        )

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "en",
    ) -> dict[str, str]:
        """Translate text locally using NLLB-200."""
        if not text.strip():
            raise TranslationError("Enter some text to translate.")

        if len(text) > self.max_characters:
            raise TranslationError(
                "The text exceeds the configured translation limit of "
                f"{self.max_characters} characters."
            )

        if target_language.strip().lower() != "en":
            raise TranslationError("Only English translation is enabled.")

        source_language = source_language.strip().lower()

        if source_language == "auto":
            raise TranslationError(
                "Automatic language detection is not enabled for local "
                "translation. Please select Hindi, Marathi, or Nepali."
            )

        if source_language not in MODEL_LANGUAGE_CODES:
            raise TranslationError(
                "Choose Hindi, Marathi, or Nepali as the source language."
            )

        if self.tokenizer is None or self.model is None:
            raise TranslationError(
                "The local translation model is unavailable. "
                "Check the backend startup logs."
            )

        tokenizer = self.tokenizer
        model = self.model

        translated_chunks: list[str] = []

        try:
            with self._inference_lock:
                tokenizer.src_lang = MODEL_LANGUAGE_CODES[source_language]

                token_ids = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                if not token_ids:
                    raise TranslationError(
                        "No translatable text was found."
                    )

                english_token_id = tokenizer.convert_tokens_to_ids(
                    TARGET_LANGUAGE_CODE
                )

                if (
                    english_token_id is None
                    or english_token_id == tokenizer.unk_token_id
                ):
                    raise TranslationError(
                        "The translation tokenizer does not recognize "
                        "the English target-language code."
                    )

                for start in range(0, len(token_ids), MAX_INPUT_TOKENS):
                    chunk_ids = token_ids[
                        start : start + MAX_INPUT_TOKENS
                    ]

                    chunk_text = tokenizer.decode(
                        chunk_ids,
                        skip_special_tokens=True,
                    )

                    if not chunk_text.strip():
                        continue

                    inputs = tokenizer(
                        chunk_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)

                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            forced_bos_token_id=english_token_id,
                            max_new_tokens=256,
                            num_beams=2,
                        )

                    translated = tokenizer.batch_decode(
                        generated,
                        skip_special_tokens=True,
                    )[0].strip()

                    if translated:
                        translated_chunks.append(translated)

        except TranslationError:
            raise
        except Exception as exc:
            logger.exception("Local translation inference failed.")
            raise TranslationError(
                "Local translation failed. Check the backend logs "
                "and try again."
            ) from exc

        if not translated_chunks:
            raise TranslationError(
                "The model did not return a translation."
            )

        return {
            "translated_text": " ".join(translated_chunks),
            "source_language": source_language,
            "target_language": "en",
        }
