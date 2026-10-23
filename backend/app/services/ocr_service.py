import logging
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.services.model_adapter import CharacterClassifier


logger = logging.getLogger("lekha.ocr")

# Supported EasyOCR languages:
# Hindi, Marathi, Nepali and English.
EASYOCR_LANGUAGES = ["hi", "mr", "ne", "en"]


class OCRCapabilityUnavailable(RuntimeError):
    """Raised when the requested OCR operation is unsupported."""


class OCRModelUnavailable(RuntimeError):
    """Raised when a required OCR model cannot be loaded."""


class OCRService:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
    ) -> None:
        # Retain the existing CNN for isolated-character classification.
        self.classifier = CharacterClassifier(
            checkpoint_path=checkpoint_path,
            requested_device=device,
        )

        self.load_error: str | None = None

        # Pretrained full-page OCR reader.
        self.reader: Any | None = None
        self.easyocr_load_error: str | None = None

        # Prevent concurrent EasyOCR inference calls.
        self._inference_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether the existing character classifier is loaded."""
        return self.classifier.is_loaded

    @property
    def is_full_page_ocr_loaded(self) -> bool:
        """Return whether the EasyOCR reader is available."""
        return self.reader is not None

    def load_model(self) -> None:
        """Load the existing isolated-character CNN."""
        try:
            self.classifier.load()
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            raise

    def load_easyocr(self) -> None:
        """
        Load the pretrained EasyOCR reader once.

        EasyOCR may download its model files during the first
        initialization. This implementation uses CPU inference.
        """
        if self.reader is not None:
            return

        try:
            import easyocr

            logger.info(
                "Loading EasyOCR for languages: %s",
                ", ".join(EASYOCR_LANGUAGES),
            )

            reader = easyocr.Reader(
                EASYOCR_LANGUAGES,
                gpu=False,
                verbose=False,
            )

            self.reader = reader
            self.easyocr_load_error = None

            logger.info(
                "EasyOCR loaded successfully on CPU."
            )

        except Exception as exc:
            self.easyocr_load_error = str(exc)

            logger.exception(
                "Could not load the EasyOCR model."
            )

            raise OCRModelUnavailable(
                "The full-page OCR model could not be loaded. "
                "Check the backend logs, network connection, "
                "and EasyOCR model download."
            ) from exc

    def recognize_character(
        self,
        image: Image.Image,
    ) -> dict[str, Any]:
        """
        Classify an isolated character using the existing CNN.
        """
        if not self.is_loaded:
            raise OCRModelUnavailable(
                "The trained character model is unavailable. "
                "Check the checkpoint path and backend logs."
            )

        try:
            return self.classifier.predict_character(image)

        except Exception as exc:
            raise OCRCapabilityUnavailable(
                "The character image could not be processed."
            ) from exc

    def extract_from_image(
        self,
        image: Image.Image,
    ) -> tuple[str, list[str]]:
        """
        Extract printed text from an entire image using EasyOCR.

        Returns:
            A tuple containing the extracted text and warnings.
        """
        if self.reader is None:
            raise OCRModelUnavailable(
                "The full-page OCR model is unavailable. "
                "Check the backend logs and try again."
            )

        self.validate_image(image)

        # EasyOCR expects an RGB image array.
        image_array = np.asarray(
            image.convert("RGB")
        )

        try:
            with self._inference_lock:
                detections = self.reader.readtext(
                    image_array,
                    detail=1,
                    paragraph=True,
                )

        except Exception as exc:
            logger.exception(
                "EasyOCR failed to process the uploaded image."
            )

            raise OCRCapabilityUnavailable(
                "The image could not be recognized. "
                "Try a clearer image with better lighting, "
                "higher contrast, and readable text."
            ) from exc

        recognized_parts: list[str] = []
        confidences: list[float] = []

        for detection in detections:
            if (
                not isinstance(detection, (list, tuple))
                or len(detection) < 2
            ):
                continue

            text = str(detection[1]).strip()

            if not text:
                continue

            recognized_parts.append(text)

            # Some paragraph-mode results may not include confidence.
            if len(detection) >= 3:
                try:
                    confidences.append(
                        float(detection[2])
                    )
                except (TypeError, ValueError):
                    pass

        extracted_text = "\n".join(
            recognized_parts
        ).strip()

        if not extracted_text:
            return "", [
                "No readable text was detected. Try a sharper "
                "image with better lighting, higher contrast, "
                "and larger text."
            ]

        warnings: list[str] = []

        if (
            confidences
            and sum(confidences) / len(confidences) < 0.50
        ):
            warnings.append(
                "OCR confidence is low. Review the extracted "
                "text and correct any recognition errors."
            )

        return extracted_text, warnings

    @staticmethod
    def validate_image(
        image: Image.Image,
    ) -> None:
        """Validate image dimensions before processing."""
        if image.width < 1 or image.height < 1:
            raise OCRCapabilityUnavailable(
                "The image has invalid dimensions."
            )

        if image.width * image.height > 25_000_000:
            raise OCRCapabilityUnavailable(
                "The image dimensions are too large to process safely."
            )

    @staticmethod
    def open_image(
        content: bytes,
    ) -> Image.Image:
        """Decode uploaded image bytes safely."""
        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                result = image.copy()

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise OCRCapabilityUnavailable(
                "The uploaded file is not a valid, readable image."
            ) from exc

        OCRService.validate_image(result)

        return result
