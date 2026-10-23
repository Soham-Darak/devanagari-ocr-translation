
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.services.full_page_ocr_service import (
    FullPageOCRService,
    FullPageOCRUnavailable,
)
from app.services.model_adapter import CharacterClassifier


class OCRCapabilityUnavailable(RuntimeError):
    """Raised when the requested OCR operation is unsupported."""


class OCRModelUnavailable(RuntimeError):
    """Raised when a required OCR model cannot be loaded."""


class OCRService:
    """Coordinates the trained CNN and CNN-based full-page OCR pipeline."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
    ) -> None:
        self.classifier = CharacterClassifier(
            checkpoint_path=checkpoint_path,
            requested_device=device,
        )

        # Full-page OCR uses the same trained CNN for its predictions.
        self.full_page_ocr = FullPageOCRService(
            classifier=self.classifier
        )

        self.load_error: str | None = None
        self.full_page_load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the trained character CNN is loaded."""
        return self.classifier.is_loaded

    @property
    def is_full_page_loaded(self) -> bool:
        """Whether the CNN-based full-page pipeline is ready."""
        return self.full_page_ocr.is_loaded

    def load_model(self) -> None:
        """Load the trained CNN and initialize full-page OCR."""
        try:
            self.classifier.load()
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            raise

        try:
            self.full_page_ocr.load_model()
            self.full_page_load_error = None
        except Exception as exc:
            # Keep isolated-character classification available even if
            # full-page OCR initialization fails.
            self.full_page_load_error = str(exc)

    def recognize_character(
        self,
        image: Image.Image,
    ) -> dict[str, Any]:
        """Classify a single character image using the trained CNN."""
        if not self.is_loaded:
            raise OCRModelUnavailable(
                "The trained character CNN is unavailable. "
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
        engine: str = "auto",
    ) -> tuple[str, list[str]]:
        """Recognize a full handwritten page.

        Parameters:
          image: PIL Image to recognize.
          engine: 'auto' (uses full Devanagari recognition including vowels/matras
                  when available, falling back to custom CNN) or 'cnn' (uses the
                  custom 46-class Devanagari CNN character-splitting pipeline).
        """
        if engine in {"auto", "full"}:
            try:
                import easyocr
                import numpy as np

                if not hasattr(self, "_easyocr_reader") or self._easyocr_reader is None:
                    self._easyocr_reader = easyocr.Reader(["hi", "mr", "en"], gpu=False)

                rgb_arr = np.asarray(image.convert("RGB"))
                lines = self._easyocr_reader.readtext(rgb_arr, detail=0)
                text = "\n".join(lines).strip()
                if text:
                    warnings = [
                        "Complete Devanagari transcription generated (including vowels and matras).",
                        "Ready for English translation.",
                    ]
                    return text, warnings
            except Exception:
                pass  # Fall back to custom CNN pipeline

        if not self.is_loaded:
            raise OCRModelUnavailable(
                "The trained character CNN is unavailable."
            )

        if not self.is_full_page_loaded:
            detail = self.full_page_load_error or (
                "The full-page OCR pipeline is not initialized."
            )
            raise OCRModelUnavailable(
                "CNN-based full-page OCR is unavailable. "
                f"Initialization detail: {detail}"
            )

        try:
            result = self.full_page_ocr.extract(image)
        except FullPageOCRUnavailable as exc:
            raise OCRCapabilityUnavailable(
                "CNN-based full-page recognition failed."
            ) from exc

        text = result.get("text", "")
        warnings = list(result.get("warnings", []))

        if not text:
            warnings.append(
                "No readable text was reconstructed from the image."
            )

        return text, warnings


    @staticmethod
    def validate_image(image: Image.Image) -> None:
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
    def open_image(content: bytes) -> Image.Image:
        """Decode uploaded image bytes safely."""
        from io import BytesIO

        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                result = image.copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OCRCapabilityUnavailable(
                "The uploaded file is not a valid, readable image."
            ) from exc

        OCRService.validate_image(result)
        return result
