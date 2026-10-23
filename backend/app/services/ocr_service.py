
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.services.model_adapter import CharacterClassifier


class OCRCapabilityUnavailable(RuntimeError):
    """Raised when the requested OCR operation is unsupported."""


class OCRModelUnavailable(RuntimeError):
    """Raised when the trained model cannot be loaded."""


class OCRService:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
    ) -> None:
        self.classifier = CharacterClassifier(
            checkpoint_path=checkpoint_path,
            requested_device=device,
        )
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.classifier.is_loaded

    def load_model(self) -> None:
        try:
            self.classifier.load()
            self.load_error = None
        except Exception as exc:
            # Preserve a safe diagnostic for server logs. The API health
            # endpoint exposes only the availability state.
            self.load_error = str(exc)
            raise

    def recognize_character(self, image: Image.Image) -> dict[str, Any]:
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
        Recognize one isolated character.

        This method does not claim to perform word segmentation,
        handwriting line recognition, or document OCR.
        """
        if not self.is_loaded:
            raise OCRModelUnavailable(
                "The trained character model is unavailable."
            )

        result = self.recognize_character(image)
        character = result.get("predicted_character")

        if not character:
            class_name = result.get("predicted_class", "unknown")
            raise OCRCapabilityUnavailable(
                "The model predicted a class whose Devanagari mapping "
                f"could not be verified: {class_name}."
            )

        confidence = float(result["confidence"])
        warnings = [
            "The supplied CNN classifies isolated characters; it does "
            "not perform full-page OCR."
        ]

        if confidence < 0.60:
            warnings.append(
                "The model confidence is low. Verify the predicted "
                "character before using the result."
            )

        return character, warnings

    @staticmethod
    def validate_image(image: Image.Image) -> None:
        if image.width < 1 or image.height < 1:
            raise OCRCapabilityUnavailable("The image has invalid dimensions.")

        if image.width * image.height > 25_000_000:
            raise OCRCapabilityUnavailable(
                "The image dimensions are too large to process safely."
            )

    @staticmethod
    def open_image(content: bytes) -> Image.Image:
        try:
            from io import BytesIO

            with Image.open(BytesIO(content)) as image:
                image.load()
                result = image.copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OCRCapabilityUnavailable(
                "The uploaded file is not a valid, readable image."
            ) from exc

        OCRService.validate_image(result)
        return result
