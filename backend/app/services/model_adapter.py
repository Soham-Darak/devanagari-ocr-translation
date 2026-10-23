
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


# This mapping is used only when a checkpoint class name matches a
# recognized DHCD-style name. Unknown labels remain explicitly unknown.
DEVANAGARI_BY_SUFFIX = {
    "ka": "क",
    "kha": "ख",
    "ga": "ग",
    "gha": "घ",
    "kna": "ङ",
    "cha": "च",
    "chha": "छ",
    "ja": "ज",
    "jha": "झ",
    "yna": "ञ",
    "taamatar": "ट",
    "thaa": "ठ",
    "daa": "ड",
    "dhaa": "ढ",
    "adna": "ण",
    "tabala": "त",
    "tha": "थ",
    "da": "द",
    "dha": "ध",
    "na": "न",
    "pa": "प",
    "pha": "फ",
    "ba": "ब",
    "bha": "भ",
    "ma": "म",
    "yaw": "य",
    "ra": "र",
    "la": "ल",
    "waw": "व",
    "motosaw": "श",
    "petchiryakha": "ष",
    "patalosaw": "स",
    "ha": "ह",
    "chhya": "क्ष",
    "tra": "त्र",
    "gya": "ज्ञ",
    **{f"digit_{i}": chr(0x0966 + i) for i in range(10)},
}


class DevanagariCNN(nn.Module):
    """CNN architecture copied from the supplied notebook."""

    def __init__(self, num_classes: int = 46) -> None:
        super().__init__()

        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.drop3(self.pool3(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        return self.fc2(x)


def devanagari_label_for(class_name: str) -> str | None:
    """Resolve only known folder-name conventions; never guess by index."""
    normalized = class_name.strip().lower()

    if normalized in DEVANAGARI_BY_SUFFIX:
        return DEVANAGARI_BY_SUFFIX[normalized]

    # Common DHCD names include prefixes such as character_01_ka.
    suffix = re.sub(r"^character_\d+_", "", normalized)
    suffix = re.sub(r"^\d+_", "", suffix)

    if suffix in DEVANAGARI_BY_SUFFIX:
        return DEVANAGARI_BY_SUFFIX[suffix]

    return None


def preprocess_character(image: Image.Image) -> torch.Tensor:
    """Match the CNN notebook's deterministic evaluation transforms.

    The CNN was trained on DHCD (white characters on black background).
    If the input image has a light background (paper scan with dark ink),
    invert it so the background is dark and strokes are bright.
    """
    image = image.convert("L")
    pixels = np.asarray(image, dtype=np.float32)

    # Invert light backgrounds (e.g., ink on white paper)
    if pixels.mean() > 127:
        pixels = 255.0 - pixels

    if image.size != (32, 32):
        inv_img = Image.fromarray(pixels.astype(np.uint8))
        inv_img = inv_img.resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(inv_img, dtype=np.float32)

    pixels = pixels / 255.0
    pixels = (pixels - 0.5) / 0.5

    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

class CharacterClassifier:
    """Loads the real trained checkpoint once and predicts one character."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        requested_device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = requested_device
        self.device = self._choose_device(requested_device)
        self.model: DevanagariCNN | None = None
        self.class_names: list[str] = []
        self.test_accuracy: float | None = None

    @staticmethod
    def _choose_device(requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable. "
                "Set MODEL_DEVICE=auto or MODEL_DEVICE=cpu."
            )

        if requested_device not in {"cpu", "cuda"}:
            raise ValueError("MODEL_DEVICE must be auto, cpu, or cuda.")

        return torch.device(requested_device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}. "
                "Supply devanagari_cnn_best.pth from the training notebook."
            )

        try:
            checkpoint: dict[str, Any] = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not read the checkpoint. Verify that it was saved "
                "by the supplied notebook and that the PyTorch version "
                "supports this checkpoint format."
            ) from exc

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Checkpoint is missing the model_state_dict entry."
            )

        raw_names = checkpoint.get("class_names")
        if not isinstance(raw_names, (list, tuple)):
            raise RuntimeError(
                "Checkpoint is missing its training class_names mapping."
            )

        class_names = [str(name) for name in raw_names]
        num_classes = int(checkpoint.get("num_classes", len(class_names)))

        if num_classes != 46 or len(class_names) != num_classes:
            raise RuntimeError(
                "Expected 46 classes and a matching class_names mapping; "
                f"received num_classes={num_classes}, "
                f"class_names={len(class_names)}."
            )

        model = DevanagariCNN(num_classes=num_classes).to(self.device)

        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint weights do not match the notebook CNN architecture."
            ) from exc

        model.eval()
        self.model = model
        self.class_names = class_names

        accuracy = checkpoint.get("test_accuracy")
        self.test_accuracy = (
            float(accuracy) if accuracy is not None else None
        )

    def predict_character(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("The character model has not been loaded.")

        if not 1 <= top_k <= min(5, len(self.class_names)):
            raise ValueError("top_k must be between 1 and 5.")

        tensor = preprocess_character(image).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            scores, indices = torch.topk(probabilities, k=top_k)

        predictions = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            class_name = self.class_names[index]
            predictions.append(
                {
                    "class_index": index,
                    "class_name": class_name,
                    "character": devanagari_label_for(class_name),
                    "confidence": float(score),
                }
            )

        return {
            "predicted_class": predictions[0]["class_name"],
            "predicted_character": predictions[0]["character"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions,
        }
