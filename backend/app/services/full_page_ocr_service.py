from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.services.model_adapter import CharacterClassifier


class FullPageOCRUnavailable(RuntimeError):
    """Raised when CNN-based full-page OCR cannot run."""


class FullPageOCRService:
    """CNN-based OCR with Shirorekha-aware character segmentation.

    Pipeline:
      1. Grayscale → adaptive threshold → binary mask (foreground = ink)
      2. Remove notebook rules via morphological horizontal filtering
      3. Detect individual text-line bands using peak-valley analysis on horizontal projection
      4. Per line: extract word-level connected components
      5. Per word CC: detect Shirorekha (headline) and split into characters via vertical projection
      6. Invert & aspect-ratio center/pad each character crop into a 32x32 canvas matching DHCD training data
      7. Classify each character with the trained CNN
      8. Reconstruct words, spaces, and lines
    """

    def __init__(self, classifier: CharacterClassifier) -> None:
        self.classifier = classifier

    @property
    def is_loaded(self) -> bool:
        return self.classifier.is_loaded

    def load_model(self) -> None:
        if not self.classifier.is_loaded:
            raise FullPageOCRUnavailable(
                "The trained CNN must be loaded before full-page OCR."
            )

    # ------------------------------------------------------------------
    # Notebook rule removal
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_notebook_rules(
        binary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect long horizontal notebook rules with minimal text damage."""
        height, width = binary.shape

        # Use a wide kernel to isolate continuous horizontal structures.
        kernel_width = max(80, int(width * 0.25))
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, 1),
        )

        # Morphological opening: only structures wider than kernel survive.
        rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # Extend horizontally to bridge small gaps within a rule.
        rules = cv2.dilate(
            rules,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)),
            iterations=1,
        )

        # Thicken vertically slightly to cover rule boundaries.
        rules = cv2.dilate(
            rules,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
            iterations=1,
        )

        # Subtract detected rule pixels from the binary mask.
        cleaned = cv2.subtract(binary, rules)

        # Ignore the notebook's vertical red-line margin on the left.
        margin = int(width * 0.10)
        cleaned[:, :margin] = 0

        return cleaned, rules

    # ------------------------------------------------------------------
    # Line detection via peak-valley horizontal projection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_line_bands(
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Find text-line bands using peak-valley analysis on horizontal projection.

        In Devanagari script, the continuous Shirorekha (headline) creates a
        distinct peak in horizontal ink projection for each text line. The
        valleys between consecutive peaks identify true line boundaries.
        """
        height, width = mask.shape
        projection = np.count_nonzero(mask, axis=1).astype(np.float32)

        if np.count_nonzero(projection) < 10:
            return []

        # Smooth projection to smooth out intra-line stroke fluctuations
        window = max(7, int(height * 0.012))
        if window % 2 == 0:
            window += 1
        smooth = cv2.GaussianBlur(
            projection.reshape(-1, 1), (1, window), 0,
        ).ravel()

        # Find headline peaks
        positive = smooth[smooth > 0]
        if len(positive) == 0:
            return []

        min_peak_height = max(15.0, float(np.percentile(positive, 50)) * 0.5)
        min_distance = max(18, int(height * 0.022))

        peaks: list[int] = []
        radius = max(8, min_distance // 2)
        for i in range(radius, len(smooth) - radius):
            if smooth[i] >= min_peak_height and smooth[i] == max(smooth[i - radius:i + radius + 1]):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)

        if len(peaks) >= 2:
            bands: list[tuple[int, int]] = []
            for idx in range(len(peaks)):
                peak = peaks[idx]
                if idx == 0:
                    top = peak
                    cutoff = max(5.0, smooth[peak] * 0.15)
                    while top > 0 and smooth[top] > cutoff:
                        top -= 1
                else:
                    prev_peak = peaks[idx - 1]
                    valley_idx = prev_peak + int(np.argmin(smooth[prev_peak:peak]))
                    top = valley_idx

                if idx == len(peaks) - 1:
                    bottom = peak
                    cutoff = max(5.0, smooth[peak] * 0.15)
                    while bottom < len(smooth) - 1 and smooth[bottom] > cutoff:
                        bottom += 1
                else:
                    next_peak = peaks[idx + 1]
                    valley_idx = peak + int(np.argmin(smooth[peak:next_peak]))
                    bottom = valley_idx

                if bottom - top >= max(10, int(height * 0.01)):
                    bands.append((top, bottom))
            return bands

        # Fallback if peak detection didn't find multiple lines:
        threshold = max(5.0, float(np.percentile(positive, 30)) * 0.3)
        active = smooth > threshold
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            return []

        bands = []
        start = previous = int(indices[0])
        gap_limit = max(4, int(height * 0.008))
        for idx in indices[1:]:
            idx = int(idx)
            if idx - previous > gap_limit:
                bands.append((start, previous + 1))
                start = idx
            previous = idx
        bands.append((start, previous + 1))

        min_height = max(8, int(height * 0.012))
        return [(top, bottom) for top, bottom in bands if bottom - top >= min_height]

    # ------------------------------------------------------------------
    # Word-level connected components per line
    # ------------------------------------------------------------------

    @staticmethod
    def _line_word_components(
        line_mask: np.ndarray,
        offset_y: int,
        image_width: int,
    ) -> list[dict[str, int]]:
        """Extract word-level connected components from one line band.

        Returns bounding boxes sorted left-to-right.
        """
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            line_mask, connectivity=8,
        )

        band_height = line_mask.shape[0]
        min_area = max(15, int(band_height * 1.2))
        components: list[dict[str, int]] = []

        for label_id in range(1, count):
            x, y, w, h, area = stats[label_id].tolist()

            # Skip tiny noise dots
            if area < min_area or w < 5 or h < 8:
                continue

            # Reject components that span nearly the full width (rule remnants)
            if w > image_width * 0.45:
                continue

            components.append({
                "x": int(x),
                "y": int(y + offset_y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
            })

        return sorted(components, key=lambda c: c["x"])

    # ------------------------------------------------------------------
    # Shirorekha-based character splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_word_into_characters(
        word_binary: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Split a word-level binary crop into individual character column spans.

        Uses vertical projection of pixels below the Shirorekha (headline).
        """
        h, w = word_binary.shape
        if w < 24 or h < 15:
            return [(0, w)]

        top_region = word_binary[: max(1, int(h * 0.35)), :]
        row_counts = np.count_nonzero(top_region, axis=1)
        max_row = row_counts.max() if len(row_counts) > 0 else 0

        if max_row < w * 0.30:
            # No clear horizontal headline detected
            return [(0, w)]

        headline_rows = np.flatnonzero(row_counts >= max_row * 0.50)
        headline_bottom = int(headline_rows[-1]) + 1 if len(headline_rows) > 0 else int(h * 0.25)

        body = word_binary[headline_bottom:, :]
        if body.shape[0] < 3:
            return [(0, w)]

        v_proj = np.count_nonzero(body, axis=0).astype(np.float32)
        v_smooth = cv2.GaussianBlur(v_proj.reshape(1, -1), (5, 1), 0).ravel()

        v_min = float(v_smooth.min())
        v_max = float(v_smooth.max())
        gap_threshold = max(1.0, v_min + (v_max - v_min) * 0.22)

        is_gap = v_smooth < gap_threshold
        cuts = [0]
        in_gap = False
        gap_start = 0

        for col in range(w):
            if is_gap[col] and not in_gap:
                in_gap = True
                gap_start = col
            elif not is_gap[col] and in_gap:
                in_gap = False
                cut = (gap_start + col) // 2
                if cut - cuts[-1] >= 10:  # minimum plausible character width
                    cuts.append(cut)

        if w - cuts[-1] >= 8:
            cuts.append(w)
        else:
            cuts[-1] = w

        segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        return segments if segments else [(0, w)]

    # ------------------------------------------------------------------
    # Classify a single character crop
    # ------------------------------------------------------------------

    def _classify_crop(
        self,
        gray: np.ndarray,
        x: int, y: int, w: int, h: int,
        img_h: int, img_w: int,
    ) -> tuple[str | None, float]:
        """Crop character region, invert to black-bg/white-text, center in 32x32, and classify."""
        pad_x = max(1, int(w * 0.08))
        pad_y = max(1, int(h * 0.08))

        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(img_w, x + w + pad_x)
        bottom = min(img_h, y + h + pad_y)

        patch = gray[top:bottom, left:right]
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
            return None, 0.0

        # Invert if light background so background is dark and strokes are bright (matching DHCD)
        if patch.mean() > 127:
            inv = 255 - patch
        else:
            inv = patch.copy()

        # Find tight bounding box of foreground ink strokes
        _, thresh = cv2.threshold(inv, 35, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            char_roi = inv[by:by + bh, bx:bx + bw]
        else:
            char_roi = inv

        rw, rh = char_roi.shape[1], char_roi.shape[0]
        if rw < 2 or rh < 2:
            return None, 0.0

        # Scale character to fit within 26x26 box (preserving aspect ratio and leaving ~3px margin in 32x32)
        scale = 26.0 / max(rw, rh)
        new_w = max(1, min(30, int(round(rw * scale))))
        new_h = max(1, min(30, int(round(rh * scale))))
        resized = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center in 32x32 black image
        padded_32 = np.zeros((32, 32), dtype=np.uint8)
        off_x = (32 - new_w) // 2
        off_y = (32 - new_h) // 2
        padded_32[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        crop = Image.fromarray(padded_32)

        try:
            prediction = self.classifier.predict_character(crop, top_k=1)
        except Exception:
            return None, 0.0

        character = prediction.get("predicted_character")
        confidence = float(prediction.get("confidence", 0.0))
        return character, confidence

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------

    def extract(self, image: Image.Image) -> dict[str, Any]:
        if not self.classifier.is_loaded:
            raise FullPageOCRUnavailable("The CNN model is not loaded.")

        image = ImageOps.exif_transpose(image).convert("RGB")

        if image.width * image.height > 25_000_000:
            raise FullPageOCRUnavailable("Image is too large.")

        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 11,
        )

        segmentation_mask, _ = self._remove_notebook_rules(binary)
        line_bands = self._find_line_bands(segmentation_mask)

        reconstructed_lines: list[str] = []
        confidences: list[float] = []
        uncertain_count = 0
        region_count = 0

        for top, bottom in line_bands:
            line_mask = segmentation_mask[top:bottom, :]
            line_gray = gray[top:bottom, :]

            word_components = self._line_word_components(
                line_mask, top, img_w,
            )

            if not word_components:
                continue

            line_characters: list[str] = []
            previous_right: int | None = None

            for comp in word_components:
                cx = comp["x"]
                cy = comp["y"] - top  # line-relative y
                cw = comp["w"]
                ch = comp["h"]

                # Detect inter-word gaps for space insertion
                if previous_right is not None:
                    gap = cx - previous_right
                    if gap > max(8, int(ch * 0.35)):
                        line_characters.append(" ")

                word_bin = line_mask[cy:cy + ch, cx:cx + cw]

                char_segments = self._split_word_into_characters(word_bin)

                for col_start, col_end in char_segments:
                    seg_w = col_end - col_start
                    if seg_w < 4:
                        continue

                    # Find vertical extent in this column
                    seg_slice = word_bin[:, col_start:col_end]
                    fg_rows = np.any(seg_slice > 0, axis=1)
                    fg_indices = np.flatnonzero(fg_rows)

                    if len(fg_indices) == 0:
                        continue

                    seg_y_start = int(fg_indices[0])
                    seg_y_end = int(fg_indices[-1]) + 1
                    seg_h = seg_y_end - seg_y_start

                    if seg_h < 4:
                        continue

                    abs_x = cx + col_start
                    abs_y = top + cy + seg_y_start

                    character, confidence = self._classify_crop(
                        gray, abs_x, abs_y, seg_w, seg_h, img_h, img_w,
                    )

                    if confidence < 0.25 or not character:
                        character = ""
                        uncertain_count += 1

                    line_characters.append(str(character))
                    confidences.append(confidence)
                    region_count += 1

                previous_right = cx + cw

            reconstructed = "".join(line_characters).strip()
            if reconstructed:
                reconstructed_lines.append(reconstructed)

        warnings = [
            "Experimental full-page OCR using a 46-class isolated-character CNN.",
            "The CNN covers base consonants, 3 conjuncts (क्ष, त्र, ज्ञ), and "
            "digits. Vowels, matras, halant, anusvara, visarga, and punctuation "
            "are not in the training set.",
            "Character segmentation uses vertical-projection splitting at the "
            "Shirorekha. Complex conjuncts and overlapping strokes may not "
            "segment correctly.",
        ]

        if not line_bands:
            warnings.append(
                "No text-line bands were detected. Check the input image."
            )

        if uncertain_count:
            warnings.append(
                f"{uncertain_count} region(s) had low confidence or "
                "an unmapped class and were marked with ."
            )

        return {
            "text": "\n".join(reconstructed_lines).strip(),
            "average_confidence": (
                float(np.mean(confidences)) if confidences else 0.0
            ),
            "detected_regions": region_count,
            "warnings": warnings,
        }
