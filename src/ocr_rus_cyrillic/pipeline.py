"""Page OCR pipeline: mobile detector + Cyrillic recognizer + bounded consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rapidocr_onnxruntime.ch_ppocr_v3_det.text_detect import TextDetector

from .corrector import normalize_russian_text
from .ensemble import EnsembleRecognizer
from .recognizer import CyrillicRecognizer


@dataclass
class OCRResult:
    text: str
    confidence: float
    certain: bool
    passes: int
    boxes: list[list[list[float]]] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "certain": self.certain,
            "passes": self.passes,
            "boxes": self.boxes,
            "lines": self.lines,
        }


class CyrillicOCR:
    """Offline page OCR for Russian/Cyrillic text."""

    def __init__(
        self,
        *,
        detector_path: str | Path,
        recognizer_path: str | Path,
        dictionary_path: str | Path,
        secondary_recognizer_path: str | Path | None = None,
        secondary_dictionary_path: str | Path | None = None,
        target_confidence: float = 0.90,
        max_passes: int = 4,
    ) -> None:
        self.target_confidence = target_confidence
        self.max_passes = max_passes
        self.detector_input_size = 736
        self.detector = TextDetector(
            {
                "model_path": str(detector_path),
                "use_cuda": False,
                "limit_side_len": 1280,
                "limit_type": "max",
                "thresh": 0.25,
                "box_thresh": 0.45,
                "max_candidates": 1000,
                "unclip_ratio": 1.6,
                "use_dilation": True,
                "score_mode": "fast",
            }
        )
        primary = CyrillicRecognizer(recognizer_path, dictionary_path)
        if secondary_recognizer_path is not None and secondary_dictionary_path is not None:
            secondary = CyrillicRecognizer(secondary_recognizer_path, secondary_dictionary_path)
            self.recognizer = EnsembleRecognizer(primary, secondary)
        else:
            self.recognizer = primary

    @staticmethod
    def _sorted_boxes(boxes: np.ndarray) -> list[np.ndarray]:
        if boxes is None or len(boxes) == 0:
            return []
        items = sorted(boxes, key=lambda b: (float(b[:, 1].min()), float(b[:, 0].min())))
        return items

    @staticmethod
    def _crop_quad(image: np.ndarray, points: np.ndarray, pad: int = 4) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        width = max(width, 4)
        height = max(height, 4)
        target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(points, target)
        crop = cv2.warpPerspective(
            image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )
        if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
            crop = np.rot90(crop)
        return cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    @staticmethod
    def _split_horizontal_words(crop: np.ndarray) -> list[np.ndarray]:
        """Split a detector box when it contains several words.

        PP-OCR detectors can merge a complete short line into one polygon. A
        vertical ink projection recovers large whitespace gaps without trying
        to segment normal inter-character gaps. If no clearly large gap exists,
        the original crop is kept intact.
        """
        if crop is None or crop.size == 0 or crop.shape[1] < 32:
            return [crop]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink = (binary > 0).sum(axis=0)
        active = np.where(ink > 0)[0]
        if active.size < 2:
            return [crop]
        runs: list[tuple[int, int]] = []
        start = prev = int(active[0])
        for x in active[1:]:
            x = int(x)
            if x > prev + 1:
                runs.append((start, prev))
                start = x
            prev = x
        runs.append((start, prev))
        if len(runs) < 2:
            return [crop]
        gaps = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
        # A word gap is usually several times larger than a glyph gap. Keep a
        # minimum so tiny fonts are not split at every antialiased stroke.
        positive_gaps = [gap for gap in gaps if gap > 0]
        median_gap = float(np.median(positive_gaps)) if positive_gaps else 1.0
        threshold = max(5, int(round(median_gap * 1.7)))
        split_after = {i for i, gap in enumerate(gaps) if gap >= threshold}
        if not split_after:
            return [crop]
        groups: list[tuple[int, int]] = []
        group_start = runs[0][0]
        for i, (_, right) in enumerate(runs[:-1]):
            if i in split_after:
                groups.append((group_start, right))
                group_start = runs[i + 1][0]
        groups.append((group_start, runs[-1][1]))
        if len(groups) <= 1:
            return [crop]
        result = []
        for left, right in groups:
            left = max(0, left - 4)
            right = min(crop.shape[1], right + 5)
            result.append(crop[:, left:right])
        return result

    @staticmethod
    def _find_white_bubbles(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find large white outlined speech bubbles as a safe page fallback."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = ((gray > 245) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape[:2]
        found: list[tuple[int, int, int, int, float]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            fill = area / max(1.0, float(w * h))
            if (
                area >= 15000
                and w >= 100
                and h >= 60
                and x > 10
                and y > 10
                and x + w < width - 10
                and y + h < height - 10
                and 1.15 <= w / max(h, 1) <= 5.5
                and fill >= 0.60
            ):
                found.append((x, y, w, h, area))
        found.sort(key=lambda item: (item[1], item[0]))
        return [(x, y, w, h) for x, y, w, h, _ in found]

    def _recognize_bubble_page(
        self,
        source: np.ndarray,
        bubbles: list[tuple[int, int, int, int]],
    ) -> OCRResult:
        results: list[dict[str, Any]] = []
        for x, y, w, h in bubbles:
            crop = source[y:y + h, x:x + w]
            # Re-run the normal detector inside the bubble, which removes most
            # of the page art and gives the recognizer a much cleaner crop.
            region = self.recognize_page(crop, bubble_mode=False)
            results.append({
                "box": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                "text": region.text,
                "confidence": region.confidence,
                "certain": region.certain,
                "passes": region.passes,
                "engine": "bubble-fallback",
            })
        text = "\n".join(r["text"] for r in results if r["text"])
        return OCRResult(
            text=normalize_russian_text(text, allow_dictionary=True),
            confidence=round(min((float(r["confidence"]) for r in results), default=0.0), 4),
            certain=bool(results) and all(bool(r["certain"]) for r in results),
            passes=max((int(r["passes"]) for r in results), default=0),
            boxes=[r["box"] for r in results],
            lines=results,
        )

    def recognize_page(
        self,
        image: np.ndarray | str | Path,
        *,
        bubble_mode: bool = True,
    ) -> OCRResult:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            raise ValueError("image could not be loaded")
        if bubble_mode:
            bubbles = self._find_white_bubbles(image)
            # Require several candidates to avoid replacing normal document OCR
            # with a false positive from a single white region.
            if len(bubbles) >= 3:
                bubble_result = self._recognize_bubble_page(image, bubbles)
                normal_result = self.recognize_page(image, bubble_mode=False)
                if normal_result.certain != bubble_result.certain:
                    return normal_result if normal_result.certain else bubble_result
                return bubble_result if bubble_result.confidence > normal_result.confidence else normal_result

        # The bundled mobile detector is exported with a fixed 736x736 input.
        # Letterbox instead of stretching, then map polygons back to the source.
        source = image
        src_h, src_w = source.shape[:2]
        scale = min(self.detector_input_size / max(src_w, 1), self.detector_input_size / max(src_h, 1))
        resized = cv2.resize(source, (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))))
        detector_image = np.full(
            (self.detector_input_size, self.detector_input_size, 3), 255, dtype=np.uint8
        )
        offset_x = (self.detector_input_size - resized.shape[1]) // 2
        offset_y = (self.detector_input_size - resized.shape[0]) // 2
        detector_image[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized

        boxes, _ = self.detector(detector_image)
        source_boxes: list[np.ndarray] = []
        for box in self._sorted_boxes(boxes):
            mapped = (np.asarray(box, dtype=np.float32) - np.array([offset_x, offset_y], dtype=np.float32)) / scale
            mapped[:, 0] = np.clip(mapped[:, 0], 0, src_w - 1)
            mapped[:, 1] = np.clip(mapped[:, 1], 0, src_h - 1)
            source_boxes.append(mapped)
        line_results: list[dict[str, Any]] = []
        for box in source_boxes:
            crop = self._crop_quad(source, box)
            for piece in self._split_horizontal_words(crop):
                result = self.recognizer.recognize_consensus(
                    piece,
                    target_confidence=self.target_confidence,
                    max_passes=self.max_passes,
                )
                # Ignore empty low-score detector noise, but retain non-empty
                # low-confidence text so the caller can request manual review.
                if not result["text"].strip() and float(result["confidence"]) < 0.5:
                    continue
                line_results.append({
                    "box": box.tolist(),
                    **result,
                })

        # Detector boxes are often word-level boxes. Group close boxes by their
        # vertical centre and join them in reading order.
        rows: list[list[dict[str, Any]]] = []
        for item in line_results:
            b = np.asarray(item["box"], dtype=np.float32)
            cy = float(b[:, 1].mean())
            h = float(np.linalg.norm(b[0] - b[3]))
            placed = False
            for row in rows:
                row_cy = float(np.mean([r["_cy"] for r in row]))
                row_h = float(np.mean([r["_h"] for r in row]))
                if abs(cy - row_cy) <= max(h, row_h) * 0.60:
                    row.append({**item, "_cy": cy, "_h": h})
                    placed = True
                    break
            if not placed:
                rows.append([{**item, "_cy": cy, "_h": h}])
        rows.sort(key=lambda row: min(r["_cy"] for r in row))
        output_lines: list[str] = []
        flat_lines: list[dict[str, Any]] = []
        for row in rows:
            row.sort(key=lambda r: min(p[0] for p in r["box"]))
            line_text = " ".join(r["text"] for r in row if r["text"])
            line_text = normalize_russian_text(line_text, allow_dictionary=True)
            line_conf = min((float(r["confidence"]) for r in row), default=0.0)
            line_certain = all(bool(r["certain"]) for r in row)
            output_lines.append(line_text)
            flat_lines.extend(row)
            for r in row:
                r.pop("_cy", None); r.pop("_h", None)
        text = "\n".join(line for line in output_lines if line)
        confidence = min((float(r["confidence"]) for r in flat_lines), default=0.0)
        certain = bool(flat_lines) and all(bool(r["certain"]) for r in flat_lines)
        passes = max((int(r["passes"]) for r in flat_lines), default=0)
        return OCRResult(
            text=text,
            confidence=round(confidence, 4),
            certain=certain,
            passes=passes,
            boxes=[r["box"] for r in flat_lines],
            lines=flat_lines,
        )
