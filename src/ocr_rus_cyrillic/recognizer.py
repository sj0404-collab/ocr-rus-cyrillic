"""ONNX Runtime recognizer for the Cyrillic PP-OCR mobile model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .corrector import normalize_russian_text


@dataclass(frozen=True)
class Candidate:
    text: str
    raw_text: str
    confidence: float
    variant: str


class CyrillicRecognizer:
    """Recognize a text-line crop using a Cyrillic-only output whitelist.

    The shipped model is an Apache-2.0 PP-OCRv3 mobile recognition model.
    Its original alphabet contains a few Latin characters because it supports
    several Cyrillic-script datasets. At decode time we mask those classes so
    this project exposes only Russian Cyrillic, digits and punctuation.
    """

    def __init__(
        self,
        model_path: str | Path,
        dict_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        input_height: int = 48,
        input_width: int = 320,
        russian_only: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.dict_path = str(dict_path)
        self.input_height = input_height
        self.input_width = input_width
        self.russian_only = russian_only
        self.characters = Path(dict_path).read_text(encoding="utf-8").splitlines()
        self.allowed = set(
            " АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "0123456789.,!?;:-()[]{}\"'«»„“”%№+/=…—–"
        )
        providers = list(providers or ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _trim(image: np.ndarray, margin: int = 4) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("empty OCR crop")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        mask = np.any(image < 245, axis=2)
        if mask.any():
            ys, xs = np.where(mask)
            y0 = max(0, int(ys.min()) - margin)
            y1 = min(image.shape[0], int(ys.max()) + margin + 1)
            x0 = max(0, int(xs.min()) - margin)
            x1 = min(image.shape[1], int(xs.max()) + margin + 1)
            image = image[y0:y1, x0:x1]
        return image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = self._trim(image)
        h, w = image.shape[:2]
        ratio = max(w / max(h, 1), 1e-3)
        resized_w = min(self.input_width, max(1, int(np.ceil(self.input_height * ratio))))
        resized = cv2.resize(image, (resized_w, self.input_height), interpolation=cv2.INTER_CUBIC)
        resized = resized.astype("float32").transpose(2, 0, 1) / 255.0
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((3, self.input_height, self.input_width), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def _variants(self, image: np.ndarray, *, extended: bool = False) -> Iterable[tuple[str, np.ndarray]]:
        base = self._trim(image, margin=4)
        yield "raw", base

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        # Keep three channels because the exported model has a 3-channel input.
        yield "gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        yield "otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

        if not extended:
            return

        # Extended agents are only used after the first pass is uncertain. They
        # target exactly the cases seen in manga/manhwa screenshots: white text
        # on dark panels, colored low-contrast lettering, and vertical text.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        yield "clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

        otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        yield "otsu_inv", cv2.cvtColor(otsu_inv, cv2.COLOR_GRAY2BGR)

        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        yield "adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

        h, w = base.shape[:2]
        if h > w * 1.20:
            yield "rot90", np.ascontiguousarray(np.rot90(base, 1))
            yield "rot270", np.ascontiguousarray(np.rot90(base, 3))

    def _decode(self, predictions: np.ndarray, variant: str) -> Candidate:
        probs = np.asarray(predictions)
        if probs.ndim == 2:
            probs = probs[None, ...]
        probs = probs[0]
        masked = probs.copy()
        if self.russian_only:
            for i, char in enumerate(self.characters, start=1):
                if char not in self.allowed and i < masked.shape[1]:
                    masked[:, i] = 0.0

        indices = masked.argmax(axis=1)
        selected = masked.max(axis=1)
        chars: list[str] = []
        confidence: list[float] = []
        last = -1
        for idx, probability in zip(indices, selected):
            idx = int(idx)
            if idx == 0 or idx == last:
                last = idx
                continue
            if idx - 1 < len(self.characters):
                chars.append(self.characters[idx - 1])
                confidence.append(float(probability))
            last = idx

        raw = "".join(chars)
        # Do not use a language model for the raw candidate; it is used later
        # only for resolving disagreements between visual passes.
        text = normalize_russian_text(raw, allow_dictionary=False)
        score = float(np.mean(confidence)) if confidence else 0.0
        return Candidate(text=text, raw_text=raw, confidence=score, variant=variant)

    def recognize_once(self, image: np.ndarray, *, variant: str = "raw") -> Candidate:
        batch = self._preprocess(image)[None, ...]
        predictions = self.session.run(None, {self.input_name: batch})[0]
        return self._decode(predictions, variant)

    def recognize_variants(self, image: np.ndarray, *, extended: bool = False) -> list[Candidate]:
        candidates: list[Candidate] = []
        for name, variant in self._variants(image, extended=extended):
            candidates.append(self.recognize_once(variant, variant=name))
        return candidates

    def recognize_consensus(
        self,
        image: np.ndarray,
        *,
        target_confidence: float = 0.90,
        max_passes: int = 4,
    ) -> dict:
        """Run bounded visual passes and return an honest confidence result."""
        all_candidates: list[Candidate] = []
        margins = [4, 6, 8, 10]
        for pass_no in range(max_passes):
            crop = self._trim(image, margin=margins[min(pass_no, len(margins) - 1)])
            candidates = self.recognize_variants(crop, extended=pass_no > 0)
            all_candidates.extend(candidates)

            # Normalize only for comparison; keep raw visual candidate in the
            # returned evidence list.
            groups: dict[str, list[Candidate]] = {}
            for candidate in all_candidates:
                key = normalize_russian_text(candidate.text, allow_dictionary=True)
                groups.setdefault(key, []).append(candidate)
            best_text, best_group = max(
                groups.items(),
                key=lambda item: (len(item[1]), max(c.confidence for c in item[1])),
            )
            model_score = float(np.mean([c.confidence for c in best_group]))
            agreement = len(best_group) / max(1, len(all_candidates))
            score = 0.70 * model_score + 0.30 * agreement
            if score >= target_confidence and len(best_group) >= 2:
                return {
                    "text": best_text,
                    "confidence": round(min(score, 0.999), 4),
                    "certain": True,
                    "passes": pass_no + 1,
                    "evidence": [c.__dict__ for c in all_candidates],
                }

        groups = {}
        for candidate in all_candidates:
            key = normalize_russian_text(candidate.text, allow_dictionary=True)
            groups.setdefault(key, []).append(candidate)
        best_text, best_group = max(
            groups.items(),
            key=lambda item: (len(item[1]), max(c.confidence for c in item[1])),
        )
        model_score = float(np.mean([c.confidence for c in best_group])) if best_group else 0.0
        agreement = len(best_group) / max(1, len(all_candidates))
        score = 0.70 * model_score + 0.30 * agreement
        return {
            "text": best_text,
            "confidence": round(min(score, 0.999), 4),
            "certain": False,
            "passes": max_passes,
            "evidence": [c.__dict__ for c in all_candidates],
        }
