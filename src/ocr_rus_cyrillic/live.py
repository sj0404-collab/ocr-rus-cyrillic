"""Low-latency live OCR session with duplicate-frame suppression."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .pipeline import CyrillicOCR, OCRResult


@dataclass(frozen=True)
class LiveOCRResult:
    text: str
    confidence: float
    certain: bool
    frame_id: int
    changed: bool
    analyzed: bool
    reused: bool
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class LiveOCRSession:
    """Process only changed frames and return the last text for duplicates.

    The signature is intentionally small and robust to camera JPEG noise. A
    changed frame is still allowed to produce the same text; the important
    invariant is that an identical frame is never sent to OCR twice unless
    ``force=True`` is used.
    """

    def __init__(
        self,
        ocr: CyrillicOCR,
        *,
        signature_size: tuple[int, int] = (48, 32),
        unchanged_delta: float = 0.008,
    ) -> None:
        self.ocr = ocr
        self.signature_size = signature_size
        self.unchanged_delta = unchanged_delta
        self._last_signature: np.ndarray | None = None
        self._last_result: LiveOCRResult | None = None
        self._frame_id = 0

    def _signature(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, self.signature_size, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32) / 255.0
        # Remove global exposure drift while retaining layout/text changes.
        small -= float(small.mean())
        scale = float(np.mean(np.abs(small))) + 1e-6
        return small / scale

    def process(self, frame: np.ndarray, *, force: bool = False) -> LiveOCRResult:
        started = time.perf_counter()
        self._frame_id += 1
        signature = self._signature(frame)
        same = (
            not force
            and self._last_signature is not None
            and float(np.mean(np.abs(signature - self._last_signature))) <= self.unchanged_delta
        )
        if same and self._last_result is not None:
            return LiveOCRResult(
                text=self._last_result.text,
                confidence=self._last_result.confidence,
                certain=self._last_result.certain,
                frame_id=self._frame_id,
                changed=False,
                analyzed=False,
                reused=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        result: OCRResult = self.ocr.recognize_page(frame)
        live = LiveOCRResult(
            text=result.text,
            confidence=result.confidence,
            certain=result.certain,
            frame_id=self._frame_id,
            changed=True,
            analyzed=True,
            reused=False,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        self._last_signature = signature
        self._last_result = live
        return live

    def reset(self) -> None:
        self._last_signature = None
        self._last_result = None
        self._frame_id = 0
