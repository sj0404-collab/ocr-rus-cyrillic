"""Smoke test intended for a Windows GitHub Actions runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ocr_rus_cyrillic.pipeline import CyrillicOCR


ROOT = Path(__file__).resolve().parents[1]
ocr = CyrillicOCR(
    detector_path=ROOT / "models/onnx/pp-ocrv4_mobile_det.onnx",
    recognizer_path=ROOT / "models/onnx/cyrillic_pp-ocrv3_mobile_rec.onnx",
    dictionary_path=ROOT / "models/dicts/cyrillic_dict.txt",
    target_confidence=0.90,
    max_passes=4,
)
result = ocr.recognize_page(ROOT / "tests/fixtures/sample_word.png")
payload = result.as_dict()
(ROOT / "windows-smoke-result.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
if payload["text"] != "проверка":
    raise SystemExit(f"Unexpected OCR output: {payload['text']!r}")
if not payload["certain"]:
    raise SystemExit("The clean Windows smoke fixture did not reach the confidence threshold")
