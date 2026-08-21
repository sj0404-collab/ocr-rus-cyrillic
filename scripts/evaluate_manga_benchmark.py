"""Run the page OCR pipeline over the generated synthetic chapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2

from ocr_rus_cyrillic.pipeline import CyrillicOCR


def levenshtein(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = current
    return row[-1]


def compact(value: str) -> str:
    return " ".join(value.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/manga_synthetic"))
    parser.add_argument("--output", type=Path, default=Path("outputs/manga_synthetic_report.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ocr = CyrillicOCR(
        detector_path=root / "models/onnx/pp-ocrv4_mobile_det.onnx",
        recognizer_path=root / "models/onnx/cyrillic_pp-ocrv3_mobile_rec.onnx",
        dictionary_path=root / "models/dicts/cyrillic_dict.txt",
        target_confidence=0.90,
        max_passes=4,
    )
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        result = ocr.recognize_page(args.input / item["image"])
        expected, actual = compact(item["text"]), compact(result.text)
        distance = levenshtein(expected, actual)
        row = {
            "page": item["page"],
            "expected": item["text"],
            "actual": result.text,
            "confidence": result.confidence,
            "certain": result.certain,
            "passes": result.passes,
            "cer": distance / max(1, len(expected)),
            "exact": expected == actual,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    report = {
        "pages": len(rows),
        "exact_pages": sum(row["exact"] for row in rows),
        "mean_cer": sum(row["cer"] for row in rows) / max(1, len(rows)),
        "certain_pages": sum(row["certain"] for row in rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
