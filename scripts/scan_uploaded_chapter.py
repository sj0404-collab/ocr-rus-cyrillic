"""Scan a local, user-provided chapter folder page by page.

This script never downloads a website. It is intended for screenshots or
licensed pages supplied by the user. Browser chrome and bottom overlays can be
cropped with --top/--bottom before OCR.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

from ocr_rus_cyrillic.pipeline import CyrillicOCR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/uploaded_chapter_ocr"))
    parser.add_argument("--top", type=int, default=0, help="pixels to remove from the top")
    parser.add_argument("--bottom", type=int, default=0, help="bottom y coordinate; 0 keeps the full image")
    parser.add_argument("--yolo-model", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = sorted(
        [p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=natural_key,
    )
    if not pages:
        raise SystemExit(f"No images in {args.input}")

    root = Path(__file__).resolve().parents[1]
    ocr = CyrillicOCR(
        detector_path=root / "models/onnx/pp-ocrv4_mobile_det.onnx",
        recognizer_path=root / "models/onnx/cyrillic_pp-ocrv3_mobile_rec.onnx",
        dictionary_path=root / "models/dicts/cyrillic_dict.txt",
        secondary_recognizer_path=root / "models/onnx/cyrillic_pp-ocrv5_mobile_rec.onnx",
        secondary_dictionary_path=root / "models/dicts/ppocrv5_cyrillic_dict.txt",
        yolo_detector_path=args.yolo_model,
        target_confidence=0.90,
        max_passes=4,
    )
    results = []
    chapter_parts = []
    for number, page in enumerate(pages, start=1):
        image = cv2.imread(str(page))
        if image is None:
            continue
        top = max(0, min(args.top, image.shape[0] - 1))
        bottom = image.shape[0] if args.bottom <= 0 else min(args.bottom, image.shape[0])
        crop = image[top:bottom]
        crop_path = args.output / f"page_{number:03d}_crop.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result = ocr.recognize_page(crop)
        payload = {"page": number, "source": page.name, **result.as_dict()}
        results.append(payload)
        (args.output / f"page_{number:03d}.txt").write_text(result.text + "\n", encoding="utf-8")
        chapter_parts.append(f"\n===== СТРАНИЦА {number}: {page.name} =====\n{result.text}\n")
        print(f"page {number}/{len(pages)} confidence={result.confidence:.3f} certain={result.certain}")

    report = {"pages": len(results), "results": results}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "chapter.txt").write_text("".join(chapter_parts), encoding="utf-8")
    print(f"Wrote {len(results)} pages to {args.output}")


if __name__ == "__main__":
    main()
