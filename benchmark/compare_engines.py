"""Compare our pipeline with EasyOCR, Tesseract and Manga OCR when available.

The benchmark is intentionally optional: Manga OCR is trained for Japanese
manga, so a Russian result is reported rather than silently treating it as a
fair Russian baseline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2

from ocr_rus_cyrillic.pipeline import CyrillicOCR


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def cer(expected: str, actual: str) -> float:
    a, b = norm(expected), norm(actual)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = current
    return row[-1] / max(1, len(a))


def run_tesseract(path: Path) -> str:
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("tesseract executable not found")
    result = subprocess.run([exe, str(path), "stdout", "-l", "rus", "--psm", "11"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("benchmark/pollination_pages/final"))
    parser.add_argument("--output", type=Path, default=Path("outputs/engine_comparison.json"))
    parser.add_argument("--easyocr", action="store_true")
    parser.add_argument("--manga-ocr", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ours = CyrillicOCR(
        detector_path=root / "models/onnx/pp-ocrv4_mobile_det.onnx",
        recognizer_path=root / "models/onnx/cyrillic_pp-ocrv3_mobile_rec.onnx",
        dictionary_path=root / "models/dicts/cyrillic_dict.txt",
        secondary_recognizer_path=root / "models/onnx/cyrillic_pp-ocrv5_mobile_rec.onnx",
        secondary_dictionary_path=root / "models/dicts/ppocrv5_cyrillic_dict.txt",
        target_confidence=0.90,
        max_passes=4,
    )
    easy = None
    if args.easyocr:
        try:
            import easyocr
            easy = easyocr.Reader(["ru"], gpu=False, verbose=False)
        except Exception as exc:
            print(f"EasyOCR unavailable: {exc}")
    manga = None
    if args.manga_ocr:
        try:
            from manga_ocr import MangaOcr
            manga = MangaOcr()
        except Exception as exc:
            print(f"Manga OCR unavailable: {exc}")

    manifest_path = args.input / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    expected = {str(x.get("page", i + 1)): x.get("text", "") for i, x in enumerate(manifest)}
    files = sorted(args.input.glob("*.png"))
    report: dict = {"input": str(args.input), "engines": {}}
    for engine in ("ours", "tesseract", "easyocr", "manga_ocr"):
        if engine == "easyocr" and easy is None:
            continue
        if engine == "manga_ocr" and manga is None:
            continue
        rows = []
        for index, path in enumerate(files, start=1):
            start = time.perf_counter()
            try:
                if engine == "ours":
                    result = ours.recognize_page(path)
                    text = result.text
                    confidence = result.confidence
                elif engine == "tesseract":
                    text = run_tesseract(path)
                    confidence = None
                elif engine == "easyocr":
                    parts = easy.readtext(str(path), detail=0, paragraph=True)
                    text = "\n".join(parts)
                    confidence = None
                else:
                    text = str(manga(str(path)))
                    confidence = None
                error = None
            except Exception as exc:
                text, confidence, error = "", None, repr(exc)
            row = {
                "page": index,
                "file": path.name,
                "text": text,
                "confidence": confidence,
                "latency_sec": round(time.perf_counter() - start, 3),
                "error": error,
            }
            if str(index) in expected:
                row["cer"] = cer(expected[str(index)], text)
            rows.append(row)
        values = [r["cer"] for r in rows if "cer" in r and r["error"] is None]
        report["engines"][engine] = {
            "pages": len(rows),
            "mean_cer": sum(values) / len(values) if values else None,
            "mean_latency_sec": sum(r["latency_sec"] for r in rows) / max(1, len(rows)),
            "results": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in value.items() if m != "results"} for k, value in report["engines"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
