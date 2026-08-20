from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import CyrillicOCR


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Russian/Cyrillic OCR")
    parser.add_argument("image", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--target", type=float, default=0.90, help="confidence threshold")
    parser.add_argument("--passes", type=int, default=4, help="maximum visual passes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ocr = CyrillicOCR(
        detector_path=root / "models/onnx/pp-ocrv4_mobile_det.onnx",
        recognizer_path=root / "models/onnx/cyrillic_pp-ocrv3_mobile_rec.onnx",
        dictionary_path=root / "models/dicts/cyrillic_dict.txt",
        target_confidence=args.target,
        max_passes=args.passes,
    )
    result = ocr.recognize_page(args.image).as_dict()
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])
        print(f"confidence={result['confidence']:.3f} certain={result['certain']} passes={result['passes']}")


if __name__ == "__main__":
    main()
