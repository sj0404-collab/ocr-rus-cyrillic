"""Fine-tune a small YOLO detector on the generated Cyrillic layout dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/yolo_russian_synthetic/dataset.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_training"))
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained nano checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        cache=False,
        project=str(args.output.resolve()),
        name="cyrillic_text_yolo",
        exist_ok=True,
        pretrained=True,
        patience=0,
        plots=True,
        verbose=False,
    )
    best = args.output / "cyrillic_text_yolo" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    trained = YOLO(str(best))
    exported = trained.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
        nms=False,
    )
    print(f"best={best}")
    print(f"onnx={exported}")


if __name__ == "__main__":
    main()
