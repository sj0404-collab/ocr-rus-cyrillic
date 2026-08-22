"""Build a complete, exactly-labelled Russian OCR dataset.

AI-generated backgrounds are used as visual variation, while every Russian
string is rendered locally from an OpenCorpora sentence file and its exact
transcript/bounding boxes are saved. This is still a synthetic image dataset;
Pollinations/image-generation cannot produce non-synthetic photographs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from synthetic_layout import (
    CORPUS,
    augment,
    font_candidates,
    load_font,
    paste_rotated,
    render_curved,
    render_horizontal,
    render_vertical,
    yolo_label,
)

PAGE_SIZE = (1024, 1024)


def load_sentences(path: Path | None) -> list[str]:
    if path is not None and path.exists():
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if len(line.strip()) >= 10]
        if values:
            return values
    return CORPUS


def load_art(path: Path | None, rng: random.Random) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("RGBA", PAGE_SIZE, (235, 239, 245, 255))
    files = sorted(path.glob("*_art.png"))
    if not files:
        return Image.new("RGBA", PAGE_SIZE, (235, 239, 245, 255))
    image = Image.open(rng.choice(files)).convert("RGB")
    image = ImageOps.fit(image, PAGE_SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return image.convert("RGBA")


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, min(width, x0)), max(0, min(height, y0)), max(0, min(width, x1)), max(0, min(height, y1))


def build_sample(index: int, out_dir: Path, sentences: list[str], fonts: list[Path], art_dir: Path | None, rng: random.Random) -> dict:
    image = load_art(art_dir, rng)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = PAGE_SIZE
    layout = rng.choice(["bubble", "paragraph", "columns", "rtl_columns", "vertical", "curved", "scene_text"])
    n = rng.randint(2, 5) if layout in {"bubble", "columns", "rtl_columns"} else rng.randint(1, 3)
    phrases = [rng.choice(sentences) for _ in range(n)]
    font_path = rng.choice(fonts)
    text_boxes: list[tuple[int, int, int, int]] = []
    bubble_boxes: list[tuple[int, int, int, int]] = []
    records: list[dict] = []

    for part, text in enumerate(phrases):
        size = rng.randint(20, 48)
        font = load_font(font_path, size)
        if layout == "vertical":
            tile = render_vertical(text[: min(len(text), 30)], font)
            direction = "ttb"
        elif layout == "curved":
            tile = render_curved(text[: min(len(text), 32)], font, radius=rng.randint(100, 220))
            direction = "ltr-curved"
        else:
            tile = render_horizontal(text, font, rng.randint(300, 700))
            direction = "rtl" if layout == "rtl_columns" else "ltr"

        if layout in {"columns", "rtl_columns"}:
            columns = n
            col_width = width // (columns + 1)
            x = (columns - part if layout == "rtl_columns" else part + 1) * col_width
            y = rng.randint(120, max(121, height - tile.height - 50))
        elif layout == "bubble":
            x = rng.randint(80, max(81, width - tile.width - 80))
            y = rng.randint(70, max(71, height - tile.height - 70))
            bx0, by0 = x - 24, y - 24
            bx1, by1 = x + tile.width + 24, y + tile.height + 24
            bubble = clamp_box((bx0, by0, bx1, by1), width, height)
            draw.rounded_rectangle(bubble, radius=35, fill=(255, 255, 255, 245), outline=(20, 20, 25, 255), width=4)
            bubble_boxes.append(bubble)
        else:
            x = rng.randint(70, max(71, width - tile.width - 70))
            y = rng.randint(70, max(71, height - tile.height - 70))
            if layout in {"paragraph", "scene_text"}:
                panel = clamp_box((x - 18, y - 18, x + tile.width + 18, y + tile.height + 18), width, height)
                draw.rounded_rectangle(panel, radius=12, fill=(255, 255, 255, 205), outline=(25, 25, 30, 220), width=2)

        angle = rng.uniform(-18, 18) if layout not in {"vertical", "curved"} else rng.uniform(-5, 5)
        box = paste_rotated(image, tile, (x + tile.width // 2, y + tile.height // 2), angle)
        box = clamp_box(box, width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        text_boxes.append(box)
        records.append({"text": text, "box": list(box), "direction": direction, "layout": layout, "font": str(font_path), "angle": angle})

    image = augment(image, rng).convert("RGB")
    split = "val" if index % 5 == 0 else "train"
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    crop_dir = out_dir / "recognition" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = f"page_{index:06d}"
    image_path = image_dir / f"{name}.jpg"
    image.save(image_path, quality=92)

    labels = [yolo_label(box, width, height, 0) for box in text_boxes]
    labels.extend(yolo_label(box, width, height, 1) for box in bubble_boxes)
    (label_dir / f"{name}.txt").write_text("\n".join(label for label in labels if label) + "\n", encoding="utf-8")

    recognition_rows: list[str] = []
    for line_no, (box, record) in enumerate(zip(text_boxes, records)):
        x0, y0, x1, y1 = clamp_box((box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8), width, height)
        crop = image.crop((x0, y0, x1, y1))
        crop_path = crop_dir / f"{name}_line_{line_no:02d}.png"
        crop.save(crop_path)
        recognition_rows.append(f"{crop_path.relative_to(out_dir).as_posix()}\t{record['text']}")
    return {
        "image": image_path.relative_to(out_dir).as_posix(),
        "split": split,
        "layout": layout,
        "transcript": "\n".join(record["text"] for record in records),
        "lines": records,
        "bubbles": [list(box) for box in bubble_boxes],
        "recognition_rows": recognition_rows,
        "source": "OpenCorpora text + original AI/procedural background",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/russian_manga_ocr_dataset"))
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--art-dir", type=Path, default=Path("benchmark/pollination_pages"))
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    fonts = font_candidates()
    if not fonts:
        raise SystemExit("No TTF/OTF fonts found")
    sentences = load_sentences(args.corpus)
    rng = random.Random(args.seed)
    manifest = [build_sample(i, args.output, sentences, fonts, args.art_dir if args.art_dir.exists() else None, rng) for i in range(args.samples)]
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("train", "val"):
        rows = [row for item in manifest if item["split"] == split for row in item["recognition_rows"]]
        (args.output / "recognition" / f"{split}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n  1: bubble\n",
        encoding="utf-8",
    )
    print(f"Generated {len(manifest)} fully-labelled pages and {sum(len(x['recognition_rows']) for x in manifest)} line crops")


if __name__ == "__main__":
    main()
