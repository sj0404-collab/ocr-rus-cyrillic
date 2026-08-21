"""Generate diverse, fully licensed synthetic Russian OCR pages and YOLO labels.

The generator uses only locally rendered text and simple procedural textures.
It covers horizontal paragraphs, columns, vertical text, RTL columns, curved
text, speech bubbles, rotated/scaled text, punctuation, erosion, blur and
JPEG-like noise. Ground-truth text and bounding boxes are saved with every
sample.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CORPUS = [
    "Проверка русского текста: ёжик идёт домой, а часы показывают 07:30.",
    "Съешь ещё этих мягких французских булок, да выпей чаю!",
    "Объём, размер, ударение и пунктуация важны для точного OCR.",
    "Если движки не согласны, система должна попросить новый снимок.",
    "Сегодня мы читаем страницу слева направо, сверху вниз.",
    "В комиксе колонки могут идти справа налево, а текст — вертикально.",
    "Тихий дождь стучит по крыше; город просыпается постепенно.",
    "Ошибка в одной букве меняет смысл: был, бил, бел и бал.",
    "Номер главы: 12. Цена: 99,90 ₽. Версия: 2.4.1.",
    "«Начнём сначала», — сказала Аня. — «Но сохраним каждую строку». ",
    "Уверенность ниже 90 процентов означает: нужно проверить результат.",
    "Читаем абзац, отдельные слова, знаки тире, кавычки и многоточия…",
    "Разные шрифты, масштабы, наклоны и фон не должны ломать распознавание.",
    "Пять быстрых агентов сравнивают буквы, слова и порядок блоков.",
]


def font_candidates() -> list[Path]:
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype"),
        Path("/usr/share/fonts/opentype"),
        Path("/usr/local/share/fonts"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.ttf"))
            files.extend(root.rglob("*.otf"))
    preferred = [
        p for p in files
        if any(token in p.name.lower() for token in ("dejavu", "noto", "arial", "segoe", "times", "liberation"))
    ]
    return list(dict.fromkeys(preferred + files))


def load_font(path: Path, size: int):
    try:
        return ImageFont.truetype(str(path), size=size)
    except Exception:
        return ImageFont.load_default()


def make_texture(size: tuple[int, int], rng: random.Random) -> Image.Image:
    w, h = size
    base = np.zeros((h, w, 3), dtype=np.uint8)
    tone = rng.randint(220, 250)
    base[:] = tone
    if rng.random() < 0.6:
        noise = rng.normalvariate(0, 1)
        del noise
        arr = rng.random()
        del arr
        grain = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(3, 15), (h, w, 1))
        base = np.clip(base.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)
    if rng.random() < 0.5:
        for _ in range(rng.randint(3, 14)):
            y = rng.randint(0, h - 1)
            draw.line((0, y, w, y + rng.randint(-5, 5)), fill=(150, 150, 150), width=1)
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 1.3)))
    return image


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def render_horizontal(text: str, font, width: int, padding: int = 12) -> Image.Image:
    probe = Image.new("RGBA", (width, 50), (255, 255, 255, 0))
    draw = ImageDraw.Draw(probe)
    lines = wrap_text(draw, text, font, width - padding * 2)
    line_height = draw.textbbox((0, 0), "АБВ", font=font)[3] + 5
    height = padding * 2 + line_height * len(lines)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=(15, 15, 20, 255))
        y += line_height
    return image


def render_vertical(text: str, font, padding: int = 10) -> Image.Image:
    glyphs = [ch for ch in text if not ch.isspace()]
    size = font.getbbox("А")[3] + padding
    width = font.getbbox("А")[2] + padding * 2
    image = Image.new("RGBA", (max(30, width), max(40, size * len(glyphs) + padding)), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    y = padding
    for glyph in glyphs:
        bbox = draw.textbbox((0, 0), glyph, font=font)
        x = (image.width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), glyph, font=font, fill=(15, 15, 20, 255))
        y += size
    return image


def render_curved(text: str, font, radius: int = 150) -> Image.Image:
    chars = list(text)
    canvas = Image.new("RGBA", (radius * 2 + 100, radius + 100), (255, 255, 255, 0))
    cx, cy = canvas.width // 2, canvas.height + 20
    total = max(1, len(chars) - 1)
    for i, glyph in enumerate(chars):
        if glyph.isspace():
            continue
        angle = -55 + 110 * i / total
        tile = Image.new("RGBA", (80, 80), (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((8, 8), glyph, font=font, fill=(15, 15, 20, 255))
        tile = tile.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        theta = math.radians(angle - 90)
        x = int(cx + radius * math.cos(theta) - tile.width / 2)
        y = int(cy + radius * math.sin(theta) - tile.height / 2)
        canvas.alpha_composite(tile, (x, y))
    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else canvas


def paste_rotated(base: Image.Image, tile: Image.Image, center: tuple[int, int], angle: float) -> tuple[int, int, int, int]:
    tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    x = int(center[0] - tile.width / 2)
    y = int(center[1] - tile.height / 2)
    base.alpha_composite(tile, (x, y))
    return x, y, x + tile.width, y + tile.height


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    if rng.random() < 0.35:
        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.erode(arr, kernel, iterations=1)
    if rng.random() < 0.30:
        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.dilate(arr, kernel, iterations=1)
    if rng.random() < 0.55:
        sigma = rng.uniform(0, 1.5)
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
    if rng.random() < 0.65:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(1, 12), arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        quality = rng.randint(35, 90)
        ok, encoded = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            arr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def yolo_label(box: tuple[int, int, int, int], width: int, height: int, cls: int) -> str:
    x0, y0, x1, y1 = box
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return ""
    cx = ((x0 + x1) / 2) / width
    cy = ((y0 + y1) / 2) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def generate_sample(index: int, out_dir: Path, fonts: list[Path], corpus: list[str], rng: random.Random, width: int, height: int) -> dict:
    image = make_texture((width, height), rng).convert("RGBA")
    draw = ImageDraw.Draw(image)
    text_boxes: list[tuple[int, int, int, int]] = []
    bubble_boxes: list[tuple[int, int, int, int]] = []
    records: list[dict] = []
    layout = rng.choice(["horizontal", "paragraph", "columns", "rtl_columns", "vertical", "curved", "bubble"])
    count = rng.randint(2, 5) if layout in {"columns", "rtl_columns", "bubble"} else rng.randint(1, 3)
    chosen = [rng.choice(corpus) for _ in range(count)]
    font_path = rng.choice(fonts)
    direction = "rtl" if layout == "rtl_columns" else "ltr"

    if layout in {"columns", "rtl_columns"}:
        columns = count
        col_width = width // (columns + 1)
        for i, text in enumerate(chosen):
            size = rng.randint(22, 48)
            font = load_font(font_path, size)
            block = render_horizontal(text, font, col_width - 20)
            x = (columns - i if direction == "rtl" else i + 1) * col_width - block.width // 2
            y = rng.randint(80, max(90, height - block.height - 40))
            box = paste_rotated(image, block, (x, y), rng.uniform(-7, 7))
            text_boxes.append(box)
            records.append({"text": text, "box": box, "direction": direction, "layout": layout})
    elif layout == "vertical":
        for i, text in enumerate(chosen):
            size = rng.randint(24, 48)
            font = load_font(font_path, size)
            block = render_vertical(text, font)
            x = rng.randint(70, max(71, width - block.width - 70))
            y = rng.randint(80, max(81, height - block.height - 50))
            box = paste_rotated(image, block, (x + block.width // 2, y + block.height // 2), rng.choice([0, 0, 180]))
            text_boxes.append(box)
            records.append({"text": text, "box": box, "direction": "ttb", "layout": layout})
    elif layout == "curved":
        text = chosen[0]
        font = load_font(font_path, rng.randint(22, 42))
        block = render_curved(text, font, radius=rng.randint(110, 220))
        box = paste_rotated(image, block, (width // 2, height // 2), rng.uniform(-10, 10))
        text_boxes.append(box)
        records.append({"text": text, "box": box, "direction": "ltr-curved", "layout": layout})
    else:
        for i, text in enumerate(chosen):
            size = rng.randint(20, 52)
            font = load_font(font_path, size)
            max_width = rng.randint(int(width * 0.38), int(width * 0.78))
            block = render_horizontal(text, font, max_width)
            x = rng.randint(45, max(46, width - block.width - 45))
            y = rng.randint(50, max(51, height - block.height - 45))
            angle = rng.uniform(-15, 15)
            if layout == "bubble":
                bx0, by0 = max(10, x - 25), max(10, y - 25)
                bx1, by1 = min(width - 10, x + block.width + 25), min(height - 10, y + block.height + 25)
                draw.rounded_rectangle((bx0, by0, bx1, by1), radius=28, fill=(255, 255, 255, 245), outline=(25, 25, 30, 255), width=4)
                bubble_boxes.append((bx0, by0, bx1, by1))
            box = paste_rotated(image, block, (x + block.width // 2, y + block.height // 2), angle)
            text_boxes.append(box)
            records.append({"text": text, "box": box, "direction": "ltr", "layout": layout})

    image = augment(image, rng).convert("RGB")
    split = "val" if index % 5 == 0 else "train"
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    name = f"sample_{index:05d}"
    image_path = image_dir / f"{name}.jpg"
    image.save(image_path, quality=92)
    labels = [yolo_label(box, width, height, 0) for box in text_boxes]
    labels.extend(yolo_label(box, width, height, 1) for box in bubble_boxes)
    labels = [label for label in labels if label]
    (label_dir / f"{name}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    return {"image": str(image_path.relative_to(out_dir)), "text": records, "layout": layout, "font": str(font_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_russian_synthetic"))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--corpus", type=Path, default=None, help="UTF-8 Russian sentence file")
    args = parser.parse_args()
    fonts = font_candidates()
    if not fonts:
        raise SystemExit("No TTF/OTF fonts found")
    corpus = CORPUS
    if args.corpus is not None and args.corpus.exists():
        loaded = [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines() if len(line.strip()) >= 12]
        if loaded:
            corpus = loaded
    rng = random.Random(args.seed)
    manifest = [generate_sample(i, args.output, fonts, corpus, rng, args.width, args.height) for i in range(args.samples)]
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n  1: bubble\n",
        encoding="utf-8",
    )
    print(f"Generated {len(manifest)} samples using {len(fonts)} fonts in {args.output}")


if __name__ == "__main__":
    main()
