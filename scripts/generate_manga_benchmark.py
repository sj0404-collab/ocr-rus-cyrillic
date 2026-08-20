"""Generate a legal, synthetic Russian comic/manga-style OCR chapter.

It deliberately renders the ground-truth text itself instead of downloading a
copyrighted manga chapter. This produces reproducible speech bubbles and
stylized page layouts for OCR evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PHRASES = [
    "Привет! Это тест русской манги.",
    "Проверим ё, ъ, ы и щ.",
    "Текст должен совпасть без ошибок.",
    "Если сомневаешься, попроси проверку.",
    "Сегодня мы читаем страницу офлайн.",
    "Камера видит буквы даже в пузыре.",
    "Русский текст — только кириллица.",
    "Съешь ещё этих мягких булок.",
    "Точная строка важнее красивого процента.",
    "Повтори анализ, если шум слишком сильный.",
    "Объём и размер шрифта тоже важны.",
    "Финальная проверка: ошибка не скрывается.",
]


def find_font(bold: bool = False) -> str | None:
    candidates = []
    if bold:
        candidates += [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
    candidates += [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_font(size: int, bold: bool = False):
    path = find_font(bold)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered_wrapped(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font) -> None:
    x0, y0, x1, y1 = box
    max_width = x1 - x0 - 36
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), proposal, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = proposal
    if current:
        lines.append(current)
    line_height = draw.textbbox((0, 0), "АБВ", font=font)[3] + 8
    total = line_height * len(lines)
    y = y0 + max(12, (y1 - y0 - total) // 2)
    for line in lines:
        width = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((x0 + x1 - width) // 2, y), line, fill=(25, 25, 35), font=font)
        y += line_height


def make_page(page_no: int, out_dir: Path, rng: random.Random) -> dict:
    width, height = 900, 1200
    image = Image.new("RGB", (width, height), (235, 239, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, width - 25, height - 25), outline=(35, 45, 65), width=5)
    # No page number is rendered: the benchmark should measure dialogue OCR,
    # not a synthetic header that is absent from most manga pages.
    panel_boxes = [(55, 70, 845, 350), (55, 390, 845, 670), (55, 710, 845, 1070)]
    selected = [PHRASES[(page_no * 3 + i) % len(PHRASES)] for i in range(3)]
    bubbles: list[dict] = []
    for idx, (panel, phrase) in enumerate(zip(panel_boxes, selected)):
        x0, y0, x1, y1 = panel
        draw.rectangle(panel, fill=(220 + idx * 7, 225 + idx * 5, 235 + idx * 4), outline=(80, 90, 110), width=4)
        # Simple original line art, not copied from any comic.
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2 + 25
        draw.ellipse((cx - 90, cy - 75, cx + 90, cy + 105), outline=(80, 90, 120), width=5)
        draw.ellipse((cx - 35, cy - 15, cx - 15, cy + 5), fill=(40, 45, 65))
        draw.ellipse((cx + 15, cy - 15, cx + 35, cy + 5), fill=(40, 45, 65))
        bubble_w = min(640, max(420, draw.textbbox((0, 0), phrase, font=load_font(28))[2] + 70))
        bubble_h = 135 if len(phrase) < 32 else 175
        bx0 = x0 + 35 + (idx * 55) % max(1, (x1 - x0 - bubble_w - 70))
        by0 = y0 + 28
        bx1, by1 = bx0 + bubble_w, by0 + bubble_h
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=35, fill=(255, 255, 255), outline=(30, 35, 50), width=4)
        draw.polygon([(bx0 + 90, by1), (bx0 + 130, by1 + 35), (bx0 + 180, by1)], fill=(255, 255, 255), outline=(30, 35, 50))
        centered_wrapped(draw, phrase, (bx0, by0, bx1, by1), load_font(30 if len(phrase) < 30 else 26))
        bubbles.append({"text": phrase, "panel": idx + 1})

    path = out_dir / f"page_{page_no:03d}.png"
    image.save(path, optimize=True)
    return {"image": path.name, "page": page_no, "bubbles": bubbles, "text": "\n".join(b["text"] for b in bubbles)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/manga_synthetic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260820)
    manifest = [make_page(i, args.output, rng) for i in range(1, args.pages + 1)]
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} pages in {args.output}")


if __name__ == "__main__":
    main()
