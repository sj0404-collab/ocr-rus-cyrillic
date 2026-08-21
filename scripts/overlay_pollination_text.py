"""Overlay exact Russian dialogue on original AI-generated empty-bubble art."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "benchmark/pollination_pages"
OUT = ART / "final"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Coordinates are for the generated original pages. Each tuple is x0,y0,x1,y1.
PAGES = {
    "page_01": [
        ((500, 70, 820, 190), "Ты тоже это видишь?"),
        ((485, 430, 820, 570), "Старая бумага светится!"),
        ((75, 760, 350, 855), "Не трогай её!"),
        ((75, 865, 350, 965), "Поздно..."),
        ((575, 760, 840, 920), "Она зовёт нас."),
    ],
    "page_02": [
        ((180, 55, 500, 190), "Я записала координаты."),
        ((75, 465, 380, 630), "Поезд уже ушёл."),
        ((245, 900, 700, 1075), "Значит, идём пешком."),
    ],
    "page_03": [
        ((80, 20, 330, 155), "Ты пришёл!"),
        ((250, 50, 540, 175), "Конечно. Я обещал."),
        ((610, 175, 785, 300), "Дождь стих."),
        ((80, 450, 350, 615), "Запись всё ещё здесь."),
        ((400, 445, 660, 575), "Начнём сначала."),
        ((70, 915, 350, 1090), "Город просыпается."),
        ((520, 920, 760, 1090), "История начинается."),
    ],
}


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
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
    return lines or [text]


def fit_font(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    max_width, max_height = x1 - x0 - 28, y1 - y0 - 24
    for size in range(34, 11, -1):
        font = ImageFont.truetype(FONT, size)
        lines = wrap_lines(draw, text, font, max_width)
        line_height = draw.textbbox((0, 0), "АБВ", font=font)[3] + 4
        if len(lines) * line_height <= max_height:
            return font, lines, line_height
    font = ImageFont.truetype(FONT, 12)
    return font, wrap_lines(draw, text, font, max_width), 16


def draw_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    x0, y0, x1, y1 = box
    # Repaint the bubble so the OCR input contains only the exact overlay text.
    draw.rounded_rectangle(box, radius=28, fill=(255, 255, 255), outline=(20, 20, 25), width=4)
    font, lines, line_height = fit_font(draw, text, box)
    total_height = len(lines) * line_height
    y = y0 + max(8, (y1 - y0 - total_height) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((x0 + x1 - tw) // 2, y - bbox[1]), line, fill=(10, 10, 15), font=font)
        y += line_height


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for page_index, (page_name, bubbles) in enumerate(PAGES.items(), start=1):
        image = Image.open(ART / f"{page_name}_art.png").convert("RGB")
        draw = ImageDraw.Draw(image)
        for box, text in bubbles:
            draw_text(draw, box, text)
        output = OUT / f"{page_name}.png"
        image.save(output, optimize=True)
        manifest.append({
            "page": page_index,
            "image": output.name,
            "text": "\n".join(text for _, text in bubbles),
            "bubbles": [{"box": list(box), "text": text} for box, text in bubbles],
        })
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(manifest)} original Russian manga-style pages in {OUT}")


if __name__ == "__main__":
    main()
